from __future__ import annotations

import datetime as dt
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split

from .data import FallDetectionDataset, custom_collate_fn
from .losses import MILFallLoss
from .metrics import calculate_metrics, get_video_prediction
from .model import FallDetectionTransformer


@dataclass
class TrainingResult:
    output_dir: str
    fold_metrics: list[dict[str, float]]
    mean_metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _build_model(config: dict, device: torch.device) -> FallDetectionTransformer:
    return FallDetectionTransformer(
        feature_dim=config["feature_dim"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device)


def _build_loss(config: dict) -> MILFallLoss:
    return MILFallLoss(
        rho2=config["rho2"],
        rho3=config["rho3"],
        rho4=config["rho4"],
        rho5=config["rho5"],
        epsilon=config["epsilon"],
        k_pos=config["k_pos"],
        k_neg=config["k_neg"],
    )


def train_and_evaluate(config: dict, output_root: str = "./runs") -> TrainingResult:
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = FallDetectionDataset(
        root_dir=config["root_dir"],
        window_size=config["window_size"],
        stride=config["stride"],
        is_train=True,
        positive_dir=config.get("positive_dir", "Fall"),
        negative_dir=config.get("negative_dir", "NonFall"),
    )
    eval_dataset = FallDetectionDataset(
        root_dir=config["root_dir"],
        window_size=config["window_size"],
        stride=config["stride"],
        is_train=False,
        positive_dir=config.get("positive_dir", "Fall"),
        negative_dir=config.get("negative_dir", "NonFall"),
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"{config.get('experiment_name', 'experiment')}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    fold_metrics: list[dict[str, float]] = []

    for fold_index, (train_val_indices, test_indices) in enumerate(kf.split(train_dataset), start=1):
        train_val_subset = Subset(train_dataset, train_val_indices)
        val_size = int(len(train_val_subset) * config["validation_split"])
        train_size = len(train_val_subset) - val_size
        train_subset, val_subset = random_split(train_val_subset, [train_size, val_size], generator=generator)
        test_subset = Subset(eval_dataset, test_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        model = _build_model(config, device)
        criterion = _build_loss(config)
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_model_path = os.path.join(output_dir, f"best_fold_{fold_index}.pt")
        best_val_loss = float("inf")

        for _ in range(config["epochs_per_fold"]):
            model.train()
            for windows, label in train_loader:
                if windows.nelement() == 0:
                    continue
                windows = windows.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                scores = model(windows)
                loss = criterion(scores, label)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for windows, label in val_loader:
                    if windows.nelement() == 0:
                        continue
                    windows = windows.to(device)
                    label = label.to(device)
                    scores = model(windows)
                    val_loss_sum += criterion(scores, label).item()
                    val_steps += 1

            avg_val_loss = val_loss_sum / max(val_steps, 1)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        labels: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for windows, label in test_loader:
                if windows.nelement() == 0:
                    continue
                scores = model(windows.to(device))
                predictions.append(get_video_prediction(scores, threshold=config["prediction_threshold"]))
                labels.append(int(label.item()))

        metrics = calculate_metrics(labels, predictions)
        metrics["fold"] = float(fold_index)
        fold_metrics.append(metrics)

    metric_keys = [key for key in fold_metrics[0].keys() if key != "fold"]
    mean_metrics = {key: float(np.mean([fold[key] for fold in fold_metrics])) for key in metric_keys}

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"config": config, "fold_metrics": fold_metrics, "mean_metrics": mean_metrics},
            file,
            indent=2,
        )

    return TrainingResult(output_dir=output_dir, fold_metrics=fold_metrics, mean_metrics=mean_metrics)

