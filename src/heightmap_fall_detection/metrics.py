from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def calculate_metrics(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1_score = (
        2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0.0
    )
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "balanced_accuracy": float(balanced_accuracy),
    }


def get_video_prediction(scores: torch.Tensor, threshold: float = 0.5) -> int:
    if scores.nelement() == 0:
        return 0
    return int(torch.max(scores).item() >= threshold)

