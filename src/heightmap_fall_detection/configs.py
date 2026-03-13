from __future__ import annotations

from copy import deepcopy


DEFAULT_DATASET_CONFIGS = {
    "rfds": {
        "root_dir": "./datasets/rfds",
        "window_size": 32,
        "stride": 8,
        "feature_dim": 64,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.10293366761443801,
        "learning_rate": 0.0002677396098899791,
        "epochs_per_fold": 50,
        "k_folds": 5,
        "prediction_threshold": 0.3109948076923432,
        "validation_split": 0.1,
        "seed": 42,
        "rho2": 0.10620104991055715,
        "rho3": 0.2962899925879182,
        "rho4": 0.48127190203512554,
        "rho5": 0.01807141433476636,
        "epsilon": 0.15608450793510792,
        "k_pos": 2,
        "k_neg": 1,
        "positive_dir": "Fall",
        "negative_dir": "NonFall",
    },
    "urfd": {
        "root_dir": "./datasets/urfd",
        "window_size": 32,
        "stride": 8,
        "feature_dim": 64,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1708469314798053,
        "learning_rate": 0.00048763731636263365,
        "epochs_per_fold": 50,
        "k_folds": 5,
        "prediction_threshold": 0.39005881743102017,
        "validation_split": 0.1,
        "seed": 42,
        "rho2": 0.01629167182577343,
        "rho3": 0.44106786503367357,
        "rho4": 0.10840560844395929,
        "rho5": 0.021411855321920765,
        "epsilon": 0.39694843175785877,
        "k_pos": 7,
        "k_neg": 4,
        "positive_dir": "Fall",
        "negative_dir": "NonFall",
    },
    "le2i": {
        "root_dir": "./datasets/le2i",
        "window_size": 32,
        "stride": 16,
        "feature_dim": 128,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.2146120173378149,
        "learning_rate": 8.317086141180386e-05,
        "epochs_per_fold": 50,
        "k_folds": 5,
        "prediction_threshold": 0.5448309948407031,
        "validation_split": 0.2,
        "seed": 42,
        "rho2": 0.18672836513567503,
        "rho3": 0.1920460978240891,
        "rho4": 0.297712866727257,
        "rho5": 0.005334206197577794,
        "epsilon": 0.27721011572227944,
        "k_pos": 5,
        "k_neg": 4,
        "positive_dir": "Fall",
        "negative_dir": "NonFall",
    },
}


def get_dataset_config(name: str) -> dict:
    key = name.lower()
    if key not in DEFAULT_DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset config: {name}")
    return deepcopy(DEFAULT_DATASET_CONFIGS[key])

