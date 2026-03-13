from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the height-map fall detector.")
    parser.add_argument("--config", choices=["rfds", "urfd", "le2i"], required=True, help="Preset dataset config.")
    parser.add_argument("--root-dir", help="Override the height-map dataset root directory.")
    parser.add_argument("--output-root", default="./runs", help="Directory where training outputs are stored.")
    parser.add_argument("--epochs-per-fold", type=int, help="Override the number of epochs per fold.")
    parser.add_argument("--prediction-threshold", type=float, help="Override the video decision threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from heightmap_fall_detection.configs import get_dataset_config
    from heightmap_fall_detection.train import train_and_evaluate

    config = get_dataset_config(args.config)
    config["experiment_name"] = args.config

    if args.root_dir:
        config["root_dir"] = args.root_dir
    if args.epochs_per_fold is not None:
        config["epochs_per_fold"] = args.epochs_per_fold
    if args.prediction_threshold is not None:
        config["prediction_threshold"] = args.prediction_threshold

    result = train_and_evaluate(config=config, output_root=args.output_root)
    print(json.dumps({"output_dir": result.output_dir, "mean_metrics": result.mean_metrics}, indent=2))


if __name__ == "__main__":
    main()
