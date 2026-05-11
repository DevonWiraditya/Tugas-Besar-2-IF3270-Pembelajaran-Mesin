from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CNN experiment config.")
    parser.add_argument(
        "--config",
        default="configs/cnn/base.json",
        help="Path to the CNN config JSON file.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Also check that configured dataset directories and class folders exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=args.check_data)
    print(f"Config OK: {args.config}")
    print(f"Classes: {', '.join(config.data.class_names)}")
    print(f"Input shape: {config.input_shape}")
    print(f"Experiment count: {config.experiment_count}")


if __name__ == "__main__":
    main()
