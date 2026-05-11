from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import count_by_class, iter_record_batches, load_cnn_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the configured CNN dataset.")
    parser.add_argument(
        "--config",
        default="configs/cnn/base.json",
        help="Path to the CNN config JSON file.",
    )
    parser.add_argument(
        "--sample-batch",
        action="store_true",
        help="Load one training batch and print its tensor shapes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    dataset = load_cnn_dataset(config)

    print(f"Classes: {', '.join(dataset.class_names)}")
    print(f"Train: {len(dataset.train)}")
    _print_counts(dataset.train, dataset.class_names)
    print(f"Validation: {len(dataset.validation)}")
    _print_counts(dataset.validation, dataset.class_names)
    print(f"Test: {len(dataset.test)}")
    _print_counts(dataset.test, dataset.class_names)
    print(f"Prediction images: {len(dataset.pred)}")

    if args.sample_batch:
        x, y = next(iter_record_batches(dataset.train, config, shuffle=False))
        print(f"Sample batch X shape: {x.shape}, dtype: {x.dtype}")
        print(f"Sample batch y shape: {y.shape}, dtype: {y.dtype}")
        print(f"Sample batch pixel range: [{x.min():.4f}, {x.max():.4f}]")


def _print_counts(records, class_names) -> None:
    counts = count_by_class(records, class_names)
    joined = ", ".join(f"{name}={counts[name]}" for name in class_names)
    print(f"  {joined}")


if __name__ == "__main__":
    main()
