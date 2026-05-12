from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.evaluation import compare_keras_and_scratch, write_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Keras CNN output with NumPy scratch forward.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    result = compare_keras_and_scratch(
        config,
        run_id=args.run_id,
        summary_path=args.summary_path,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
    )

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.8f}")
        else:
            print(f"{key}={value}")

    if args.output_path:
        write_comparison_report(result, args.output_path)
        print(f"output_path={args.output_path}")

    if result["max_abs_diff"] > args.tolerance:
        raise SystemExit(f"max_abs_diff exceeded tolerance {args.tolerance}")


if __name__ == "__main__":
    main()
