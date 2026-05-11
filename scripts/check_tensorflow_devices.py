from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.tensorflow_runtime import configure_tensorflow_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TensorFlow runtime devices.")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--disable-memory-growth", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = configure_tensorflow_runtime(
        use_gpu=not args.force_cpu,
        memory_growth=not args.disable_memory_growth,
        mixed_precision=args.mixed_precision,
    )
    print(f"Physical GPUs: {runtime['physical_gpus']}")
    print(f"Logical GPUs: {runtime['logical_gpus']}")
    print(f"Memory growth: {runtime['memory_growth']}")
    print(f"Mixed precision policy: {runtime['mixed_precision_policy']}")
    for error in runtime["errors"]:
        print(f"Runtime warning: {error}")


if __name__ == "__main__":
    main()
