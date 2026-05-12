from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import iter_record_batches, load_cnn_dataset
from src.cnn.scratch_model import build_scratch_model_from_keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Keras CNN output with NumPy scratch forward.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    run_id = args.run_id or select_best_run_id(Path(args.summary_path))
    model_path = config.output.models_dir / run_id / "model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    keras_model = tf.keras.models.load_model(model_path, compile=False)
    scratch_model = build_scratch_model_from_keras(keras_model)
    dataset = load_cnn_dataset(config)
    records = getattr(dataset, args.split)

    max_abs_diff = 0.0
    mean_abs_diffs: list[float] = []
    keras_predictions: list[int] = []
    scratch_predictions: list[int] = []
    labels: list[int] = []

    for index, (x, y) in enumerate(
        iter_record_batches(records, config, batch_size=args.batch_size, shuffle=False)
    ):
        if index >= args.max_batches:
            break

        keras_probabilities = keras_model.predict(x, verbose=0)
        scratch_probabilities = scratch_model.forward(x)
        diff = np.abs(keras_probabilities - scratch_probabilities)

        max_abs_diff = max(max_abs_diff, float(np.max(diff)))
        mean_abs_diffs.append(float(np.mean(diff)))
        keras_predictions.extend(np.argmax(keras_probabilities, axis=1).tolist())
        scratch_predictions.extend(np.argmax(scratch_probabilities, axis=1).tolist())
        labels.extend(y.tolist())

    if not labels:
        raise ValueError("No records were evaluated.")

    mean_abs_diff = float(np.mean(mean_abs_diffs))
    agreement = float(np.mean(np.asarray(keras_predictions) == np.asarray(scratch_predictions)))
    scratch_macro_f1 = f1_score(
        labels,
        scratch_predictions,
        labels=list(range(len(config.data.class_names))),
        average="macro",
        zero_division=0,
    )

    print(f"run_id={run_id}")
    print(f"split={args.split}")
    print(f"samples={len(labels)}")
    print(f"max_abs_diff={max_abs_diff:.8f}")
    print(f"mean_abs_diff={mean_abs_diff:.8f}")
    print(f"prediction_agreement={agreement:.4f}")
    print(f"scratch_macro_f1={scratch_macro_f1:.4f}")

    if max_abs_diff > args.tolerance:
        raise SystemExit(f"max_abs_diff exceeded tolerance {args.tolerance}")


def select_best_run_id(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file, pass --run-id instead: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Summary file is empty: {path}")

    best = max(rows, key=lambda row: float(row["macro_f1"]))
    return best["run_id"]


if __name__ == "__main__":
    main()
