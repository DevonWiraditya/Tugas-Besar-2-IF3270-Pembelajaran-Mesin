from __future__ import annotations

import csv
import os
from pathlib import Path
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.data import iter_record_batches, load_cnn_dataset
from src.cnn.scratch_model import build_scratch_model_from_keras
from src.utils.io import write_csv


def compare_keras_and_scratch(
    config: CNNConfig,
    run_id: str | None = None,
    summary_path: str | Path | None = None,
    split: str = "test",
    batch_size: int | None = None,
    max_batches: int | None = None,
    max_samples: int | None = None,
) -> dict[str, float | int | str]:
    selected_run_id = run_id or select_best_run_id(
        Path(summary_path) if summary_path else config.output.reports_dir / "shared_training_summary.csv"
    )
    model_path = config.output.models_dir / selected_run_id / "model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    keras_model = tf.keras.models.load_model(model_path, compile=False)
    scratch_model = build_scratch_model_from_keras(keras_model)
    records = _split_records(config, split)
    if max_samples is not None:
        records = records[:max_samples]

    size = config.training.batch_size if batch_size is None else batch_size
    started_at = time.perf_counter()
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_values = 0
    batches = 0
    keras_predictions: list[int] = []
    scratch_predictions: list[int] = []
    labels: list[int] = []

    for index, (x, y) in enumerate(
        iter_record_batches(records, config, batch_size=size, shuffle=False)
    ):
        if max_batches is not None and index >= max_batches:
            break

        keras_probabilities = keras_model.predict(x, verbose=0)
        scratch_probabilities = scratch_model.forward(x)
        diff = np.abs(keras_probabilities - scratch_probabilities)

        max_abs_diff = max(max_abs_diff, float(np.max(diff)))
        total_abs_diff += float(np.sum(diff))
        total_values += int(diff.size)
        batches += 1
        keras_predictions.extend(np.argmax(keras_probabilities, axis=1).tolist())
        scratch_predictions.extend(np.argmax(scratch_probabilities, axis=1).tolist())
        labels.extend(y.tolist())

    if not labels:
        raise ValueError("No records were evaluated.")

    class_labels = list(range(len(config.data.class_names)))
    keras_array = np.asarray(keras_predictions)
    scratch_array = np.asarray(scratch_predictions)
    labels_array = np.asarray(labels)

    return {
        "run_id": selected_run_id,
        "split": split,
        "samples": len(labels),
        "batches": batches,
        "batch_size": size,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": total_abs_diff / total_values,
        "prediction_agreement": float(np.mean(keras_array == scratch_array)),
        "keras_accuracy": float(np.mean(keras_array == labels_array)),
        "scratch_accuracy": float(np.mean(scratch_array == labels_array)),
        "keras_macro_f1": float(
            f1_score(
                labels,
                keras_predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "scratch_macro_f1": float(
            f1_score(
                labels,
                scratch_predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "seconds": time.perf_counter() - started_at,
        "model_path": _portable_path(model_path),
    }


def select_best_run_id(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file, pass --run-id instead: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Summary file is empty: {path}")

    best = max(rows, key=lambda row: float(row["macro_f1"]))
    return best["run_id"]


def write_comparison_report(result: dict[str, float | int | str], path: str | Path) -> None:
    header = [
        "run_id",
        "split",
        "samples",
        "batches",
        "batch_size",
        "keras_macro_f1",
        "scratch_macro_f1",
        "keras_accuracy",
        "scratch_accuracy",
        "prediction_agreement",
        "max_abs_diff",
        "mean_abs_diff",
        "seconds",
        "model_path",
    ]
    write_csv(path, header, [[result[key] for key in header]])


def _split_records(config: CNNConfig, split: str):
    dataset = load_cnn_dataset(config)
    if split == "train":
        return dataset.train
    if split == "validation":
        return dataset.validation
    if split == "test":
        return dataset.test
    raise ValueError(f"Unsupported split: {split}")


def _portable_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return value.as_posix()
