from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from src.captioning.config import CaptioningConfig
from src.captioning.keras_models import build_caption_decoder
from src.utils.io import write_csv, write_history_csv, write_json


@dataclass(frozen=True)
class CaptionExperiment:
    recurrent_type: str
    layer_count: int
    hidden_size: int

    @property
    def run_id(self) -> str:
        return f"{self.recurrent_type}_preinject_layers{self.layer_count}_hidden{self.hidden_size}"


def iter_caption_experiments(
    config: CaptioningConfig,
) -> tuple[CaptionExperiment, ...]:
    experiments: list[CaptionExperiment] = []
    for recurrent_type in config.experiment_grid.recurrent_types:
        for layer_count in config.experiment_grid.layer_counts:
            for hidden_size in config.experiment_grid.hidden_sizes:
                experiments.append(
                    CaptionExperiment(
                        recurrent_type,
                        layer_count,
                        hidden_size,
                    )
                )
    return tuple(experiments)


def build_teacher_forcing_data(
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return sequences[:, :-1], sequences[:, 1:]


def build_sequence_sample_weights(
    targets: np.ndarray,
    pad_id: int,
) -> np.ndarray:
    return (targets != pad_id).astype(np.float32)


def train_caption_experiment(
    config: CaptioningConfig,
    experiment: CaptionExperiment,
    train_features: np.ndarray,
    train_sequences: np.ndarray,
    validation_features: np.ndarray,
    validation_sequences: np.ndarray,
    vocab_size: int,
    pad_id: int,
    epochs: int | None = None,
    batch_size: int | None = None,
    skip_existing: bool = False,
    verbose: int = 1,
) -> dict[str, object]:
    run_dir = config.output.models_dir / experiment.run_id
    if skip_existing and is_completed_run(run_dir):
        metrics_path = run_dir / "metrics.validation.json"
        metrics = {"runtime_seconds": 0.0}
        if metrics_path.exists():
            metrics.update(_read_json(metrics_path))
        return {
            "run_id": experiment.run_id,
            "decoder_type": experiment.recurrent_type,
            "layer_count": experiment.layer_count,
            "hidden_size": experiment.hidden_size,
            "runtime_seconds": float(metrics.get("runtime_seconds", 0.0)),
            "run_dir": str(run_dir),
            "status": "skipped",
        }

    train_inputs, train_targets = build_teacher_forcing_data(train_sequences)
    validation_inputs, validation_targets = build_teacher_forcing_data(
        validation_sequences
    )
    train_features = align_features_to_sequences(train_features, train_sequences)
    validation_features = align_features_to_sequences(
        validation_features,
        validation_sequences,
    )
    train_weights = build_sequence_sample_weights(train_targets, pad_id)
    validation_weights = build_sequence_sample_weights(validation_targets, pad_id)
    fit_epochs = config.training.epochs if epochs is None else epochs
    fit_batch_size = config.training.batch_size if batch_size is None else batch_size

    model = build_caption_decoder(
        config,
        experiment.recurrent_type,
        experiment.layer_count,
        experiment.hidden_size,
        vocab_size,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = model.fit(
        [train_features, train_inputs],
        train_targets,
        sample_weight=train_weights,
        validation_data=(
            [validation_features, validation_inputs],
            validation_targets,
            validation_weights,
        ),
        epochs=fit_epochs,
        batch_size=fit_batch_size,
        verbose=verbose,
    )
    duration = time.perf_counter() - started
    metrics = _extract_validation_metrics(history.history, duration)

    model.save(run_dir / "model.keras")
    model.save_weights(run_dir / "weights.weights.h5")
    write_history_csv(history.history, run_dir / "history.csv")
    write_json({"runtime_seconds": duration}, run_dir / "runtime.json")
    write_json(metrics, run_dir / "metrics.validation.json")
    write_json(
        {
            "run_id": experiment.run_id,
            "recurrent_type": experiment.recurrent_type,
            "layer_count": experiment.layer_count,
            "hidden_size": experiment.hidden_size,
        },
        run_dir / "experiment.json",
    )
    return {
        "run_id": experiment.run_id,
        "decoder_type": experiment.recurrent_type,
        "layer_count": experiment.layer_count,
        "hidden_size": experiment.hidden_size,
        "runtime_seconds": duration,
        "run_dir": str(run_dir),
        "status": "trained",
        "validation_loss": metrics.get("loss", 0.0),
        "validation_accuracy": metrics.get("sparse_categorical_accuracy", 0.0),
    }


def align_features_to_sequences(
    features: np.ndarray,
    sequences: np.ndarray,
) -> np.ndarray:
    if len(features) == len(sequences):
        return features
    if len(features) == 0 or len(sequences) == 0:
        return features
    if len(sequences) % len(features) != 0:
        raise ValueError(
            "Cannot align features to sequences: "
            f"{len(features)=}, {len(sequences)=}"
        )
    repeats = len(sequences) // len(features)
    return np.repeat(features, repeats, axis=0)


def is_completed_run(run_dir: Path) -> bool:
    required = (
        run_dir / "model.keras",
        run_dir / "weights.weights.h5",
        run_dir / "history.csv",
        run_dir / "runtime.json",
        run_dir / "experiment.json",
        run_dir / "metrics.validation.json",
    )
    return all(path.exists() for path in required)


def write_training_summary(results: Iterable[dict[str, object]], path: str | Path) -> None:
    rows = []
    for result in results:
        rows.append(
            [
                result["run_id"],
                result["decoder_type"],
                result["layer_count"],
                result["hidden_size"],
                result["status"],
                result["runtime_seconds"],
                _portable_path(result["run_dir"]),
                result.get("validation_loss", ""),
                result.get("validation_accuracy", ""),
            ]
        )
    write_csv(
        path,
        [
            "run_id",
            "decoder_type",
            "layer_count",
            "hidden_size",
            "status",
            "runtime_seconds",
            "model_dir",
            "validation_loss",
            "validation_accuracy",
        ],
        rows,
    )


def _extract_validation_metrics(
    history: dict[str, list[float]],
    runtime_seconds: float,
) -> dict[str, float]:
    metrics = {"runtime_seconds": float(runtime_seconds)}
    for key in ("val_loss", "val_sparse_categorical_accuracy"):
        values = history.get(key, [])
        if values:
            target_key = key.replace("val_", "")
            metrics[target_key] = float(values[-1])
    return metrics


def _portable_path(path: str | Path) -> str:
    value = Path(path)
    root = Path(__file__).resolve().parents[2]
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: str | Path) -> dict:
    import json

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
