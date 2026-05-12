from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.data import ImageRecord, load_cnn_dataset, records_to_arrays
from src.cnn.experiments import (
    CNNExperiment,
    iter_cnn_experiments,
    run_id_for_parameter_sharing,
)
from src.cnn.keras_models import build_cnn_model, compile_cnn_model
from src.utils.io import read_json, write_csv, write_history_csv, write_json
from src.utils.random import set_global_seed


class CNNImageSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        records: Iterable[ImageRecord],
        config: CNNConfig,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.records = list(records)
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.indexes = list(range(len(self.records)))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_records = [self.records[item] for item in batch_indexes]
        return records_to_arrays(batch_records, self.config)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(
            [self.records[index].label for index in self.indexes],
            dtype=np.int64,
        )


def train_experiment(
    config: CNNConfig,
    experiment: CNNExperiment,
    epochs: int | None = None,
    batch_size: int | None = None,
    skip_existing: bool = False,
    parameter_sharing: str = "shared",
) -> dict:
    set_global_seed(config.seed)
    run_id = run_id_for_parameter_sharing(experiment, parameter_sharing)
    run_dir = config.output.models_dir / run_id

    if skip_existing and is_completed_run(run_dir):
        metrics = read_json(run_dir / "metrics.json")
        return {
            "run_id": run_id,
            "base_run_id": experiment.run_id,
            "parameter_sharing": parameter_sharing,
            "model_dir": str(run_dir),
            "metrics": metrics,
            "status": "skipped",
        }

    dataset = load_cnn_dataset(config)
    fit_epochs = config.training.epochs if epochs is None else epochs
    fit_batch_size = config.training.batch_size if batch_size is None else batch_size

    train_sequence = CNNImageSequence(
        dataset.train,
        config=config,
        batch_size=fit_batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_sequence = CNNImageSequence(
        dataset.validation,
        config=config,
        batch_size=fit_batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = compile_cnn_model(
        build_cnn_model(config, experiment, parameter_sharing=parameter_sharing),
        config,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=fit_epochs,
        verbose=1,
    )

    metrics = evaluate_sequence(model, validation_sequence)
    save_run_artifacts(
        model,
        history.history,
        metrics,
        experiment,
        config,
        run_dir,
        parameter_sharing=parameter_sharing,
    )
    return {
        "run_id": run_id,
        "base_run_id": experiment.run_id,
        "parameter_sharing": parameter_sharing,
        "model_dir": str(run_dir),
        "metrics": metrics,
        "status": "trained",
    }


def train_experiments(
    config: CNNConfig,
    experiments: Iterable[CNNExperiment] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    skip_existing: bool = False,
    parameter_sharing: str = "shared",
) -> list[dict]:
    selected = iter_cnn_experiments(config) if experiments is None else tuple(experiments)
    results: list[dict] = []

    for experiment in selected:
        results.append(
            train_experiment(
                config,
                experiment,
                epochs=epochs,
                batch_size=batch_size,
                skip_existing=skip_existing,
                parameter_sharing=parameter_sharing,
            )
        )

    return results


def is_completed_run(run_dir: Path) -> bool:
    required = (
        run_dir / "weights.weights.h5",
        run_dir / "history.csv",
        run_dir / "metrics.json",
        run_dir / "experiment.json",
        run_dir / "contract.json",
    )
    return all(path.exists() for path in required)


def evaluate_sequence(
    model: tf.keras.Model,
    sequence: CNNImageSequence,
) -> dict[str, float]:
    probabilities = model.predict(sequence, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = sequence.labels
    loss, accuracy = model.evaluate(sequence, verbose=0)
    return {
        "loss": float(loss),
        "sparse_categorical_accuracy": float(accuracy),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_run_artifacts(
    model: tf.keras.Model,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    experiment: CNNExperiment,
    config: CNNConfig,
    run_dir: Path,
    parameter_sharing: str = "shared",
) -> None:
    run_id = run_id_for_parameter_sharing(experiment, parameter_sharing)
    experiment_payload = experiment.to_dict()
    experiment_payload["run_id"] = run_id
    experiment_payload["base_run_id"] = experiment.run_id
    experiment_payload["parameter_sharing"] = parameter_sharing

    if parameter_sharing == "shared":
        model.save(run_dir / "model.keras")
    model.save_weights(run_dir / "weights.weights.h5")
    write_history_csv(history, run_dir / "history.csv")
    write_json(metrics, run_dir / "metrics.json")
    write_json(experiment_payload, run_dir / "experiment.json")
    write_json(
        {
            "input_shape": config.input_shape,
            "class_names": config.data.class_names,
            "normalization": config.data.normalization,
            "data_format": config.data.data_format,
            "parameter_sharing": parameter_sharing,
            "conv_padding": config.architecture_defaults.conv_padding,
            "conv_strides": config.architecture_defaults.conv_strides,
            "pool_padding": config.architecture_defaults.pool_padding,
            "pool_size": config.architecture_defaults.pool_size,
            "pool_strides": config.architecture_defaults.pool_strides,
        },
        run_dir / "contract.json",
    )


def write_training_summary(results: Iterable[dict], path: Path) -> None:
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            [
                result["run_id"],
                result.get("base_run_id", result["run_id"]),
                result.get("parameter_sharing", "shared"),
                result.get("status", ""),
                _portable_path(result["model_dir"]),
                metrics.get("loss", ""),
                metrics.get("sparse_categorical_accuracy", ""),
                metrics.get("macro_f1", ""),
            ]
        )

    write_csv(
        path,
        [
            "run_id",
            "base_run_id",
            "parameter_sharing",
            "status",
            "model_dir",
            "loss",
            "sparse_categorical_accuracy",
            "macro_f1",
        ],
        rows,
    )


def _portable_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return value.as_posix()
