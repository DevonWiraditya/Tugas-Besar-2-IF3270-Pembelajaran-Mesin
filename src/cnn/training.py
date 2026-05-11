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
from src.cnn.experiments import CNNExperiment, iter_cnn_experiments
from src.cnn.keras_models import build_shared_cnn_model, compile_cnn_model
from src.utils.io import write_history_csv, write_json
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
) -> dict:
    set_global_seed(config.seed)
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

    model = compile_cnn_model(build_shared_cnn_model(config, experiment), config)
    run_dir = config.output.models_dir / experiment.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=fit_epochs,
        verbose=1,
    )

    metrics = evaluate_sequence(model, validation_sequence)
    save_run_artifacts(model, history.history, metrics, experiment, config, run_dir)
    return {
        "run_id": experiment.run_id,
        "model_dir": str(run_dir),
        "metrics": metrics,
    }


def train_experiments(
    config: CNNConfig,
    experiments: Iterable[CNNExperiment] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
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
            )
        )

    return results


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
) -> None:
    model.save(run_dir / "model.keras")
    model.save_weights(run_dir / "weights.weights.h5")
    write_history_csv(history, run_dir / "history.csv")
    write_json(metrics, run_dir / "metrics.json")
    write_json(experiment.to_dict(), run_dir / "experiment.json")
    write_json(
        {
            "input_shape": config.input_shape,
            "class_names": config.data.class_names,
            "normalization": config.data.normalization,
            "data_format": config.data.data_format,
            "conv_padding": config.architecture_defaults.conv_padding,
            "conv_strides": config.architecture_defaults.conv_strides,
            "pool_padding": config.architecture_defaults.pool_padding,
            "pool_size": config.architecture_defaults.pool_size,
            "pool_strides": config.architecture_defaults.pool_strides,
        },
        run_dir / "contract.json",
    )
