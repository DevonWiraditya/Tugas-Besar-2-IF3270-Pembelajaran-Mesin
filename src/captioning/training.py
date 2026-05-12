from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.captioning.config import CaptioningConfig
from src.captioning.keras_models import build_caption_decoder
from src.utils.io import write_history_csv, write_json


@dataclass(frozen=True)
class CaptionExperiment:
    recurrent_type: str
    layer_count: int
    hidden_size: int

    @property
    def run_id(self) -> str:
        return f'{self.recurrent_type}_layers{self.layer_count}_hidden{self.hidden_size}'


def iter_caption_experiments(config: CaptioningConfig) -> tuple[CaptionExperiment, ...]:
    experiments: list[CaptionExperiment] = []
    for recurrent_type in config.experiment_grid.recurrent_types:
        for layer_count in config.experiment_grid.layer_counts:
            for hidden_size in config.experiment_grid.hidden_sizes:
                experiments.append(CaptionExperiment(recurrent_type, layer_count, hidden_size))
    return tuple(experiments)


def build_training_targets(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sequences[:, :-1], sequences[:, -1]


def train_caption_experiment(config: CaptioningConfig, experiment: CaptionExperiment, train_features: np.ndarray, train_sequences: np.ndarray, validation_features: np.ndarray, validation_sequences: np.ndarray, vocab_size: int) -> dict[str, object]:
    train_inputs, train_targets = build_training_targets(train_sequences)
    validation_inputs, validation_targets = build_training_targets(validation_sequences)
    model = build_caption_decoder(config, experiment.recurrent_type, experiment.layer_count, experiment.hidden_size, vocab_size)
    run_dir = config.output.models_dir / experiment.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = model.fit([train_features, train_inputs], train_targets, validation_data=([validation_features, validation_inputs], validation_targets), epochs=config.training.epochs, batch_size=config.training.batch_size, verbose=1)
    duration = time.perf_counter() - started
    model.save(run_dir / 'model.keras')
    model.save_weights(run_dir / 'weights.weights.h5')
    write_history_csv(history.history, run_dir / 'history.csv')
    write_json({'runtime_seconds': duration}, run_dir / 'runtime.json')
    write_json({'run_id': experiment.run_id, 'recurrent_type': experiment.recurrent_type, 'layer_count': experiment.layer_count, 'hidden_size': experiment.hidden_size}, run_dir / 'experiment.json')
    return {'run_id': experiment.run_id, 'runtime_seconds': duration, 'run_dir': str(run_dir)}
