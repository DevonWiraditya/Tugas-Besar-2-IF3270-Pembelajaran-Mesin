from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.data import CNNDataset, ImageRecord, load_cnn_dataset
from src.cnn.experiments import CNNExperiment, iter_cnn_experiments
from src.cnn.training import CNNImageSequence
from src.utils.io import read_json, write_csv, write_json

_PROJECT_ROOT = Path.cwd().resolve()


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics: dict[str, float]
    predictions: list[dict[str, object]]
    confusion: list[list[int]]


def evaluate_model_records(
    model: tf.keras.Model,
    records: Iterable[ImageRecord],
    config: CNNConfig,
    split_name: str,
    batch_size: int | None = None,
) -> EvaluationArtifacts:
    sequence = CNNImageSequence(
        records,
        config=config,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        shuffle=False,
        seed=config.seed,
    )
    probabilities = model.predict(sequence, verbose=0)
    labels = np.asarray([record.label for record in sequence.records], dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    loss, accuracy = model.evaluate(sequence, verbose=0)
    macro_f1 = f1_score(labels, predictions, average='macro')
    confusion = confusion_matrix(labels, predictions).tolist()
    rows = []
    project_root = Path.cwd().resolve()
    for record, predicted, distribution in zip(sequence.records, predictions, probabilities):
        record_path = _to_relative_path(record.path, project_root)
        rows.append(
            {
                'path': record_path,
                'class_name': record.class_name,
                'label': int(record.label),
                'predicted_label': int(predicted),
                'predicted_class_name': config.data.class_names[int(predicted)],
                'confidence': float(np.max(distribution)),
            }
        )

    metrics = {
        'split': split_name,
        'loss': float(loss),
        'sparse_categorical_accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
    }
    return EvaluationArtifacts(metrics=metrics, predictions=rows, confusion=confusion)


def save_evaluation_artifacts(
    artifacts: EvaluationArtifacts,
    output_dir: Path,
    split_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts.metrics, output_dir / f'metrics.{split_name}.json')
    _write_predictions_csv(artifacts.predictions, output_dir / f'{split_name}_predictions.csv')
    write_json({'matrix': artifacts.confusion}, output_dir / f'confusion.{split_name}.json')


def evaluate_saved_run(
    config: CNNConfig,
    run_dir: str | Path,
    split_name: str = 'test',
) -> dict[str, float]:
    run_path = Path(run_dir)
    dataset = load_cnn_dataset(config)
    records = _records_for_split(dataset, split_name)
    model = tf.keras.models.load_model(run_path / 'model.keras')
    artifacts = evaluate_model_records(model, records, config, split_name=split_name)
    save_evaluation_artifacts(artifacts, run_path, split_name)
    return artifacts.metrics


def evaluate_all_shared_runs(config: CNNConfig, split_name: str = 'test') -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for experiment in iter_cnn_experiments(config):
        run_dir = config.output.models_dir / experiment.run_id
        if not (run_dir / 'model.keras').exists():
            continue
        metrics = evaluate_saved_run(config, run_dir, split_name=split_name)
        results.append({'run_id': experiment.run_id, 'run_dir': _to_relative_run_dir(run_dir), **metrics})
    return results


def write_ranking(results: Iterable[dict[str, object]], path: Path) -> list[dict[str, object]]:
    ranked = sorted(results, key=lambda item: float(item['macro_f1']), reverse=True)
    write_csv(
        path,
        ['rank', 'run_id', 'run_dir', 'split', 'loss', 'sparse_categorical_accuracy', 'macro_f1'],
        [
            [index, row['run_id'], row['run_dir'], row['split'], row['loss'], row['sparse_categorical_accuracy'], row['macro_f1']]
            for index, row in enumerate(ranked, start=1)
        ],
    )
    return ranked


def select_best_run(ranking: Iterable[dict[str, object]], path: Path) -> dict[str, object]:
    ranked = list(ranking)
    if not ranked:
        raise ValueError('No completed runs available to select the best model.')
    best = ranked[0]
    write_json(best, path)
    return best


def plot_training_history(history_csv_path: str | Path, output_path: str | Path) -> None:
    history_path = Path(history_csv_path)
    if not history_path.exists():
        raise FileNotFoundError(history_path)

    with history_path.open('r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    epochs = [int(row['epoch']) for row in rows]
    loss = [float(row['loss']) for row in rows if row.get('loss')]
    val_loss = [float(row['val_loss']) for row in rows if row.get('val_loss')]
    accuracy = [float(row['sparse_categorical_accuracy']) for row in rows if row.get('sparse_categorical_accuracy')]
    val_accuracy = [float(row['val_sparse_categorical_accuracy']) for row in rows if row.get('val_sparse_categorical_accuracy')]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs[: len(loss)], loss, label='train_loss')
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs[: len(accuracy)], accuracy, label='train_accuracy')
    if val_accuracy:
        axes[1].plot(epochs[: len(val_accuracy)], val_accuracy, label='val_accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _write_predictions_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_csv(path, ['path', 'class_name', 'label', 'predicted_label', 'predicted_class_name', 'confidence'], [])
        return

    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _records_for_split(dataset: CNNDataset, split_name: str) -> tuple[ImageRecord, ...]:
    lookup = {
        'train': dataset.train,
        'validation': dataset.validation,
        'test': dataset.test,
    }
    try:
        return lookup[split_name]
    except KeyError as exc:
        raise ValueError(f'Unsupported split_name: {split_name}') from exc


def _to_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _to_relative_run_dir(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)
