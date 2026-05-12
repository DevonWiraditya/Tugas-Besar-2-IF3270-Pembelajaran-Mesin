from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import load_cnn_dataset, records_to_arrays
from src.cnn.experiments import get_cnn_experiment
from src.cnn.scratch.weights import load_shared_scratch_model
from src.utils.io import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a shared CNN scratch model loaded from Keras weights.')
    parser.add_argument('--config', default='configs/cnn/base.json')
    parser.add_argument('--run-id', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    experiment = get_cnn_experiment(config, args.run_id)
    run_dir = config.output.models_dir / experiment.run_id
    dataset = load_cnn_dataset(config)
    x_test, y_test = records_to_arrays(dataset.test, config)
    scratch_model = load_shared_scratch_model(config, experiment, run_dir / 'weights.weights.h5')
    probabilities = scratch_model.predict_proba(x_test)
    predictions = np.argmax(probabilities, axis=1)
    metrics = {'macro_f1': float(f1_score(y_test, predictions, average='macro'))}
    write_json(metrics, run_dir / 'scratch_metrics.json')
    rows = [[str(record.path), int(label), int(prediction)] for record, label, prediction in zip(dataset.test, y_test, predictions)]
    write_csv(run_dir / 'scratch_predictions.csv', ['path', 'label', 'predicted_label'], rows)
    print(metrics)


if __name__ == '__main__':
    main()
