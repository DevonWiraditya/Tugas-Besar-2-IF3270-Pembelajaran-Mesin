from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import load_cnn_dataset
from src.cnn.experiments import get_cnn_experiment
from src.cnn.non_shared_models import build_non_shared_cnn_model
from src.cnn.training import CNNImageSequence
from src.utils.io import write_history_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a non-shared CNN run equivalent to a shared run.')
    parser.add_argument('--config', default='configs/cnn/base.json')
    parser.add_argument('--run-id', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    experiment = get_cnn_experiment(config, args.run_id)
    dataset = load_cnn_dataset(config)
    train_sequence = CNNImageSequence(dataset.train, config, config.training.batch_size, True, config.seed)
    validation_sequence = CNNImageSequence(dataset.validation, config, config.training.batch_size, False, config.seed)
    model = build_non_shared_cnn_model(config, experiment)
    run_dir = config.output.models_dir / f'{experiment.run_id}_non_shared'
    run_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(train_sequence, validation_data=validation_sequence, epochs=config.training.epochs, verbose=1)
    model.save(run_dir / 'model.keras')
    model.save_weights(run_dir / 'weights.weights.h5')
    write_history_csv(history.history, run_dir / 'history.csv')
    write_json(experiment.to_dict(), run_dir / 'experiment.json')
    print(f'run_dir={run_dir}')


if __name__ == '__main__':
    main()
