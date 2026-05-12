from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.training import iter_caption_experiments, train_caption_experiment
from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Flickr8k captioning experiments.')
    parser.add_argument('--config', default='configs/captioning/base.json')
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    preprocessing_dir = config.output.artifacts_dir / 'preprocessing'
    train_sequences = np.load(preprocessing_dir / 'train_sequences.npy')
    validation_sequences = np.load(preprocessing_dir / 'validation_sequences.npy')
    train_features = np.load((config.output.artifacts_dir / 'train_features').with_suffix('.npy'))
    validation_features = np.load((config.output.artifacts_dir / 'validation_features').with_suffix('.npy'))
    token_to_id = read_json(preprocessing_dir / 'token_to_id.json')
    vocab_size = len(token_to_id)
    experiments = iter_caption_experiments(config)
    if args.limit is not None:
        experiments = experiments[: args.limit]
    for experiment in experiments:
        print(train_caption_experiment(config, experiment, train_features, train_sequences, validation_features, validation_sequences, vocab_size))


if __name__ == '__main__':
    main()
