from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.data import load_caption_dataset
from src.captioning.preprocessing import build_vocabulary, prepare_split_sequences, save_vocabulary_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare Flickr8k caption sequences and vocabulary.')
    parser.add_argument('--config', default='configs/captioning/base.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    dataset = load_caption_dataset(config)
    vocabulary = build_vocabulary(dataset, config)
    save_vocabulary_artifacts(vocabulary, config)
    output_dir = config.output.artifacts_dir / 'preprocessing'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'train_sequences.npy', prepare_split_sequences(dataset.train.records, vocabulary, config))
    np.save(output_dir / 'validation_sequences.npy', prepare_split_sequences(dataset.validation.records, vocabulary, config))
    np.save(output_dir / 'test_sequences.npy', prepare_split_sequences(dataset.test.records, vocabulary, config))
    print(f'prepared={output_dir}')


if __name__ == '__main__':
    main()
