from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.data import load_caption_dataset
from src.captioning.features import extract_split_features, save_split_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract Flickr8k image features.')
    parser.add_argument('--config', default='configs/captioning/base.json')
    parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    dataset = load_caption_dataset(config)
    split_dataset = getattr(dataset, args.split)
    features, image_ids = extract_split_features(split_dataset, config)
    save_split_features(features, image_ids, config.output.artifacts_dir / f'{args.split}_features')
    print(f'saved_split={args.split}')


if __name__ == '__main__':
    main()
