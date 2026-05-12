from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.data import load_cnn_dataset
from src.cnn.features import extract_features_for_paths, save_feature_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract frozen CNN features for a dataset split.')
    parser.add_argument('--config', default='configs/cnn/base.json')
    parser.add_argument('--split', default='test', choices=['train', 'validation', 'test', 'pred'])
    parser.add_argument('--output-prefix', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=True)
    dataset = load_cnn_dataset(config)
    if args.split == 'pred':
        paths = list(dataset.pred)
    else:
        records = getattr(dataset, args.split)
        paths = [record.path for record in records]
    features, ordered_paths = extract_features_for_paths(paths, config)
    output_prefix = args.output_prefix or str(config.output.artifacts_dir / f'{args.split}_features')
    save_feature_artifacts(features, ordered_paths, output_prefix)
    print(f'features={Path(output_prefix).with_suffix(".npy")}')


if __name__ == '__main__':
    main()
