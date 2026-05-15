from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate the captioning config.')
    parser.add_argument('--config', default='configs/captioning/base.json')
    parser.add_argument('--check-data', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=args.check_data)
    print(f'Config OK: {args.config}')
    print(f'Backbone: {config.encoder.backbone}')
    print(f'Max caption length: {config.text.max_caption_length}')


if __name__ == '__main__':
    main()
