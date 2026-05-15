from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.training import (
    iter_caption_experiments,
    train_caption_experiment,
    write_training_summary,
)
from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Flickr8k captioning experiments.")
    parser.add_argument("--config", default="configs/captioning/base.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--summary-path",
        default="reports/captioning/training_summary.csv",
    )
    parser.add_argument("--decoder-type", choices=["rnn", "lstm"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    preprocessing_dir = config.output.artifacts_dir / "preprocessing"
    train_sequences = np.load(preprocessing_dir / "train_sequences.npy")
    validation_sequences = np.load(preprocessing_dir / "validation_sequences.npy")
    train_features = np.load((config.output.artifacts_dir / "train_features").with_suffix(".npy"))
    validation_features = np.load(
        (config.output.artifacts_dir / "validation_features").with_suffix(".npy")
    )
    token_to_id = read_json(preprocessing_dir / "token_to_id.json")
    vocab_size = len(token_to_id)
    experiments = iter_caption_experiments(config)
    experiments = _filter_experiments(experiments, args.run_id, args.decoder_type)
    if args.limit is not None:
        experiments = experiments[: args.limit]

    results = []
    for experiment in experiments:
        result = train_caption_experiment(
            config,
            experiment,
            train_features,
            train_sequences,
            validation_features,
            validation_sequences,
            vocab_size,
            pad_id=token_to_id["<pad>"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )
        results.append(result)
        print(result)

    write_training_summary(results, args.summary_path)
    print(f"summary_path={args.summary_path}")


def _filter_experiments(experiments, run_ids, decoder_type):
    selected = tuple(experiments)
    if run_ids:
        allowed = set(run_ids)
        selected = tuple(experiment for experiment in selected if experiment.run_id in allowed)
    if decoder_type:
        selected = tuple(
            experiment
            for experiment in selected
            if experiment.recurrent_type == decoder_type
        )
    return selected


if __name__ == "__main__":
    main()
