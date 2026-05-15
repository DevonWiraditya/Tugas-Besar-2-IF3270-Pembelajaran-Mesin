from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.data import Vocabulary, load_caption_dataset
from src.captioning.evaluation import evaluate_caption_model, save_caption_metrics
from src.utils.io import read_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained captioning run.")
    parser.add_argument("--config", default="configs/captioning/base.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-summary-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--skip-model-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    run_dir = config.output.models_dir / args.run_id
    preprocessing_dir = config.output.artifacts_dir / "preprocessing"
    token_to_id = read_json(preprocessing_dir / "token_to_id.json")
    id_to_token_raw = read_json(preprocessing_dir / "id_to_token.json")
    vocabulary = Vocabulary(
        token_to_id=token_to_id,
        id_to_token={int(key): value for key, value in id_to_token_raw.items()},
        pad_id=token_to_id["<pad>"],
        start_id=token_to_id["<start>"],
        end_id=token_to_id["<end>"],
        oov_id=token_to_id["<unk>"],
    )
    dataset = load_caption_dataset(config)
    split_dataset = getattr(dataset, args.split)
    features = np.load((config.output.artifacts_dir / f"{args.split}_features").with_suffix(".npy"))
    image_ids = list(split_dataset.image_ids)
    if args.max_samples is not None:
        features = features[: args.max_samples]
        image_ids = image_ids[: args.max_samples]
    grouped: dict[str, list[str]] = {}
    for record in split_dataset.records:
        grouped.setdefault(record.image_id, []).append(record.raw_caption.lower())
    references = [grouped[image_id] for image_id in image_ids]
    model = tf.keras.models.load_model(run_dir / "model.keras")
    metrics, predictions = evaluate_caption_model(
        model,
        features,
        references,
        vocabulary,
        config,
    )
    if not args.skip_model_save:
        save_caption_metrics(
            metrics,
            predictions,
            run_dir,
            args.split,
        )
    if args.output_summary_path:
        write_csv(
            args.output_summary_path,
            [
                "run_id",
                "split",
                "bleu4",
                "meteor",
                "runtime_seconds",
                "model_dir",
            ],
            [[
                args.run_id,
                args.split,
                metrics.bleu4,
                metrics.meteor,
                metrics.runtime_seconds,
                _portable_path(run_dir),
            ]],
        )
    print(
        {
            "bleu4": metrics.bleu4,
            "meteor": metrics.meteor,
            "runtime_seconds": metrics.runtime_seconds,
        }
    )


def _portable_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return value.as_posix()


if __name__ == "__main__":
    main()
