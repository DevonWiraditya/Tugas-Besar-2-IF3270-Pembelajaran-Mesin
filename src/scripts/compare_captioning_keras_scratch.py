from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.captioning.config import load_captioning_config
from src.captioning.data import Vocabulary, load_caption_dataset
from src.captioning.evaluation import sentence_bleu, sentence_meteor
from src.captioning.scratch_pipeline import generate_scratch_captions, load_scratch_decoder
from src.captioning.decoding import decode_token_ids, generate_caption_greedy
from src.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Keras and scratch captioning decoders.")
    parser.add_argument("--config", default="configs/captioning/base.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--decoder-type", required=True, choices=["rnn", "lstm"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_captioning_config(args.config, check_data=True)
    vocabulary = load_vocabulary(config)
    dataset = load_caption_dataset(config)
    split_dataset = getattr(dataset, args.split)
    features = np.load((config.output.artifacts_dir / f"{args.split}_features").with_suffix(".npy"))
    image_ids = list(split_dataset.image_ids)
    if args.max_samples is not None:
        features = features[: args.max_samples]
        image_ids = image_ids[: args.max_samples]
    grouped = group_references(split_dataset.records)
    references = [grouped[image_id] for image_id in image_ids]

    keras_model = tf.keras.models.load_model(config.output.models_dir / args.run_id / "model.keras")
    experiment = read_json(config.output.models_dir / args.run_id / "experiment.json")
    if int(experiment["layer_count"]) != 1:
        raise ValueError(
            "Scratch comparison currently supports only 1-layer decoders. "
            f"Run '{args.run_id}' has layer_count={experiment['layer_count']}."
        )
    keras_started = time.perf_counter()
    keras_predictions = [
        decode_token_ids(generate_caption_greedy(keras_model, feature, vocabulary, config), vocabulary)
        for feature in features
    ]
    keras_runtime = time.perf_counter() - keras_started

    scratch_decoder = load_scratch_decoder(config, args.run_id, args.decoder_type, vocabulary)
    scratch_started = time.perf_counter()
    scratch_predictions = generate_scratch_captions(scratch_decoder, features, vocabulary, config)
    scratch_runtime = time.perf_counter() - scratch_started

    result = {
        "run_id": args.run_id,
        "split": args.split,
        "decoder_type": args.decoder_type,
        "samples": len(image_ids),
        "keras_bleu4": float(np.mean([sentence_bleu(refs, pred) for refs, pred in zip(references, keras_predictions)])),
        "keras_meteor": float(np.mean([sentence_meteor(refs, pred) for refs, pred in zip(references, keras_predictions)])),
        "keras_runtime_seconds": float(keras_runtime),
        "scratch_bleu4": float(np.mean([sentence_bleu(refs, pred) for refs, pred in zip(references, scratch_predictions)])),
        "scratch_meteor": float(np.mean([sentence_meteor(refs, pred) for refs, pred in zip(references, scratch_predictions)])),
        "scratch_runtime_seconds": float(scratch_runtime),
    }

    samples = []
    for image_id, refs, keras_prediction, scratch_prediction in zip(
        image_ids,
        references,
        keras_predictions,
        scratch_predictions,
    ):
        samples.append(
            {
                "image_id": image_id,
                "ground_truths": refs,
                "keras_prediction": keras_prediction,
                "scratch_prediction": scratch_prediction,
                "bleu4_like_sample": sentence_bleu(refs, scratch_prediction),
                "meteor_like_sample": sentence_meteor(refs, scratch_prediction),
            }
        )

    payload = {"summary": result, "samples": samples}
    output_path = Path(args.output_path) if args.output_path else ROOT / f"reports/captioning/keras_vs_scratch_{args.run_id}_{args.split}.json"
    write_json(payload, output_path)
    print(result)


def load_vocabulary(config: CaptioningConfig) -> Vocabulary:
    preprocessing_dir = config.output.artifacts_dir / "preprocessing"
    token_to_id = read_json(preprocessing_dir / "token_to_id.json")
    id_to_token_raw = read_json(preprocessing_dir / "id_to_token.json")
    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token={int(key): value for key, value in id_to_token_raw.items()},
        pad_id=token_to_id["<pad>"],
        start_id=token_to_id["<start>"],
        end_id=token_to_id["<end>"],
        oov_id=token_to_id["<unk>"],
    )


def group_references(records) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(record.image_id, []).append(record.raw_caption.lower())
    return grouped


if __name__ == "__main__":
    main()
