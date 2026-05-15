from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import tensorflow as tf

from src.captioning.config import CaptioningConfig
from src.captioning.data import Vocabulary
from src.captioning.decoding import (
    decode_token_ids,
    generate_caption_greedy,
)
from src.utils.io import write_json


@dataclass(frozen=True)
class CaptionMetrics:
    bleu4: float
    meteor: float
    runtime_seconds: float


def evaluate_caption_model(
    model: tf.keras.Model,
    features: np.ndarray,
    references: list[list[str]],
    vocabulary: Vocabulary,
    config: CaptioningConfig,
) -> tuple[CaptionMetrics, list[str]]:
    started = time.perf_counter()
    predictions = []
    for feature in features:
        token_ids = generate_caption_greedy(model, feature, vocabulary, config)
        predictions.append(decode_token_ids(token_ids, vocabulary))
    runtime = time.perf_counter() - started
    bleu = (
        np.mean(
            [sentence_bleu(refs, pred) for refs, pred in zip(references, predictions)]
        )
        if predictions
        else 0.0
    )
    meteor = (
        np.mean(
            [sentence_meteor(refs, pred) for refs, pred in zip(references, predictions)]
        )
        if predictions
        else 0.0
    )
    return (
        CaptionMetrics(
            bleu4=float(bleu),
            meteor=float(meteor),
            runtime_seconds=float(runtime),
        ),
        predictions,
    )


def sentence_bleu(references: list[str], prediction: str, max_n: int = 4) -> float:
    predicted_tokens = prediction.split()
    if not predicted_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(_ngrams(predicted_tokens, n))
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        ref_max = Counter()
        for reference in references:
            ref_counts = Counter(_ngrams(reference.split(), n))
            for gram, count in ref_counts.items():
                ref_max[gram] = max(ref_max[gram], count)
        overlap = sum(min(count, ref_max[gram]) for gram, count in pred_ngrams.items())
        precisions.append(overlap / sum(pred_ngrams.values()))
    if min(precisions) == 0.0:
        return 0.0
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_n)
    ref_lengths = [len(reference.split()) for reference in references]
    pred_length = len(predicted_tokens)
    closest_ref = min(ref_lengths, key=lambda value: (abs(value - pred_length), value))
    brevity_penalty = (
        1.0
        if pred_length > closest_ref
        else math.exp(1.0 - closest_ref / max(pred_length, 1))
    )
    return brevity_penalty * geo_mean


def sentence_meteor(references: list[str], prediction: str) -> float:
    predicted_tokens = prediction.split()
    if not predicted_tokens:
        return 0.0
    predicted_set = set(predicted_tokens)
    best = 0.0
    for reference in references:
        reference_set = set(reference.split())
        overlap = len(predicted_set & reference_set)
        if overlap == 0:
            continue
        precision = overlap / len(predicted_set)
        recall = overlap / len(reference_set)
        best = max(best, (10 * precision * recall) / (recall + 9 * precision))
    return best


def save_caption_metrics(
    metrics: CaptionMetrics,
    predictions: list[str],
    output_dir: Path,
    split_name: str,
) -> None:
    suffix = split_name
    write_json(
        {
            "bleu4": metrics.bleu4,
            "meteor": metrics.meteor,
            "runtime_seconds": metrics.runtime_seconds,
        },
        output_dir / f"metrics.{suffix}.json",
    )
    write_json(
        {
            "predictions": predictions,
        },
        output_dir / f"predictions.{suffix}.json",
    )


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
