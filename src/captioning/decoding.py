from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.captioning.config import CaptioningConfig
from src.captioning.data import Vocabulary


def generate_caption_greedy(
    model: tf.keras.Model,
    feature_vector: np.ndarray,
    vocabulary: Vocabulary,
    config: CaptioningConfig,
) -> list[int]:
    tokens = [vocabulary.start_id]
    while len(tokens) < config.text.max_caption_length:
        probabilities = _predict_next_token_distribution(
            model,
            feature_vector,
            tokens,
            vocabulary,
            config,
        )
        next_id = int(np.argmax(probabilities))
        tokens.append(next_id)
        if next_id == vocabulary.end_id:
            break
    return tokens


def decode_token_ids(token_ids: list[int], vocabulary: Vocabulary) -> str:
    words = []
    for token_id in token_ids:
        token = vocabulary.id_to_token.get(int(token_id), "")
        if token in {"<start>", "<pad>"}:
            continue
        if token == "<end>":
            break
        words.append(token)
    return " ".join(words).strip()


def _predict_next_token_distribution(
    model: tf.keras.Model,
    feature_vector: np.ndarray,
    tokens: list[int],
    vocabulary: Vocabulary,
    config: CaptioningConfig,
) -> np.ndarray:
    input_length = _caption_input_length(model)
    truncated = tokens[:input_length]
    padded = truncated + [vocabulary.pad_id] * (input_length - len(truncated))
    outputs = model.predict(
        [
            feature_vector[np.newaxis, ...],
            np.asarray([padded], dtype=np.int64),
        ],
        verbose=0,
    )[0]
    timestep_index = max(len(truncated) - 1, 0)
    return outputs[timestep_index]


def _caption_input_length(model: tf.keras.Model) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list) and len(input_shape) >= 2:
        caption_shape = input_shape[1]
        if len(caption_shape) >= 2 and caption_shape[1] is not None:
            return int(caption_shape[1])
    return int(model.inputs[1].shape[1])
