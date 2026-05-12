from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.captioning.config import CaptioningConfig
from src.captioning.data import Vocabulary


def generate_caption_greedy(model: tf.keras.Model, feature_vector: np.ndarray, vocabulary: Vocabulary, config: CaptioningConfig) -> list[int]:
    tokens = [vocabulary.start_id]
    while len(tokens) < config.text.max_caption_length:
        padded = tokens + [vocabulary.pad_id] * ((config.text.max_caption_length - 1) - len(tokens))
        probabilities = model.predict([feature_vector[np.newaxis, ...], np.asarray([padded], dtype=np.int64)], verbose=0)[0]
        next_id = int(np.argmax(probabilities))
        tokens.append(next_id)
        if next_id == vocabulary.end_id:
            break
    return tokens


def decode_token_ids(token_ids: list[int], vocabulary: Vocabulary) -> str:
    words = []
    for token_id in token_ids:
        token = vocabulary.id_to_token.get(int(token_id), '')
        if token in {'<start>', '<pad>'}:
            continue
        if token == '<end>':
            break
        words.append(token)
    return ' '.join(words).strip()
