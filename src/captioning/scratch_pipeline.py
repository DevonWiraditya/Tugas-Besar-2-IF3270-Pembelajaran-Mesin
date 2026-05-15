from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.captioning.config import CaptioningConfig
from src.captioning.data import Vocabulary
from src.captioning.decoding import decode_token_ids
from src.captioning.scratch.weights import build_scratch_decoder_from_keras


def load_scratch_decoder(
    config: CaptioningConfig,
    run_id: str,
    recurrent_type: str,
    vocabulary: Vocabulary,
):
    model = tf.keras.models.load_model(config.output.models_dir / run_id / "model.keras")
    return build_scratch_decoder_from_keras(model, recurrent_type, vocabulary)


def generate_scratch_caption(
    decoder,
    feature_vector: np.ndarray,
    vocabulary: Vocabulary,
    config: CaptioningConfig,
) -> str:
    token_ids = decoder.generate(feature_vector, config.text.max_caption_length)
    return decode_token_ids(token_ids, vocabulary)


def generate_scratch_captions(
    decoder,
    feature_vectors: np.ndarray,
    vocabulary: Vocabulary,
    config: CaptioningConfig,
) -> list[str]:
    token_batches = decoder.generate_batch(feature_vectors, config.text.max_caption_length)
    return [decode_token_ids(token_ids, vocabulary) for token_ids in token_batches]
