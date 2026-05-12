from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

from src.captioning.config import CaptioningConfig
from src.captioning.data import SplitCaptionDataset
from src.utils.images import load_image_batch
from src.utils.io import write_json


def build_caption_encoder(config: CaptioningConfig) -> tf.keras.Model:
    encoder = tf.keras.applications.InceptionV3(include_top=False, weights=config.encoder.weights, input_shape=(*config.data.image_size, config.data.channels), pooling=config.encoder.pooling)
    encoder.trainable = config.encoder.trainable
    return encoder


def extract_split_features(dataset: SplitCaptionDataset, config: CaptioningConfig, batch_size: int | None = None) -> tuple[np.ndarray, list[str]]:
    encoder = build_caption_encoder(config)
    image_ids = list(dataset.image_ids)
    image_paths = [str(config.data.images_dir / image_id) for image_id in image_ids]
    size = config.training.batch_size if batch_size is None else batch_size
    outputs: list[np.ndarray] = []
    for start in range(0, len(image_paths), size):
        batch_paths = image_paths[start : start + size]
        images = load_image_batch(batch_paths, image_size=config.data.image_size, color_mode=config.data.color_mode, normalization=config.data.normalization)
        processed = tf.keras.applications.inception_v3.preprocess_input(images * 255.0)
        outputs.append(encoder.predict(processed, verbose=0))
    if not outputs:
        return np.empty((0, config.encoder.feature_dim), dtype=np.float32), image_ids
    return np.concatenate(outputs, axis=0), image_ids


def save_split_features(features: np.ndarray, image_ids: list[str], output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix('.npy'), features)
    write_json({'image_ids': image_ids, 'shape': list(features.shape)}, prefix.with_suffix('.json'))
