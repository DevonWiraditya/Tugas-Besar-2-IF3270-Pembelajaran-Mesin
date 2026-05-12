from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

from src.cnn.config import CNNConfig
from src.utils.io import write_json
from src.utils.images import load_image_batch


def build_frozen_encoder(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg',
    )
    backbone.trainable = False
    return backbone


def extract_features_for_paths(
    paths: Iterable[str | Path],
    config: CNNConfig,
    batch_size: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    path_list = [str(Path(path)) for path in paths]
    encoder = build_frozen_encoder((*config.data.image_size, config.data.channels))
    size = config.training.batch_size if batch_size is None else batch_size
    features: list[np.ndarray] = []
    for start in range(0, len(path_list), size):
        batch_paths = path_list[start : start + size]
        images = load_image_batch(
            batch_paths,
            image_size=config.data.image_size,
            color_mode=config.data.color_mode,
            normalization=config.data.normalization,
        )
        resized = tf.image.resize(images, (299, 299)).numpy()
        processed = tf.keras.applications.inception_v3.preprocess_input(resized * 255.0)
        features.append(encoder.predict(processed, verbose=0))
    if not features:
        return np.empty((0, 2048), dtype=np.float32), []
    return np.concatenate(features, axis=0), path_list


def save_feature_artifacts(features: np.ndarray, paths: list[str], output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_suffix('.npy'), features)
    write_json({'paths': paths, 'feature_shape': list(features.shape)}, prefix.with_suffix('.json'))
