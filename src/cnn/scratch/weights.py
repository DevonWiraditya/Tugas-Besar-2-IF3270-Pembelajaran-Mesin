from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.experiments import CNNExperiment
from src.cnn.keras_models import build_shared_cnn_model
from src.cnn.scratch.conv import Conv2DLayer
from src.cnn.scratch.dense import DenseLayer
from src.cnn.scratch.flatten import FlattenLayer
from src.cnn.scratch.model import ScratchSequentialModel
from src.cnn.scratch.pooling import AveragePooling2DLayer, MaxPooling2DLayer


def load_shared_scratch_model(config: CNNConfig, experiment: CNNExperiment, weights_path: str | Path) -> ScratchSequentialModel:
    keras_model = build_shared_cnn_model(config, experiment)
    keras_model.load_weights(weights_path)
    layers = []
    for layer in keras_model.layers:
        name = layer.__class__.__name__
        if name == 'Conv2D':
            kernel, bias = layer.get_weights()
            layers.append(Conv2DLayer(kernel=kernel, bias=bias, strides=config.architecture_defaults.conv_strides, activation=layer.activation.__name__ if layer.activation else None))
        elif name == 'MaxPooling2D':
            layers.append(MaxPooling2DLayer(config.architecture_defaults.pool_size, config.architecture_defaults.pool_strides))
        elif name == 'AveragePooling2D':
            layers.append(AveragePooling2DLayer(config.architecture_defaults.pool_size, config.architecture_defaults.pool_strides))
        elif name == 'Flatten':
            layers.append(FlattenLayer())
        elif name == 'Dense':
            kernel, bias = layer.get_weights()
            activation = layer.activation.__name__ if layer.activation else None
            layers.append(DenseLayer(kernel=kernel, bias=bias, activation=activation))
    return ScratchSequentialModel(layers)
