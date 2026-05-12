from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.experiments import CNNExperiment


def build_shared_cnn_model(config: CNNConfig, experiment: CNNExperiment) -> tf.keras.Model:
    defaults = config.architecture_defaults
    inputs = tf.keras.layers.Input(shape=config.input_shape, name="image")
    x = inputs

    for index, (filters, kernel_size) in enumerate(
        zip(experiment.filters, experiment.kernel_sizes),
        start=1,
    ):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=defaults.conv_strides,
            padding=defaults.conv_padding,
            activation=defaults.conv_activation,
            name=f"conv2d_{index}",
        )(x)
        x = _pooling_layer(experiment.pooling_type, config, index)(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    for index, units in enumerate(defaults.dense_units, start=1):
        x = tf.keras.layers.Dense(
            units,
            activation=defaults.dense_activation,
            name=f"dense_{index}",
        )(x)

    outputs = tf.keras.layers.Dense(
        len(config.data.class_names),
        activation=defaults.output_activation,
        name="predictions",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=experiment.run_id)


def compile_cnn_model(model: tf.keras.Model, config: CNNConfig) -> tf.keras.Model:
    optimizer = _optimizer(config)
    model.compile(
        optimizer=optimizer,
        loss=config.training.loss,
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def _pooling_layer(
    pooling_type: str,
    config: CNNConfig,
    index: int,
) -> tf.keras.layers.Layer:
    defaults = config.architecture_defaults
    kwargs = {
        "pool_size": defaults.pool_size,
        "strides": defaults.pool_strides,
        "padding": defaults.pool_padding,
        "name": f"{pooling_type}_pooling2d_{index}",
    }

    if pooling_type == "max":
        return tf.keras.layers.MaxPooling2D(**kwargs)
    if pooling_type == "average":
        return tf.keras.layers.AveragePooling2D(**kwargs)
    raise ValueError(f"Unsupported pooling_type: {pooling_type}")


def _optimizer(config: CNNConfig) -> tf.keras.optimizers.Optimizer:
    name = config.training.optimizer.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
