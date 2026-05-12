from __future__ import annotations

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf

from src.cnn.config import CNNConfig
from src.cnn.experiments import CNNExperiment
from src.cnn.keras_models import compile_cnn_model


class FallbackLocallyConnected2D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: tuple[int, int], activation: str, name: str) -> None:
        super().__init__(name=name)
        self.inner = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding='valid',
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.inner(inputs)


def build_non_shared_cnn_model(config: CNNConfig, experiment: CNNExperiment) -> tf.keras.Model:
    defaults = config.architecture_defaults
    inputs = tf.keras.layers.Input(shape=config.input_shape, name='image')
    x = inputs

    layer_factory = getattr(tf.keras.layers, 'LocallyConnected2D', None)

    for index, (filters, kernel_size) in enumerate(
        zip(experiment.filters, experiment.kernel_sizes),
        start=1,
    ):
        if layer_factory is None:
            x = FallbackLocallyConnected2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=defaults.conv_activation,
                name=f'locally_connected2d_{index}',
            )(x)
        else:
            x = layer_factory(
                filters=filters,
                kernel_size=kernel_size,
                strides=defaults.conv_strides,
                activation=defaults.conv_activation,
                name=f'locally_connected2d_{index}',
            )(x)

        pool_name = f"{experiment.pooling_type}_pooling2d_{index}"
        if experiment.pooling_type == 'max':
            x = tf.keras.layers.MaxPooling2D(
                pool_size=defaults.pool_size,
                strides=defaults.pool_strides,
                padding=defaults.pool_padding,
                name=pool_name,
            )(x)
        else:
            x = tf.keras.layers.AveragePooling2D(
                pool_size=defaults.pool_size,
                strides=defaults.pool_strides,
                padding=defaults.pool_padding,
                name=pool_name,
            )(x)

    x = tf.keras.layers.Flatten(name='flatten')(x)
    for index, units in enumerate(defaults.dense_units, start=1):
        x = tf.keras.layers.Dense(units, activation=defaults.dense_activation, name=f'dense_{index}')(x)

    outputs = tf.keras.layers.Dense(
        len(config.data.class_names),
        activation=defaults.output_activation,
        name='predictions',
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'{experiment.run_id}_non_shared')
    return compile_cnn_model(model, config)
