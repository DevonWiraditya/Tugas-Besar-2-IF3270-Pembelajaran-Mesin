from __future__ import annotations

import os
from math import ceil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="cnn")
class LocallyConnected2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int, int] = (1, 1),
        padding: str = "valid",
        activation: str | None = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding.lower()
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"Expected rank-4 input, got shape {input_shape}.")

        height = int(input_shape[1])
        width = int(input_shape[2])
        channels = int(input_shape[3])
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides
        self.output_rows = _output_size(height, kernel_height, stride_height, self.padding)
        self.output_cols = _output_size(width, kernel_width, stride_width, self.padding)
        self.patch_dim = kernel_height * kernel_width * channels
        locations = self.output_rows * self.output_cols

        self.kernel = self.add_weight(
            name="kernel",
            shape=(locations, self.patch_dim, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(locations, self.filters),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        padding = self.padding.upper()
        batch_size = tf.shape(inputs)[0]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=(1, self.kernel_size[0], self.kernel_size[1], 1),
            strides=(1, self.strides[0], self.strides[1], 1),
            rates=(1, 1, 1, 1),
            padding=padding,
        )
        patches = tf.reshape(
            patches,
            (batch_size, self.output_rows * self.output_cols, self.patch_dim),
        )
        output = tf.einsum("blp,lpf->blf", patches, self.kernel)
        if self.bias is not None:
            output = output + self.bias
        output = tf.reshape(
            output,
            (batch_size, self.output_rows, self.output_cols, self.filters),
        )
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        height = int(input_shape[1])
        width = int(input_shape[2])
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides
        return (
            input_shape[0],
            _output_size(height, kernel_height, stride_height, self.padding),
            _output_size(width, kernel_width, stride_width, self.padding),
            self.filters,
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


def _pair(value) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _output_size(input_size: int, kernel_size: int, stride: int, padding: str) -> int:
    if padding == "valid":
        return (input_size - kernel_size) // stride + 1
    if padding == "same":
        return int(ceil(input_size / stride))
    raise ValueError(f"Unsupported padding: {padding}")
