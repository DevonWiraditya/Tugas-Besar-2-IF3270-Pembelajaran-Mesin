from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ScratchLayer(Protocol):
    name: str

    def forward(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class Conv2DLayer:
    kernel: np.ndarray
    bias: np.ndarray
    strides: tuple[int, int]
    padding: str
    activation: str = "linear"
    name: str = "conv2d"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        kernel = self.kernel.astype(np.float32, copy=False)
        bias = self.bias.astype(np.float32, copy=False)
        kernel_height, kernel_width, _, out_channels = kernel.shape
        stride_height, stride_width = self.strides
        padded = _pad_input(batch, (kernel_height, kernel_width), self.strides, self.padding)
        out_height = _output_size(batch.shape[1], kernel_height, stride_height, self.padding)
        out_width = _output_size(batch.shape[2], kernel_width, stride_width, self.padding)
        output = np.empty((batch.shape[0], out_height, out_width, out_channels), dtype=np.float32)

        for row in range(out_height):
            row_start = row * stride_height
            for col in range(out_width):
                col_start = col * stride_width
                patch = padded[
                    :,
                    row_start : row_start + kernel_height,
                    col_start : col_start + kernel_width,
                    :,
                ]
                output[:, row, col, :] = (
                    np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias
                )

        return apply_activation(output, self.activation)


@dataclass(frozen=True)
class LocallyConnected2DLayer:
    kernel: np.ndarray
    bias: np.ndarray
    kernel_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str
    activation: str = "linear"
    name: str = "locally_connected2d"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        kernel = self.kernel.astype(np.float32, copy=False)
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.strides
        padded = _pad_input(batch, self.kernel_size, self.strides, self.padding)
        out_height = _output_size(batch.shape[1], kernel_height, stride_height, self.padding)
        out_width = _output_size(batch.shape[2], kernel_width, stride_width, self.padding)
        out_channels = kernel.shape[-1]
        bias = _local_bias(self.bias, out_height, out_width, out_channels)
        output = np.empty((batch.shape[0], out_height, out_width, out_channels), dtype=np.float32)

        for row in range(out_height):
            row_start = row * stride_height
            for col in range(out_width):
                col_start = col * stride_width
                location = row * out_width + col
                patch = padded[
                    :,
                    row_start : row_start + kernel_height,
                    col_start : col_start + kernel_width,
                    :,
                ].reshape(batch.shape[0], -1)
                output[:, row, col, :] = patch @ kernel[location] + bias[location]

        return apply_activation(output, self.activation)


@dataclass(frozen=True)
class Pooling2DLayer:
    pool_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str
    mode: str
    name: str = "pooling2d"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides
        padded = _pad_pool_input(batch, self.pool_size, self.strides, self.padding, self.mode)
        out_height = _output_size(batch.shape[1], pool_height, stride_height, self.padding)
        out_width = _output_size(batch.shape[2], pool_width, stride_width, self.padding)
        output = np.empty((batch.shape[0], out_height, out_width, batch.shape[-1]), dtype=np.float32)

        for row in range(out_height):
            row_start = row * stride_height
            for col in range(out_width):
                col_start = col * stride_width
                patch = padded[
                    :,
                    row_start : row_start + pool_height,
                    col_start : col_start + pool_width,
                    :,
                ]
                if self.mode == "max":
                    output[:, row, col, :] = np.max(patch, axis=(1, 2))
                elif self.mode == "average":
                    output[:, row, col, :] = np.mean(patch, axis=(1, 2))
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.mode}")

        return output


@dataclass(frozen=True)
class GlobalPooling2DLayer:
    mode: str
    name: str = "global_pooling2d"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        if self.mode == "max":
            return np.max(batch, axis=(1, 2))
        if self.mode == "average":
            return np.mean(batch, axis=(1, 2))
        raise ValueError(f"Unsupported global pooling mode: {self.mode}")


@dataclass(frozen=True)
class FlattenLayer:
    name: str = "flatten"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        return batch.reshape(batch.shape[0], -1)


@dataclass(frozen=True)
class DenseLayer:
    kernel: np.ndarray
    bias: np.ndarray
    activation: str = "linear"
    name: str = "dense"

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch = _as_batch(x)
        output = batch @ self.kernel.astype(np.float32, copy=False)
        output = output + self.bias.astype(np.float32, copy=False)
        return apply_activation(output, self.activation)


def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    name = activation.lower()
    if name in {"linear", "identity"}:
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "softmax":
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unsupported activation: {activation}")


def _as_batch(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 3:
        return array.reshape(1, *array.shape)
    if array.ndim in {2, 4}:
        return array
    raise ValueError(f"Expected 1D, 2D, 3D, or 4D input, got shape {array.shape}.")


def _output_size(input_size: int, kernel_size: int, stride: int, padding: str) -> int:
    mode = padding.lower()
    if mode == "valid":
        return (input_size - kernel_size) // stride + 1
    if mode == "same":
        return int(np.ceil(input_size / stride))
    raise ValueError(f"Unsupported padding: {padding}")


def _pad_input(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
) -> np.ndarray:
    if padding.lower() == "valid":
        return x
    top, bottom = _padding_pair(x.shape[1], kernel_size[0], strides[0], padding)
    left, right = _padding_pair(x.shape[2], kernel_size[1], strides[1], padding)
    return np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant")


def _pad_pool_input(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: str,
    mode: str,
) -> np.ndarray:
    if padding.lower() == "valid":
        return x
    fill_value = -np.inf if mode == "max" else 0.0
    top, bottom = _padding_pair(x.shape[1], kernel_size[0], strides[0], padding)
    left, right = _padding_pair(x.shape[2], kernel_size[1], strides[1], padding)
    return np.pad(
        x,
        ((0, 0), (top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=fill_value,
    )


def _padding_pair(input_size: int, kernel_size: int, stride: int, padding: str) -> tuple[int, int]:
    if padding.lower() != "same":
        raise ValueError(f"Unsupported padding: {padding}")
    output_size = int(np.ceil(input_size / stride))
    total = max((output_size - 1) * stride + kernel_size - input_size, 0)
    before = total // 2
    return before, total - before


def _local_bias(
    bias: np.ndarray,
    out_height: int,
    out_width: int,
    out_channels: int,
) -> np.ndarray:
    array = bias.astype(np.float32, copy=False)
    if array.shape == (out_channels,):
        return np.tile(array.reshape(1, out_channels), (out_height * out_width, 1))
    return array.reshape(out_height * out_width, out_channels)
