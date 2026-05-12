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
        windows = _patch_windows(padded, (kernel_height, kernel_width), self.strides)
        windows = windows[:, :out_height, :out_width, :, :, :]
        output = np.tensordot(windows, kernel, axes=([3, 4, 5], [0, 1, 2])) + bias
        output = output.astype(np.float32, copy=False)

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
        windows = _patch_windows(padded, self.kernel_size, self.strides)
        windows = windows[:, :out_height, :out_width, :, :, :]
        patches = windows.reshape(batch.shape[0], out_height * out_width, -1)
        output = np.einsum("nlf,lfo->nlo", patches, kernel) + bias
        output = output.reshape(batch.shape[0], out_height, out_width, out_channels)
        output = output.astype(np.float32, copy=False)

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
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (pool_height, pool_width),
            axis=(1, 2),
        )
        windows = windows[:, ::stride_height, ::stride_width, :, :, :]
        windows = windows[:, :out_height, :out_width, :, :, :]

        if self.mode == "max":
            output = np.max(windows, axis=(4, 5))
        elif self.mode == "average":
            output = np.mean(windows, axis=(4, 5))
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")

        return output.astype(np.float32, copy=False)


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


def _patch_windows(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
) -> np.ndarray:
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = strides
    windows = np.lib.stride_tricks.sliding_window_view(
        x,
        (kernel_height, kernel_width),
        axis=(1, 2),
    )
    windows = windows[:, ::stride_height, ::stride_width, :, :, :]
    return np.transpose(windows, (0, 1, 2, 4, 5, 3))


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
