from __future__ import annotations

import numpy as np


class MaxPooling2DLayer:
    def __init__(self, pool_size: tuple[int, int], strides: tuple[int, int]) -> None:
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return _pool(inputs, self.pool_size, self.strides, reducer=np.max)


class AveragePooling2DLayer:
    def __init__(self, pool_size: tuple[int, int], strides: tuple[int, int]) -> None:
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return _pool(inputs, self.pool_size, self.strides, reducer=np.mean)


class GlobalAveragePooling2DLayer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.mean(inputs, axis=(1, 2))


class GlobalMaxPooling2DLayer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.max(inputs, axis=(1, 2))


def _pool(inputs: np.ndarray, pool_size: tuple[int, int], strides: tuple[int, int], reducer) -> np.ndarray:
    batch, height, width, channels = inputs.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = strides
    out_height = 1 + (height - pool_height) // stride_height
    out_width = 1 + (width - pool_width) // stride_width
    outputs = np.empty((batch, out_height, out_width, channels), dtype=np.float32)
    for row in range(out_height):
        for col in range(out_width):
            row_start = row * stride_height
            col_start = col * stride_width
            patch = inputs[:, row_start:row_start + pool_height, col_start:col_start + pool_width, :]
            outputs[:, row, col, :] = reducer(patch, axis=(1, 2))
    return outputs
