from __future__ import annotations

import numpy as np

from src.cnn.scratch.activations import relu


class Conv2DLayer:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, strides: tuple[int, int] = (1, 1), activation: str | None = None) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.strides = strides
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch, height, width, channels = inputs.shape
        kernel_height, kernel_width, _, filters = self.kernel.shape
        stride_height, stride_width = self.strides
        out_height = 1 + (height - kernel_height) // stride_height
        out_width = 1 + (width - kernel_width) // stride_width
        outputs = np.empty((batch, out_height, out_width, filters), dtype=np.float32)
        for row in range(out_height):
            for col in range(out_width):
                row_start = row * stride_height
                col_start = col * stride_width
                patch = inputs[:, row_start:row_start + kernel_height, col_start:col_start + kernel_width, :]
                for filter_index in range(filters):
                    outputs[:, row, col, filter_index] = np.sum(patch * self.kernel[..., filter_index], axis=(1, 2, 3)) + self.bias[filter_index]
        if self.activation == 'relu':
            return relu(outputs)
        return outputs
