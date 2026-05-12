from __future__ import annotations

import numpy as np

from src.cnn.scratch.activations import relu


class LocallyConnected2DLayer:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, output_shape: tuple[int, int], activation: str | None = None) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.output_shape = output_shape
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch = inputs.shape[0]
        out_height, out_width = self.output_shape
        filters = self.kernel.shape[-1]
        outputs = np.empty((batch, out_height, out_width, filters), dtype=np.float32)
        flat_inputs = inputs.reshape(batch, -1)
        for index in range(out_height * out_width):
            row = index // out_width
            col = index % out_width
            outputs[:, row, col, :] = flat_inputs @ self.kernel[index] + self.bias[index]
        if self.activation == 'relu':
            return relu(outputs)
        return outputs
