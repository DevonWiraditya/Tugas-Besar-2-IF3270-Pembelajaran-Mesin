from __future__ import annotations

import numpy as np

from src.cnn.scratch.activations import relu, softmax


class DenseLayer:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, activation: str | None = None) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs @ self.kernel + self.bias
        if self.activation == 'relu':
            return relu(outputs)
        if self.activation == 'softmax':
            return softmax(outputs)
        return outputs
