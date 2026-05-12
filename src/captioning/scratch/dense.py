from __future__ import annotations

import numpy as np


class DenseLayer:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, activation: str | None = None) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs @ self.kernel + self.bias
        if self.activation == 'relu':
            return np.maximum(outputs, 0.0)
        if self.activation == 'softmax':
            shifted = outputs - np.max(outputs)
            exp = np.exp(shifted)
            return exp / np.sum(exp)
        return outputs
