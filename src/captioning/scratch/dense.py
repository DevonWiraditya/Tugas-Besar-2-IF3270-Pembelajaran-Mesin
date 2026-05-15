from __future__ import annotations

import numpy as np


class DenseLayer:
    def __init__(
        self,
        kernel: np.ndarray,
        bias: np.ndarray,
        activation: str | None = None,
    ) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        outputs = values @ self.kernel + self.bias
        if self.activation == "relu":
            return np.maximum(outputs, 0.0)
        if self.activation == "softmax":
            shifted = outputs - np.max(outputs, axis=-1, keepdims=True)
            exp = np.exp(shifted)
            return exp / np.sum(exp, axis=-1, keepdims=True)
        return outputs
