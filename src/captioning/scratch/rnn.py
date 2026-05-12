from __future__ import annotations

import numpy as np


class SimpleRNNCell:
    def __init__(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = np.asarray(recurrent_kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)

    def step(self, x_t: np.ndarray, h_t: np.ndarray) -> np.ndarray:
        return np.tanh(x_t @ self.kernel + h_t @ self.recurrent_kernel + self.bias)
