from __future__ import annotations

import numpy as np


class LSTMCell:
    def __init__(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray) -> None:
        self.kernel = np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = np.asarray(recurrent_kernel, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.hidden_units = recurrent_kernel.shape[0]

    def step(self, x_t: np.ndarray, h_t: np.ndarray, c_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gates = x_t @ self.kernel + h_t @ self.recurrent_kernel + self.bias
        i, f, g, o = np.split(gates, 4, axis=-1)
        i = _sigmoid(i)
        f = _sigmoid(f)
        g = np.tanh(g)
        o = _sigmoid(o)
        next_c = f * c_t + i * g
        next_h = o * np.tanh(next_c)
        return next_h, next_c


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))
