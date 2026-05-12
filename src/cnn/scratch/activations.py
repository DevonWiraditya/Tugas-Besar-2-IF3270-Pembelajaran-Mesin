from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)
