from __future__ import annotations

import numpy as np


class EmbeddingLayer:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.weights[token_ids]
