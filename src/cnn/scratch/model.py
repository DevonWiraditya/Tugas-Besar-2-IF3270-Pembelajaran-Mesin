from __future__ import annotations

import numpy as np


class ScratchSequentialModel:
    def __init__(self, layers: list[object]) -> None:
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(inputs)
        return np.argmax(probabilities, axis=-1)
