from __future__ import annotations

import numpy as np


class FlattenLayer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs.reshape(inputs.shape[0], -1)
