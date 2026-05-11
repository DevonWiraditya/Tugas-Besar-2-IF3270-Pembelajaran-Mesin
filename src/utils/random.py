from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, include_tensorflow: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if include_tensorflow:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
