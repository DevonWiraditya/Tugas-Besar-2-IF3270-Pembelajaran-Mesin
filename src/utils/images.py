from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ImagePath = str | Path

_PIL_MODES = {
    "rgb": "RGB",
    "grayscale": "L",
}


def load_image(
    path: ImagePath,
    image_size: tuple[int, int],
    color_mode: str = "rgb",
    normalization: str = "zero_one",
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    target_height, target_width = image_size
    pil_mode = _pil_mode(color_mode)

    with Image.open(path) as image:
        image = image.convert(pil_mode)
        image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=dtype)

    if color_mode == "grayscale":
        array = array[..., np.newaxis]

    if normalization == "zero_one":
        array = array / np.asarray(255.0, dtype=dtype)
    elif normalization != "none":
        raise ValueError(f"Unsupported normalization: {normalization}")

    expected_channels = 1 if color_mode == "grayscale" else 3
    expected_shape = (target_height, target_width, expected_channels)
    if array.shape != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {array.shape}: {path}")

    return np.ascontiguousarray(array, dtype=dtype)


def load_image_batch(
    paths: Iterable[ImagePath],
    image_size: tuple[int, int],
    color_mode: str = "rgb",
    normalization: str = "zero_one",
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    path_list = list(paths)
    channels = 1 if color_mode == "grayscale" else 3
    if not path_list:
        height, width = image_size
        return np.empty((0, height, width, channels), dtype=dtype)

    images = [
        load_image(
            path,
            image_size=image_size,
            color_mode=color_mode,
            normalization=normalization,
            dtype=dtype,
        )
        for path in path_list
    ]
    return np.stack(images, axis=0).astype(dtype, copy=False)


def _pil_mode(color_mode: str) -> str:
    try:
        return _PIL_MODES[color_mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_PIL_MODES))
        raise ValueError(f"Unsupported color_mode '{color_mode}'. Use one of: {supported}") from exc
