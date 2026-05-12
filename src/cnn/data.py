from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np

from src.cnn.config import CNNConfig
from src.utils.images import load_image_batch


DEFAULT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class CNNDataset:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]
    pred: tuple[Path, ...]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_cnn_dataset(config: CNNConfig) -> CNNDataset:
    train_records = scan_labeled_image_dir(config.data.train_dir, config.data.class_names)
    train, validation = stratified_train_validation_split(
        train_records,
        validation_split=config.data.validation_split,
        seed=config.seed,
    )
    test = scan_labeled_image_dir(config.data.test_dir, config.data.class_names)
    pred = scan_unlabeled_image_dir(config.data.pred_dir)

    return CNNDataset(
        train=train,
        validation=validation,
        test=test,
        pred=pred,
        class_names=config.data.class_names,
    )


def scan_labeled_image_dir(
    root: str | Path,
    class_names: Iterable[str],
    extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> tuple[ImageRecord, ...]:
    root_path = Path(root)
    records: list[ImageRecord] = []

    for label, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in _iter_image_files(class_dir, extensions):
            records.append(ImageRecord(path=path, label=label, class_name=class_name))

    return tuple(records)


def scan_unlabeled_image_dir(
    root: str | Path,
    extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> tuple[Path, ...]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Missing image directory: {root_path}")
    return tuple(_iter_image_files(root_path, extensions))


def stratified_train_validation_split(
    records: Iterable[ImageRecord],
    validation_split: float,
    seed: int,
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    by_label: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    rng = random.Random(seed)

    for label in sorted(by_label):
        class_records = sorted(by_label[label], key=lambda item: item.path.as_posix())
        rng.shuffle(class_records)

        validation_count = int(round(len(class_records) * validation_split))
        if len(class_records) > 1:
            validation_count = min(max(validation_count, 1), len(class_records) - 1)

        validation.extend(class_records[:validation_count])
        train.extend(class_records[validation_count:])

    return _sort_records(train), _sort_records(validation)


def records_to_arrays(
    records: Iterable[ImageRecord],
    config: CNNConfig,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    record_list = list(records)
    x = load_image_batch(
        [record.path for record in record_list],
        image_size=config.data.image_size,
        color_mode=config.data.color_mode,
        normalization=config.data.normalization,
        dtype=dtype,
    )
    y = np.asarray([record.label for record in record_list], dtype=np.int64)
    return x, y


def iter_record_batches(
    records: Iterable[ImageRecord],
    config: CNNConfig,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dtype: np.dtype | type = np.float32,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    record_list = list(records)
    if shuffle:
        rng = random.Random(config.seed if seed is None else seed)
        rng.shuffle(record_list)

    size = config.training.batch_size if batch_size is None else batch_size
    if size <= 0:
        raise ValueError("batch_size must be positive.")

    for start in range(0, len(record_list), size):
        yield records_to_arrays(record_list[start : start + size], config, dtype=dtype)


def count_by_class(
    records: Iterable[ImageRecord],
    class_names: Iterable[str],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for record in records:
        counts[record.class_name] += 1
    return counts


def _iter_image_files(
    root: Path,
    extensions: frozenset[str],
) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _sort_records(records: Iterable[ImageRecord]) -> tuple[ImageRecord, ...]:
    return tuple(sorted(records, key=lambda item: (item.label, item.path.as_posix())))
