from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.captioning.config import CaptioningConfig


@dataclass(frozen=True)
class CaptionRecord:
    image_id: str
    image_path: Path
    raw_caption: str


@dataclass(frozen=True)
class SplitCaptionDataset:
    records: tuple[CaptionRecord, ...]
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaptionDataset:
    train: SplitCaptionDataset
    validation: SplitCaptionDataset
    test: SplitCaptionDataset


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_id: int
    start_id: int
    end_id: int
    oov_id: int

    @property
    def size(self) -> int:
        return len(self.token_to_id)


def load_caption_dataset(config: CaptioningConfig) -> CaptionDataset:
    caption_lookup = load_caption_lookup(config.data.captions_path)
    return CaptionDataset(
        train=build_split_dataset(config.data.images_dir, config.data.train_split_path, caption_lookup),
        validation=build_split_dataset(config.data.images_dir, config.data.validation_split_path, caption_lookup),
        test=build_split_dataset(config.data.images_dir, config.data.test_split_path, caption_lookup),
    )


def load_caption_lookup(path: str | Path) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    with Path(path).open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split('	', maxsplit=1)
            image_id = key.split('#', maxsplit=1)[0]
            lookup[image_id].append(caption)
    return dict(lookup)


def build_split_dataset(images_dir: Path, split_path: Path, caption_lookup: dict[str, list[str]]) -> SplitCaptionDataset:
    image_ids = [line.strip() for line in split_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    records: list[CaptionRecord] = []
    for image_id in image_ids:
        image_path = images_dir / image_id
        for caption in caption_lookup.get(image_id, []):
            records.append(CaptionRecord(image_id=image_id, image_path=image_path, raw_caption=caption))
    return SplitCaptionDataset(records=tuple(records), image_ids=tuple(image_ids))
