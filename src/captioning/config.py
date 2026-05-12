from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptioningDataConfig:
    dataset_root: Path
    images_dir: Path
    captions_path: Path
    train_split_path: Path
    validation_split_path: Path
    test_split_path: Path
    image_size: tuple[int, int]
    channels: int
    color_mode: str
    normalization: str


@dataclass(frozen=True)
class CaptioningTextConfig:
    lowercase: bool
    strip_punctuation: bool
    start_token: str
    end_token: str
    pad_token: str
    oov_token: str
    min_word_frequency: int
    max_caption_length: int
    embedding_dim: int


@dataclass(frozen=True)
class CaptioningEncoderConfig:
    backbone: str
    weights: str
    trainable: bool
    pooling: str
    feature_dim: int
    projection_dim: int


@dataclass(frozen=True)
class CaptioningTrainingConfig:
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    loss: str


@dataclass(frozen=True)
class CaptioningExperimentGrid:
    recurrent_types: tuple[str, ...]
    layer_counts: tuple[int, ...]
    hidden_sizes: tuple[int, ...]


@dataclass(frozen=True)
class CaptioningOutputConfig:
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class CaptioningConfig:
    project: str
    seed: int
    data: CaptioningDataConfig
    text: CaptioningTextConfig
    encoder: CaptioningEncoderConfig
    training: CaptioningTrainingConfig
    experiment_grid: CaptioningExperimentGrid
    output: CaptioningOutputConfig


def load_captioning_config(path: str | Path, check_data: bool = False) -> CaptioningConfig:
    config_path = Path(path)
    with config_path.open('r', encoding='utf-8') as file:
        raw = json.load(file)
    root = config_path.resolve().parents[2]
    config = _parse_config(raw, root)
    validate_captioning_config(config, check_data=check_data)
    return config


def validate_captioning_config(config: CaptioningConfig, check_data: bool = False) -> None:
    if config.encoder.backbone != 'inception_v3':
        raise ValueError('Only inception_v3 is currently implemented for captioning.')
    if check_data:
        for path in (config.data.images_dir, config.data.captions_path, config.data.train_split_path, config.data.validation_split_path, config.data.test_split_path):
            if not path.exists():
                raise FileNotFoundError(path)


def _parse_config(raw: dict[str, Any], root: Path) -> CaptioningConfig:
    data = raw['data']
    text = raw['text']
    encoder = raw['encoder']
    training = raw['training']
    grid = raw['experiment_grid']
    output = raw['output']
    return CaptioningConfig(
        project=raw['project'],
        seed=int(raw['seed']),
        data=CaptioningDataConfig(dataset_root=_resolve(root, data['dataset_root']), images_dir=_resolve(root, data['images_dir']), captions_path=_resolve(root, data['captions_path']), train_split_path=_resolve(root, data['train_split_path']), validation_split_path=_resolve(root, data['validation_split_path']), test_split_path=_resolve(root, data['test_split_path']), image_size=_pair(data['image_size']), channels=int(data['channels']), color_mode=data['color_mode'], normalization=data['normalization']),
        text=CaptioningTextConfig(lowercase=bool(text['lowercase']), strip_punctuation=bool(text['strip_punctuation']), start_token=text['start_token'], end_token=text['end_token'], pad_token=text['pad_token'], oov_token=text['oov_token'], min_word_frequency=int(text['min_word_frequency']), max_caption_length=int(text['max_caption_length']), embedding_dim=int(text['embedding_dim'])),
        encoder=CaptioningEncoderConfig(backbone=encoder['backbone'], weights=encoder['weights'], trainable=bool(encoder['trainable']), pooling=encoder['pooling'], feature_dim=int(encoder['feature_dim']), projection_dim=int(encoder['projection_dim'])),
        training=CaptioningTrainingConfig(batch_size=int(training['batch_size']), epochs=int(training['epochs']), optimizer=training['optimizer'], learning_rate=float(training['learning_rate']), loss=training['loss']),
        experiment_grid=CaptioningExperimentGrid(recurrent_types=tuple(grid['recurrent_types']), layer_counts=tuple(int(v) for v in grid['layer_counts']), hidden_sizes=tuple(int(v) for v in grid['hidden_sizes'])),
        output=CaptioningOutputConfig(artifacts_dir=_resolve(root, output['artifacts_dir']), models_dir=_resolve(root, output['models_dir']), reports_dir=_resolve(root, output['reports_dir'])),
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _pair(values: list[int] | tuple[int, int]) -> tuple[int, int]:
    return int(values[0]), int(values[1])
