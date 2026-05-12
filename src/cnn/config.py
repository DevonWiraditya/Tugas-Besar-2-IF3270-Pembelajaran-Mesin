from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CNNDataConfig:
    dataset_root: Path
    train_dir: Path
    test_dir: Path
    pred_dir: Path
    validation_split: float
    image_size: tuple[int, int]
    channels: int
    data_format: str
    color_mode: str
    normalization: str
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class CNNTrainingConfig:
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    loss: str
    comparison_metric: str


@dataclass(frozen=True)
class CNNArchitectureDefaults:
    conv_activation: str
    dense_activation: str
    output_activation: str
    conv_padding: str
    conv_strides: tuple[int, int]
    pool_padding: str
    pool_size: tuple[int, int]
    pool_strides: tuple[int, int]
    dense_units: tuple[int, ...]


@dataclass(frozen=True)
class CNNExperimentGrid:
    conv_layer_counts: tuple[int, ...]
    filter_profiles: tuple[tuple[int, ...], ...]
    kernel_profiles: tuple[tuple[tuple[int, int], ...], ...]
    pooling_types: tuple[str, ...]

    @property
    def experiment_count(self) -> int:
        return (
            len(self.conv_layer_counts)
            * len(self.filter_profiles)
            * len(self.kernel_profiles)
            * len(self.pooling_types)
        )


@dataclass(frozen=True)
class CNNOutputConfig:
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class CNNConfig:
    project: str
    seed: int
    data: CNNDataConfig
    training: CNNTrainingConfig
    architecture_defaults: CNNArchitectureDefaults
    experiment_grid: CNNExperimentGrid
    output: CNNOutputConfig

    @property
    def input_shape(self) -> tuple[int, int, int]:
        height, width = self.data.image_size
        return height, width, self.data.channels

    @property
    def experiment_count(self) -> int:
        return self.experiment_grid.experiment_count


def load_cnn_config(path: str | Path, check_data: bool = False) -> CNNConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    root = config_path.resolve().parents[2]
    config = _parse_config(raw, root)
    validate_cnn_config(config, check_data=check_data)
    return config


def validate_cnn_config(config: CNNConfig, check_data: bool = False) -> None:
    if config.data.data_format != "channels_last":
        raise ValueError("CNN scratch implementation expects channels_last/NHWC tensors.")

    if config.data.channels != 3:
        raise ValueError("Intel image classification config expects RGB images.")

    if config.data.normalization != "zero_one":
        raise ValueError("Only zero_one normalization is supported by the shared config.")

    if not 0.0 < config.data.validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    if config.training.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if config.training.epochs <= 0:
        raise ValueError("epochs must be positive.")

    if config.experiment_count != 16:
        raise ValueError(f"Expected 16 CNN experiments, got {config.experiment_count}.")

    max_layers = max(config.experiment_grid.conv_layer_counts)
    for profile in config.experiment_grid.filter_profiles:
        if len(profile) < max_layers:
            raise ValueError("Every filter profile must cover the maximum conv layer count.")

    for profile in config.experiment_grid.kernel_profiles:
        if len(profile) < max_layers:
            raise ValueError("Every kernel profile must cover the maximum conv layer count.")

    if check_data:
        _validate_dataset_layout(config)


def _parse_config(raw: dict[str, Any], root: Path) -> CNNConfig:
    data = raw["data"]
    training = raw["training"]
    defaults = raw["architecture_defaults"]
    grid = raw["experiment_grid"]
    output = raw["output"]

    return CNNConfig(
        project=raw["project"],
        seed=int(raw["seed"]),
        data=CNNDataConfig(
            dataset_root=_resolve(root, data["dataset_root"]),
            train_dir=_resolve(root, data["train_dir"]),
            test_dir=_resolve(root, data["test_dir"]),
            pred_dir=_resolve(root, data["pred_dir"]),
            validation_split=float(data["validation_split"]),
            image_size=_pair(data["image_size"]),
            channels=int(data["channels"]),
            data_format=data["data_format"],
            color_mode=data["color_mode"],
            normalization=data["normalization"],
            class_names=tuple(data["class_names"]),
        ),
        training=CNNTrainingConfig(
            batch_size=int(training["batch_size"]),
            epochs=int(training["epochs"]),
            optimizer=training["optimizer"],
            learning_rate=float(training["learning_rate"]),
            loss=training["loss"],
            comparison_metric=training["comparison_metric"],
        ),
        architecture_defaults=CNNArchitectureDefaults(
            conv_activation=defaults["conv_activation"],
            dense_activation=defaults["dense_activation"],
            output_activation=defaults["output_activation"],
            conv_padding=defaults["conv_padding"],
            conv_strides=_pair(defaults["conv_strides"]),
            pool_padding=defaults["pool_padding"],
            pool_size=_pair(defaults["pool_size"]),
            pool_strides=_pair(defaults["pool_strides"]),
            dense_units=tuple(int(unit) for unit in defaults["dense_units"]),
        ),
        experiment_grid=CNNExperimentGrid(
            conv_layer_counts=tuple(int(value) for value in grid["conv_layer_counts"]),
            filter_profiles=tuple(
                tuple(int(value) for value in profile)
                for profile in grid["filter_profiles"]
            ),
            kernel_profiles=tuple(
                tuple(_pair(kernel) for kernel in profile)
                for profile in grid["kernel_profiles"]
            ),
            pooling_types=tuple(grid["pooling_types"]),
        ),
        output=CNNOutputConfig(
            artifacts_dir=_resolve(root, output["artifacts_dir"]),
            models_dir=_resolve(root, output["models_dir"]),
            reports_dir=_resolve(root, output["reports_dir"]),
        ),
    )


def _validate_dataset_layout(config: CNNConfig) -> None:
    for directory in (config.data.train_dir, config.data.test_dir, config.data.pred_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Configured dataset directory does not exist: {directory}")

    for split_dir in (config.data.train_dir, config.data.test_dir):
        missing = [
            class_name
            for class_name in config.data.class_names
            if not (split_dir / class_name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"{split_dir} is missing class folders: {', '.join(missing)}"
            )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _pair(values: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"Expected a pair of values, got {values}.")
    return int(values[0]), int(values[1])
