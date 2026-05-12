from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from src.cnn.config import CNNConfig


@dataclass(frozen=True)
class CNNExperiment:
    run_id: str
    conv_layer_count: int
    filters: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, int], ...]
    pooling_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def iter_cnn_experiments(config: CNNConfig) -> tuple[CNNExperiment, ...]:
    experiments: list[CNNExperiment] = []
    grid = config.experiment_grid

    for conv_count, filter_profile, kernel_profile, pooling_type in product(
        grid.conv_layer_counts,
        grid.filter_profiles,
        grid.kernel_profiles,
        grid.pooling_types,
    ):
        filters = filter_profile[:conv_count]
        kernels = kernel_profile[:conv_count]
        run_id = _make_run_id(conv_count, filters, kernels, pooling_type)
        experiments.append(
            CNNExperiment(
                run_id=run_id,
                conv_layer_count=conv_count,
                filters=filters,
                kernel_sizes=kernels,
                pooling_type=pooling_type,
            )
        )

    return tuple(experiments)


def get_cnn_experiment(config: CNNConfig, run_id: str) -> CNNExperiment:
    for experiment in iter_cnn_experiments(config):
        if experiment.run_id == run_id:
            return experiment
    raise ValueError(f"Unknown CNN experiment run_id: {run_id}")


def _make_run_id(
    conv_count: int,
    filters: tuple[int, ...],
    kernels: tuple[tuple[int, int], ...],
    pooling_type: str,
) -> str:
    filters_part = "-".join(str(value) for value in filters)
    kernels_part = "-".join(f"{height}x{width}" for height, width in kernels)
    return f"shared_l{conv_count}_f{filters_part}_k{kernels_part}_{pooling_type}"
