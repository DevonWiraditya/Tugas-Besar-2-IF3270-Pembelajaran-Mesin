from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.experiments import iter_cnn_experiments
from src.cnn.keras_models import build_shared_cnn_model
from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CNN shared experiment results.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--details-path", default="reports/cnn/shared_experiment_details.csv")
    parser.add_argument("--effects-path", default="reports/cnn/shared_hyperparameter_effects.csv")
    parser.add_argument("--best-path", default="reports/cnn/best_shared_architecture.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=False)
    summary = read_summary(Path(args.summary_path))
    experiments = iter_cnn_experiments(config)
    rows = [detail_row(config, experiment, summary[experiment.run_id]) for experiment in experiments]
    rows = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    effect_rows = []
    for factor in ("conv_layer_count", "filter_profile", "kernel_profile", "pooling_type"):
        effect_rows.extend(group_effects(rows, factor))

    write_csv(args.details_path, detail_header(), [[row[key] for key in detail_header()] for row in rows])
    write_csv(
        args.effects_path,
        effect_header(),
        [[row[key] for key in effect_header()] for row in effect_rows],
    )
    write_csv(args.best_path, detail_header(), [[rows[0][key] for key in detail_header()]])

    print(f"best_run_id={rows[0]['run_id']}")
    print(f"best_macro_f1={float(rows[0]['macro_f1']):.6f}")
    print(f"details_path={args.details_path}")
    print(f"effects_path={args.effects_path}")
    print(f"best_path={args.best_path}")


def detail_row(config, experiment, metrics: dict) -> dict:
    model = build_shared_cnn_model(config, experiment)
    return {
        "rank": "",
        "run_id": experiment.run_id,
        "status": metrics["status"],
        "conv_layer_count": experiment.conv_layer_count,
        "filter_profile": "-".join(str(value) for value in experiment.filters),
        "kernel_profile": "-".join(f"{height}x{width}" for height, width in experiment.kernel_sizes),
        "pooling_type": experiment.pooling_type,
        "params": model.count_params(),
        "loss": metrics["loss"],
        "sparse_categorical_accuracy": metrics["sparse_categorical_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "model_dir": metrics["model_dir"],
    }


def group_effects(rows: list[dict], factor: str) -> list[dict]:
    values = sorted({str(row[factor]) for row in rows})
    results = []
    for value in values:
        selected = [row for row in rows if str(row[factor]) == value]
        best = max(selected, key=lambda row: float(row["macro_f1"]))
        results.append(
            {
                "factor": factor,
                "value": value,
                "experiment_count": len(selected),
                "mean_macro_f1": statistics.fmean(float(row["macro_f1"]) for row in selected),
                "max_macro_f1": float(best["macro_f1"]),
                "mean_accuracy": statistics.fmean(
                    float(row["sparse_categorical_accuracy"]) for row in selected
                ),
                "mean_loss": statistics.fmean(float(row["loss"]) for row in selected),
                "best_run_id": best["run_id"],
            }
        )
    return results


def read_summary(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return {row["run_id"]: row for row in rows}


def detail_header() -> list[str]:
    return [
        "rank",
        "run_id",
        "status",
        "conv_layer_count",
        "filter_profile",
        "kernel_profile",
        "pooling_type",
        "params",
        "loss",
        "sparse_categorical_accuracy",
        "macro_f1",
        "model_dir",
    ]


def effect_header() -> list[str]:
    return [
        "factor",
        "value",
        "experiment_count",
        "mean_macro_f1",
        "max_macro_f1",
        "mean_accuracy",
        "mean_loss",
        "best_run_id",
    ]


if __name__ == "__main__":
    main()
