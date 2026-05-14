from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CNN training histories into report CSV files.")
    parser.add_argument("--shared-details-path", default="reports/cnn/shared_experiment_details.csv")
    parser.add_argument("--parameter-sharing-path", default="reports/cnn/parameter_sharing_comparison.csv")
    parser.add_argument("--shared-output-path", default="reports/cnn/shared_training_history.csv")
    parser.add_argument("--parameter-sharing-output-path", default="reports/cnn/parameter_sharing_history.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_details = read_csv_rows(Path(args.shared_details_path))
    parameter_sharing = read_csv_rows(Path(args.parameter_sharing_path))
    shared_rows = collect_shared_history(shared_details)
    parameter_rows = collect_parameter_sharing_history(parameter_sharing, shared_details)

    write_csv(args.shared_output_path, shared_header(), shared_rows)
    write_csv(args.parameter_sharing_output_path, parameter_header(), parameter_rows)

    print(f"shared_rows={len(shared_rows)}")
    print(f"parameter_sharing_rows={len(parameter_rows)}")
    print(f"shared_output_path={args.shared_output_path}")
    print(f"parameter_sharing_output_path={args.parameter_sharing_output_path}")


def collect_shared_history(details: list[dict]) -> list[list]:
    rows = []
    for detail in details:
        for history in read_history(detail):
            rows.append(
                [
                    detail["run_id"],
                    detail["conv_layer_count"],
                    detail["filter_profile"],
                    detail["kernel_profile"],
                    detail["pooling_type"],
                    detail["params"],
                    history["epoch"],
                    history["loss"],
                    history["sparse_categorical_accuracy"],
                    history["val_loss"],
                    history["val_sparse_categorical_accuracy"],
                ]
            )
    return rows


def collect_parameter_sharing_history(
    comparison_rows: list[dict],
    shared_details: list[dict],
) -> list[list]:
    details_by_run_id = {row["run_id"]: row for row in shared_details}
    rows = []
    for comparison in comparison_rows:
        detail = details_by_run_id.get(comparison["base_run_id"], {})
        for history in read_history(comparison):
            rows.append(
                [
                    comparison["parameter_sharing"],
                    comparison["run_id"],
                    comparison["base_run_id"],
                    detail.get("conv_layer_count", ""),
                    detail.get("filter_profile", ""),
                    detail.get("kernel_profile", ""),
                    detail.get("pooling_type", ""),
                    comparison["params"],
                    history["epoch"],
                    history["loss"],
                    history["sparse_categorical_accuracy"],
                    history["val_loss"],
                    history["val_sparse_categorical_accuracy"],
                ]
            )
    return rows


def read_history(row: dict) -> list[dict]:
    history_path = Path(row["model_dir"]) / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    return read_csv_rows(history_path)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def shared_header() -> list[str]:
    return [
        "run_id",
        "conv_layer_count",
        "filter_profile",
        "kernel_profile",
        "pooling_type",
        "params",
        "epoch",
        "loss",
        "sparse_categorical_accuracy",
        "val_loss",
        "val_sparse_categorical_accuracy",
    ]


def parameter_header() -> list[str]:
    return [
        "parameter_sharing",
        "run_id",
        "base_run_id",
        "conv_layer_count",
        "filter_profile",
        "kernel_profile",
        "pooling_type",
        "params",
        "epoch",
        "loss",
        "sparse_categorical_accuracy",
        "val_loss",
        "val_sparse_categorical_accuracy",
    ]


if __name__ == "__main__":
    main()
