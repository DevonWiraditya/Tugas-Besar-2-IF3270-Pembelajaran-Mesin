from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnn.config import load_cnn_config
from src.cnn.evaluation import select_best_run_id
from src.cnn.experiments import get_cnn_experiment, run_id_for_parameter_sharing
from src.cnn.keras_models import build_cnn_model
from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN shared and non-shared parameter variants.")
    parser.add_argument("--config", default="configs/cnn/base.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--shared-summary-path", default="reports/cnn/shared_training_summary.csv")
    parser.add_argument("--nonshared-summary-path", default="reports/cnn/nonshared_training_summary.csv")
    parser.add_argument("--output-path", default="reports/cnn/parameter_sharing_comparison.csv")
    parser.add_argument("--require-nonshared", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cnn_config(args.config, check_data=False)
    base_run_id = args.run_id or select_best_run_id(Path(args.shared_summary_path))
    experiment = get_cnn_experiment(config, base_run_id)
    shared_run_id = run_id_for_parameter_sharing(experiment, "shared")
    nonshared_run_id = run_id_for_parameter_sharing(experiment, "nonshared")

    shared_rows = read_summary(Path(args.shared_summary_path))
    nonshared_rows = read_summary(Path(args.nonshared_summary_path))
    shared_metrics = shared_rows.get(shared_run_id)
    nonshared_metrics = nonshared_rows.get(nonshared_run_id)

    if shared_metrics is None:
        raise ValueError(f"Missing shared metrics for {shared_run_id}.")
    if args.require_nonshared and nonshared_metrics is None:
        raise ValueError(f"Missing non-shared metrics for {nonshared_run_id}.")

    rows = [
        comparison_row(config, experiment, "shared", shared_metrics),
        comparison_row(config, experiment, "nonshared", nonshared_metrics),
    ]
    header = [
        "parameter_sharing",
        "run_id",
        "base_run_id",
        "status",
        "model_dir",
        "params",
        "loss",
        "sparse_categorical_accuracy",
        "macro_f1",
    ]

    write_csv(args.output_path, header, [[row.get(key, "") for key in header] for row in rows])
    for row in rows:
        print(
            f"{row['run_id']}: "
            f"status={row.get('status', 'missing')}, "
            f"params={row['params']}, "
            f"macro_f1={row.get('macro_f1', '')}"
        )
    print(f"output_path={args.output_path}")


def comparison_row(config, experiment, parameter_sharing: str, metrics: dict | None) -> dict:
    model = build_cnn_model(config, experiment, parameter_sharing=parameter_sharing)
    run_id = run_id_for_parameter_sharing(experiment, parameter_sharing)
    row = {
        "parameter_sharing": parameter_sharing,
        "run_id": run_id,
        "base_run_id": experiment.run_id,
        "status": "missing",
        "model_dir": (config.output.models_dir / run_id).as_posix(),
        "params": model.count_params(),
    }
    if metrics:
        row.update(metrics)
    return row


def read_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return {row["run_id"]: row for row in rows}


if __name__ == "__main__":
    main()
