from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze captioning experiment results.")
    parser.add_argument("--summary-path", default="reports/captioning/experiment_summary.csv")
    parser.add_argument("--ranked-path", default="reports/captioning/ranked_runs.csv")
    parser.add_argument("--effects-path", default="reports/captioning/hyperparameter_effects.csv")
    parser.add_argument("--history-path", default="reports/captioning/training_history_long.csv")
    parser.add_argument("--best-rnn-path", default="reports/captioning/best_rnn.csv")
    parser.add_argument("--best-lstm-path", default="reports/captioning/best_lstm.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    rows = load_or_build_summary(summary_path)
    rows = sorted(rows, key=lambda row: float(row["bleu4"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    write_csv(args.ranked_path, ranked_header(), [[row.get(key, "") for key in ranked_header()] for row in rows])
    write_csv(
        args.effects_path,
        effect_header(),
        [[row[key] for key in effect_header()] for row in build_effect_rows(rows)],
    )
    history_rows = collect_history_rows(rows)
    write_csv(args.history_path, history_header(), history_rows)

    best_rnn = next((row for row in rows if row["decoder_type"] == "rnn"), None)
    best_lstm = next((row for row in rows if row["decoder_type"] == "lstm"), None)
    if best_rnn:
        write_csv(args.best_rnn_path, ranked_header(), [[best_rnn.get(key, "") for key in ranked_header()]])
    if best_lstm:
        write_csv(args.best_lstm_path, ranked_header(), [[best_lstm.get(key, "") for key in ranked_header()]])


def load_or_build_summary(path: Path) -> list[dict]:
    rows = build_summary_from_models()
    if rows:
        write_csv(
            path,
            summary_header(),
            [[row.get(key, "") for key in summary_header()] for row in rows],
        )
        return rows
    if path.exists():
        return normalize_rows(read_csv_rows(path))
    if not rows:
        raise FileNotFoundError(
            f"Summary file not found and no evaluated model metrics were found: {path}"
        )
    return rows


def build_summary_from_models() -> list[dict]:
    model_root = ROOT / "models" / "captioning"
    rows: list[dict] = []
    if not model_root.exists():
        return rows

    for run_dir in sorted(model_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_files = [run_dir / "metrics.test.json"] if (run_dir / "metrics.test.json").exists() else []
        experiment_path = run_dir / "experiment.json"
        if not metrics_files or not experiment_path.exists():
            continue

        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        for metrics_path in metrics_files:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": experiment["run_id"],
                    "decoder_type": experiment["recurrent_type"],
                    "layer_count": experiment["layer_count"],
                    "hidden_size": experiment["hidden_size"],
                    "split": "test",
                    "bleu4": metrics["bleu4"],
                    "meteor": metrics["meteor"],
                    "runtime_seconds": metrics["runtime_seconds"],
                    "model_dir": _portable_path(run_dir),
                }
            )
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize_rows(rows: list[dict]) -> list[dict]:
    keys = set(summary_header())
    normalized = []
    for row in rows:
        normalized.append({key: row.get(key, "") for key in keys})
    return normalized


def build_effect_rows(rows: list[dict]) -> list[dict]:
    results: list[dict] = []
    for factor in ("decoder_type", "layer_count", "hidden_size"):
        values = sorted({str(row[factor]) for row in rows})
        for value in values:
            selected = [row for row in rows if str(row[factor]) == value]
            best = max(selected, key=lambda row: float(row["bleu4"]))
            results.append(
                {
                    "factor": factor,
                    "value": value,
                    "experiment_count": len(selected),
                    "mean_bleu4": statistics.fmean(float(row["bleu4"]) for row in selected),
                    "mean_meteor": statistics.fmean(float(row["meteor"]) for row in selected),
                    "mean_runtime_seconds": statistics.fmean(float(row["runtime_seconds"]) for row in selected),
                    "best_run_id": best["run_id"],
                }
            )
    return results


def collect_history_rows(rows: list[dict]) -> list[list]:
    output: list[list] = []
    seen_run_ids: set[str] = set()
    for row in rows:
        run_id = row["run_id"]
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        history_path = Path(row["model_dir"]) / "history.csv"
        if not history_path.exists():
            continue
        for history_row in read_csv_rows(history_path):
            output.append(
                [
                    row["run_id"],
                    row["decoder_type"],
                    row["layer_count"],
                    row["hidden_size"],
                    history_row.get("epoch", ""),
                    history_row.get("loss", ""),
                    history_row.get("sparse_categorical_accuracy", ""),
                    history_row.get("val_loss", ""),
                    history_row.get("val_sparse_categorical_accuracy", ""),
                ]
            )
    return output


def summary_header() -> list[str]:
    return [
        "run_id",
        "decoder_type",
        "layer_count",
        "hidden_size",
        "split",
        "bleu4",
        "meteor",
        "runtime_seconds",
        "model_dir",
    ]


def ranked_header() -> list[str]:
    return ["rank", *summary_header()]


def effect_header() -> list[str]:
    return [
        "factor",
        "value",
        "experiment_count",
        "mean_bleu4",
        "mean_meteor",
        "mean_runtime_seconds",
        "best_run_id",
    ]


def history_header() -> list[str]:
    return [
        "run_id",
        "decoder_type",
        "layer_count",
        "hidden_size",
        "epoch",
        "loss",
        "sparse_categorical_accuracy",
        "val_loss",
        "val_sparse_categorical_accuracy",
    ]


def _portable_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return value.as_posix()


if __name__ == "__main__":
    main()
