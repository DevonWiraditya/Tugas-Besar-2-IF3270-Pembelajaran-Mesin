from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run caption length comparison experiments.")
    parser.add_argument("--config", default="configs/captioning/base.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decoder-type", required=True, choices=["rnn", "lstm"])
    parser.add_argument("--length", action="append", type=int, default=None)
    parser.add_argument("--summary-path", default="reports/captioning/max_caption_length_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = args.length or [20, 30, 40]
    rows = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for length in lengths:
            config_path = write_temp_config(Path(args.config), temp_path / f"captioning_{length}.json", length)
            output_summary = temp_path / f"eval_{length}.csv"
            run_subprocess(
                [
                    sys.executable,
                    "src/scripts/evaluate_captioning.py",
                    "--config",
                    str(config_path),
                    "--run-id",
                    args.run_id,
                    "--split",
                    "test",
                    "--skip-model-save",
                    "--output-summary-path",
                    str(output_summary),
                ]
            )
            result = read_single_row(output_summary)
            rows.append(
                {
                    "max_caption_length": str(length),
                    "run_id": args.run_id,
                    "decoder_type": args.decoder_type,
                    "bleu4": result["bleu4"],
                    "meteor": result["meteor"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )

    output_path = ROOT / args.summary_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_rows = merge_rows(read_existing_rows(output_path), rows)
    write_rows(output_path, merged_rows)


def write_temp_config(source_path: Path, target_path: Path, max_caption_length: int) -> Path:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    raw["text"]["max_caption_length"] = int(max_caption_length)
    raw["data"]["dataset_root"] = str((ROOT / raw["data"]["dataset_root"]).resolve())
    raw["data"]["images_dir"] = str((ROOT / raw["data"]["images_dir"]).resolve())
    raw["data"]["captions_path"] = str((ROOT / raw["data"]["captions_path"]).resolve())
    raw["data"]["train_split_path"] = str((ROOT / raw["data"]["train_split_path"]).resolve())
    raw["data"]["validation_split_path"] = str((ROOT / raw["data"]["validation_split_path"]).resolve())
    raw["data"]["test_split_path"] = str((ROOT / raw["data"]["test_split_path"]).resolve())
    raw["output"]["artifacts_dir"] = str((ROOT / raw["output"]["artifacts_dir"]).resolve())
    raw["output"]["models_dir"] = str((ROOT / raw["output"]["models_dir"]).resolve())
    raw["output"]["reports_dir"] = str((ROOT / raw["output"]["reports_dir"]).resolve())
    target_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return target_path


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def read_single_row(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    values = lines[1].split(",")
    return dict(zip(header, values))


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def merge_rows(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in existing_rows:
        merged[row_key(row)] = row
    for row in new_rows:
        merged[row_key(row)] = row
    return sorted(
        merged.values(),
        key=lambda row: (row["decoder_type"], row["run_id"], int(row["max_caption_length"])),
    )


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["decoder_type"], row["run_id"], row["max_caption_length"]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "max_caption_length",
        "run_id",
        "decoder_type",
        "bleu4",
        "meteor",
        "runtime_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
