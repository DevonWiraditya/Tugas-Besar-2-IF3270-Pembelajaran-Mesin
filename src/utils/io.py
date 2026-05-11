from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict, path: str | Path, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent)


def write_csv(
    path: str | Path,
    header: Sequence[str],
    rows: Iterable[Sequence],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_history_csv(history: dict[str, list[float]], path: str | Path) -> None:
    keys = list(history.keys())
    rows = ([index, *values] for index, values in enumerate(zip(*history.values()), start=1))
    write_csv(path, ["epoch", *keys], rows)
