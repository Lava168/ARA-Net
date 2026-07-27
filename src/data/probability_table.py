"""CSV helpers for RC-SPE probability-stream inputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUIRED_META_COLUMNS = ("subject_id",)
OPTIONAL_META_COLUMNS = ("scan_id", "dataset", "split", "y_true")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def probability_column_names(model: str, classes: Sequence[str]) -> list[str]:
    return [f"{model}__prob_{cls}" for cls in classes]


def required_probability_columns(base_models: Iterable[str], classes: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for model in base_models:
        columns.extend(probability_column_names(model, classes))
    return columns


def validate_probability_table(rows: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> None:
    if not rows:
        raise ValueError("Input probability table has no rows.")
    columns = set(rows[0].keys())
    for column in REQUIRED_META_COLUMNS:
        if column not in columns:
            raise ValueError(f"Missing required metadata column: {column}")
    classes = list(config["classes"])
    base_models = list(config["base_models"])
    missing = [
        column
        for column in required_probability_columns(base_models, classes)
        if column not in columns
    ]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise ValueError(f"Missing required probability columns: {preview}{suffix}")
