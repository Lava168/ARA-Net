#!/usr/bin/env python3
"""Create public demo base-model probability streams from synthetic features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import LOCKED_BASE_MODEL_STREAMS


BASE_PROBS = {
    "CN": (0.82, 0.14, 0.04),
    "MCI": (0.08, 0.84, 0.08),
    "AD": (0.10, 0.28, 0.62),
}


def adjusted(prob: tuple[float, float, float], index: int) -> tuple[float, float, float]:
    cn, mci, ad = prob
    shift = (index - 2.5) * 0.01
    values = [max(0.01, cn - shift), max(0.01, mci), max(0.01, ad + shift)]
    total = sum(values)
    return tuple(value / total for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/expected_results/demo_atlas_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/demo_probability_streams.csv"))
    args = parser.parse_args()

    with args.features.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "scan_id", "dataset", "split", "y_true"]
    for model in LOCKED_BASE_MODEL_STREAMS:
        fieldnames += [f"{model}__prob_CN", f"{model}__prob_MCI", f"{model}__prob_AD"]

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in ["subject_id", "scan_id", "dataset", "split", "y_true"]}
            for index, model in enumerate(LOCKED_BASE_MODEL_STREAMS):
                cn, mci, ad = adjusted(BASE_PROBS[row["y_true"]], index)
                out[f"{model}__prob_CN"] = f"{cn:.6f}"
                out[f"{model}__prob_MCI"] = f"{mci:.6f}"
                out[f"{model}__prob_AD"] = f"{ad:.6f}"
            writer.writerow(out)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
