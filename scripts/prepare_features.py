#!/usr/bin/env python3
"""Prepare a public synthetic atlas-feature table.

This command is a reproducibility smoke-test entry point. It does not process
restricted MRI data. For full studies, replace the metadata input and feature
construction with site-local FreeSurfer/FastSurfer-derived features.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas import feature_row_from_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/example_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/demo_atlas_features.csv"))
    args = parser.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "scan_id",
        "dataset",
        "split",
        "y_true",
        "atlas_hippocampus_volume",
        "atlas_amygdala_volume",
        "atlas_lateral_ventricle_volume",
        "atlas_cortex_volume",
        "atlas_ad_like_z",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            feats = feature_row_from_label(row["y_true"])
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    "scan_id": row["scan_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "y_true": row["y_true"],
                    **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in feats.items()},
                }
            )
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
