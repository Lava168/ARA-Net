#!/usr/bin/env python3
"""Export the public RC-SPE ablation summary used by the manuscript package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATION_ROWS = [
    {"comparison": "Best single base model", "aibl_balanced_accuracy": 0.756, "mci_recall": 0.571, "ad_recall": 0.741, "ixi_cn_retention": 0.997},
    {"comparison": "Equal log-pooling", "aibl_balanced_accuracy": 0.648, "mci_recall": 0.171, "ad_recall": 0.778, "ixi_cn_retention": 1.000},
    {"comparison": "Full RC-SPE subject-level", "aibl_balanced_accuracy": 0.833, "mci_recall": 0.686, "ad_recall": 0.852, "ixi_cn_retention": 1.000},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/rc_spe_ablation_summary.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": ABLATION_ROWS}, indent=2), encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
