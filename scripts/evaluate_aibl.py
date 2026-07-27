#!/usr/bin/env python3
"""Evaluate the public AIBL-style RC-SPE demo input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import run_rc_spe_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("examples/demo_probability_input.csv"))
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--unit", choices=["scan", "subject"], default="subject")
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/aibl_demo_predictions.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/expected_results/aibl_demo_metrics.json"))
    args = parser.parse_args()
    result = run_rc_spe_evaluation(args.input_csv, args.config, args.output, args.unit, args.metrics_json)
    print(f"[saved] {args.output}")
    print(f"[metrics] {result.get('metrics', 'labels unavailable')}")


if __name__ == "__main__":
    main()
