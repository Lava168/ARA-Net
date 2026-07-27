#!/usr/bin/env python3
"""Validate and export the locked public RC-SPE configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_csv_rows, validate_probability_table
from src.fusion import load_ensemble_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("examples/demo_probability_input.csv"))
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/locked_rc_spe_config.json"))
    args = parser.parse_args()

    config = load_ensemble_config(args.config)
    rows = read_csv_rows(args.input_csv)
    validate_probability_table(rows, config)
    payload = {
        "source_config": str(args.config),
        "n_rows_validated": len(rows),
        "locked_rc_spe": config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
