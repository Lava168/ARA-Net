#!/usr/bin/env python3
"""Create a manifest for the public manuscript figure assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/v6_final_model/manual_paper_figures"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/figure_manifest.json"))
    args = parser.parse_args()
    figures = sorted(str(path) for path in args.figure_dir.glob("*.png"))
    payload = {
        "figure_dir": str(args.figure_dir),
        "n_figures": len(figures),
        "figures": figures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
