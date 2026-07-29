#!/usr/bin/env python3
"""Create a manifest for the public manuscript figure assets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def figure_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"figure(\d+)", path.name)
    return (int(match.group(1)) if match else 999, path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("assets/manuscript_figures"))
    parser.add_argument("--output", type=Path, default=Path("outputs/expected_results/figure_manifest.json"))
    args = parser.parse_args()
    figures = []
    for pattern in ("*.svg", "*.png", "*.jpg", "*.jpeg", "*.webp"):
        figures.extend(args.figure_dir.glob(pattern))
    figures = [str(path) for path in sorted(figures, key=figure_sort_key)]
    payload = {
        "figure_dir": str(args.figure_dir),
        "n_figures": len(figures),
        "formats": sorted({Path(path).suffix.lstrip(".") for path in figures}),
        "note": "Final README figures are manuscript-level public SVG assets.",
        "figures": figures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
