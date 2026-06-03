#!/usr/bin/env python3
"""Summarize targeted hybrid replicate runs across seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np

from train_atlas_feature_baseline import CLASS_NAMES, classification_metrics


SPLITS = ["aibl_adapt_val", "aibl_heldout", "internal_test", "ixi_external", "oasis_external"]
METRICS = ["acc", "balanced_acc", "macro_auc_ovr", "ad_vs_cn_auc", "cn_retention_rate"]
RUNS = {
    "atlas_core_clinical__hgb",
    "atlas_biomarker_enhanced__hgb",
    "clinical_core_only__rf_balanced",
}


def load(path: Path):
    return json.load(path.open()) if path.exists() else None


def prediction_metrics(base_dir: Path, run_name: str) -> Dict[str, dict] | None:
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    metrics = {}
    for split in SPLITS:
        path = base_dir / f"aibl_adapted_{run_name}_{split}_predictions.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        y = [label_map[row["y_true"]] for row in rows]
        probs = np.array(
            [[float(row["prob_CN"]), float(row["prob_MCI"]), float(row["prob_AD"])] for row in rows],
            dtype=float,
        )
        metrics[split] = classification_metrics(y, probs)
    return metrics if metrics else None


def metric_row(seed: int, run_name: str, metrics: Dict[str, dict]) -> dict:
    row = {"seed": seed, "run": run_name}
    for split in SPLITS:
        item = metrics.get(split, {})
        for metric in METRICS:
            row[f"{split}.{metric}"] = item.get(metric)
        for cls in ["CN", "MCI", "AD"]:
            row[f"{split}.recall_{cls}"] = item.get("per_class", {}).get(cls, {}).get("recall")
    return row


def summarize_values(rows: List[dict]) -> Dict[str, dict]:
    keys = sorted(k for row in rows for k in row if k not in {"seed", "run"})
    out = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            out[key] = {
                "mean": float(mean(values)),
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    seed_values = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows_by_run: Dict[str, List[dict]] = {}
    missing = []
    partial = []
    for seed in seed_values:
        base_dir = args.v4_root / ("hybrid_atlas_clinical_baseline" if seed == 42 else f"hybrid_replicates_seed{seed}")
        path = base_dir / "summary.json"
        summary = load(path)
        if summary:
            protocol = "aibl_adapted"
            for run_name, item in summary.get("results", {}).get(protocol, {}).items():
                if run_name in RUNS:
                    rows_by_run.setdefault(run_name, []).append(metric_row(seed, run_name, item["metrics"]))
            continue

        found_any = False
        for run_name in RUNS:
            metrics = prediction_metrics(base_dir, run_name)
            if metrics:
                rows_by_run.setdefault(run_name, []).append(metric_row(seed, run_name, metrics))
                partial.append({"seed": seed, "run": run_name, "source": "prediction_csv"})
                found_any = True
        if not found_any:
            missing.append({"seed": seed, "path": str(path)})

    summary_out = {
        "seeds_requested": seed_values,
        "missing": missing,
        "partial_from_prediction_csv": partial,
        "runs": {
            run: {
                "rows": rows,
                "summary": summarize_values(rows),
            }
            for run, rows in sorted(rows_by_run.items())
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    lines = ["# Hybrid Replicate Summary", ""]
    if missing:
        lines += ["## Missing", ""]
        for item in missing:
            lines.append(f"- seed {item['seed']}: {item['path']}")
        lines.append("")
    if partial:
        lines += ["## Partial Prediction CSV Sources", ""]
        for item in partial:
            lines.append(f"- seed {item['seed']} `{item['run']}` from {item['source']}")
        lines.append("")
    for run, item in sorted(summary_out["runs"].items()):
        lines += [f"## {run}", ""]
        stats = item["summary"]
        for key in [
            "aibl_heldout.acc",
            "aibl_heldout.balanced_acc",
            "aibl_heldout.macro_auc_ovr",
            "aibl_heldout.ad_vs_cn_auc",
            "aibl_heldout.recall_CN",
            "aibl_heldout.recall_MCI",
            "aibl_heldout.recall_AD",
            "ixi_external.cn_retention_rate",
            "internal_test.balanced_acc",
            "oasis_external.balanced_acc",
        ]:
            if key in stats:
                value = stats[key]
                lines.append(f"- {key}: {value['mean']:.3f} +/- {value['std']:.3f} (n={value['n']})")
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
