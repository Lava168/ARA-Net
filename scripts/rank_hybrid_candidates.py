#!/usr/bin/env python3
"""Rank hybrid candidates by external heldout behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def val(metric: dict, key: str) -> float:
    value = metric.get(key)
    return float(value) if value is not None else 0.0


def recall(metric: dict, name: str) -> float:
    return float(metric.get("per_class", {}).get(name, {}).get("recall") or 0.0)


def row_for(protocol: str, run_name: str, item: dict) -> dict:
    metrics = item["metrics"]
    aibl = metrics.get("aibl_heldout", {})
    ixi = metrics.get("ixi_external", {})
    internal = metrics.get("internal_test", {})
    oasis = metrics.get("oasis_external", {})
    return {
        "protocol": protocol,
        "run": run_name,
        "selection_score": float(item.get("selection_score") or 0.0),
        "aibl_acc": val(aibl, "acc"),
        "aibl_bacc": val(aibl, "balanced_acc"),
        "aibl_auc": val(aibl, "macro_auc_ovr"),
        "aibl_adcn_auc": val(aibl, "ad_vs_cn_auc"),
        "aibl_recall_cn": recall(aibl, "CN"),
        "aibl_recall_mci": recall(aibl, "MCI"),
        "aibl_recall_ad": recall(aibl, "AD"),
        "ixi_cn_retention": val(ixi, "cn_retention_rate"),
        "internal_bacc": val(internal, "balanced_acc"),
        "oasis_bacc": val(oasis, "balanced_acc"),
        "aibl_pred": aibl.get("prediction_distribution", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    summary = json.load(args.summary_json.open())
    rows = []
    for protocol, runs in summary["results"].items():
        for run_name, item in runs.items():
            rows.append(row_for(protocol, run_name, item))

    lines = ["# Hybrid Candidate Ranking", ""]
    for protocol in sorted(summary["results"]):
        protocol_rows = [row for row in rows if row["protocol"] == protocol]
        protocol_rows.sort(
            key=lambda row: (
                row["aibl_bacc"],
                row["ixi_cn_retention"],
                row["aibl_auc"],
                row["internal_bacc"],
            ),
            reverse=True,
        )
        lines += [f"## {protocol}", ""]
        lines.append(
            "| rank | run | AIBL BAcc | AIBL AUC | CN/MCI/AD recall | IXI CN retention | Internal BAcc | OASIS BAcc | pred |"
        )
        lines.append("|---:|---|---:|---:|---|---:|---:|---:|---|")
        for rank, row in enumerate(protocol_rows[: args.top_k], 1):
            recalls = f"{row['aibl_recall_cn']:.3f}/{row['aibl_recall_mci']:.3f}/{row['aibl_recall_ad']:.3f}"
            lines.append(
                f"| {rank} | `{row['run']}` | {row['aibl_bacc']:.3f} | {row['aibl_auc']:.3f} | "
                f"{recalls} | {row['ixi_cn_retention']:.3f} | {row['internal_bacc']:.3f} | "
                f"{row['oasis_bacc']:.3f} | {row['aibl_pred']} |"
            )
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
