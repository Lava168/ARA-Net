#!/usr/bin/env python3
"""Export manuscript-ready tables from v4 JSON summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path):
    return json.load(path.open())


def recall(m: dict, label: str) -> float:
    return float(m.get("per_class", {}).get(label, {}).get("recall") or 0.0)


def metric_row(model: str, evaluation: str, m: dict, note: str) -> dict:
    if "accuracy" in m:
        return {
            "model": model,
            "evaluation": evaluation,
            "acc": m.get("accuracy"),
            "balanced_acc": m.get("balanced_accuracy_present"),
            "macro_auc": m.get("macro_auc_ovr_valid"),
            "ad_vs_cn_auc": "",
            "cn_retention": m.get("ixi_cn_retention_rate", ""),
            "recall_cn": "",
            "recall_mci": "",
            "recall_ad": "",
            "note": note,
        }
    return {
        "model": model,
        "evaluation": evaluation,
        "acc": m.get("acc"),
        "balanced_acc": m.get("balanced_acc"),
        "macro_auc": m.get("macro_auc_ovr", ""),
        "ad_vs_cn_auc": m.get("ad_vs_cn_auc", ""),
        "cn_retention": m.get("cn_retention_rate", ""),
        "recall_cn": recall(m, "CN"),
        "recall_mci": recall(m, "MCI"),
        "recall_ad": recall(m, "AD"),
        "note": note,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if value is None:
                values.append("NA")
                continue
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.v4_root
    manifest = load(root / "manifest_v4_summary.json")
    old_v3 = load(root.parent / "analysis" / "external_validation_v3_merged.json")
    feature = load(root / "atlas_feature_baseline" / "summary.json")
    cascade = load(root / "atlas_cascade_baseline" / "summary.json")
    hybrid = load(root / "hybrid_atlas_clinical_baseline" / "summary.json")
    biomarker = load(root / "atlas_feature_biomarkers" / "summary.json")
    reps = load(root / "hybrid_replicate_summary.json")

    table1 = []
    roles = {
        "train": "ADNI training",
        "val": "ADNI validation",
        "internal_test": "ADNI internal test",
        "aibl_adapt_train": "AIBL adaptation training",
        "aibl_adapt_val": "AIBL adaptation validation",
        "aibl_heldout": "Locked AIBL heldout external test",
        "oasis_external": "OASIS external stress test",
        "ixi_external": "IXI healthy negative-control cohort",
    }
    for split, role in roles.items():
        item = manifest["split_counts"][split]
        labels = item["labels"]
        table1.append({
            "split": split,
            "scans": item["scans"],
            "subjects": item["subjects"],
            "CN": labels.get("CN", 0),
            "MCI": labels.get("MCI", 0),
            "AD": labels.get("AD", 0),
            "role": role,
        })

    feature_best = feature["results"][feature["best_model"]]["metrics"]
    cascade_best = cascade["results"][cascade["best_model"]]["metrics"]
    h = hybrid["results"]
    table2 = [
        metric_row("Old v3 ensemble", "AIBL external", old_v3["ensemble"]["aibl"], "Failed external baseline"),
        metric_row("Old v3 ensemble", "IXI healthy", old_v3["ensemble"]["ixi"], "High false impairment rate"),
        metric_row("Atlas-only HGB", "AIBL heldout", feature_best["aibl_heldout"], "MRI-only baseline"),
        metric_row("Cascade RF-logreg", "AIBL heldout", cascade_best["aibl_heldout"], "High specificity but weak MCI"),
        metric_row("ADNI-only hybrid RF", "AIBL heldout", h["adni_only"]["atlas_biomarker_enhanced__rf_balanced"]["metrics"]["aibl_heldout"], "Zero-shot remains insufficient"),
        metric_row("Recommended atlas+clinical HGB", "AIBL heldout", h["aibl_adapted"]["atlas_core_clinical__hgb"]["metrics"]["aibl_heldout"], "Main atlas-guided multimodal model"),
        metric_row("Recommended atlas+clinical HGB", "IXI healthy", h["aibl_adapted"]["atlas_core_clinical__hgb"]["metrics"]["ixi_external"], "Healthy negative control"),
        metric_row("Clinical-only RF", "AIBL heldout", h["aibl_adapted"]["clinical_core_only__rf_balanced"]["metrics"]["aibl_heldout"], "Comparator / upper bound"),
        metric_row("Biomarker-enhanced HGB", "AIBL heldout", h["aibl_adapted"]["atlas_biomarker_enhanced__hgb"]["metrics"]["aibl_heldout"], "Sensitivity analysis"),
    ]

    table3 = []
    for run, item in reps["runs"].items():
        stats = item["summary"]
        table3.append({
            "run": run,
            "aibl_heldout_bacc_mean": stats["aibl_heldout.balanced_acc"]["mean"],
            "aibl_heldout_bacc_std": stats["aibl_heldout.balanced_acc"]["std"],
            "aibl_heldout_auc_mean": stats["aibl_heldout.macro_auc_ovr"]["mean"],
            "ixi_cn_retention_mean": stats["ixi_external.cn_retention_rate"]["mean"],
            "ixi_cn_retention_std": stats["ixi_external.cn_retention_rate"]["std"],
            "n": stats["aibl_heldout.balanced_acc"]["n"],
        })

    table4 = []
    for group in ["aibl_heldout", "aibl_adapt_heldout", "all_labeled_ad", "adni_val_internal_test"]:
        score = biomarker[group]["ad_key_volume_score"]
        ci = score["bootstrap_ci"]
        table4.append({
            "group": group,
            "ad_key_score": score["ad_key_score"],
            "uniform_null": score["uniform_null"],
            "delta": score["score_minus_uniform"],
            "ci_low": ci[0],
            "ci_high": ci[1],
            "permutation_p": score["permutation_p_greater"],
        })

    tables = {
        "table1_cohort_splits": table1,
        "table2_classification": table2,
        "table3_replicates": table3,
        "table4_neurodegeneration": table4,
    }
    for name, rows in tables.items():
        write_csv(args.out_dir / f"{name}.csv", rows)
        write_md(args.out_dir / f"{name}.md", rows)
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
