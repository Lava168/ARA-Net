#!/usr/bin/env python3
"""Summarize v4 rescue results into a compact JSON/Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.load(path.open()) if path.exists() else None


def fmt_metric(m: dict) -> str:
    if not m:
        return "NA"
    parts = [
        f"n={m.get('n')}",
        f"Acc={m.get('acc'):.3f}" if m.get("acc") is not None else "Acc=NA",
        f"BAcc={m.get('balanced_acc'):.3f}" if m.get("balanced_acc") is not None else "BAcc=NA",
        f"AUC={m.get('macro_auc_ovr'):.3f}" if m.get("macro_auc_ovr") is not None else "AUC=NA",
    ]
    if m.get("ad_vs_cn_auc") is not None:
        parts.append(f"ADvCN_AUC={m.get('ad_vs_cn_auc'):.3f}")
    if m.get("cn_retention_rate") is not None:
        parts.append(f"CN_retention={m.get('cn_retention_rate'):.3f}")
    pred = m.get("prediction_distribution")
    if pred:
        parts.append(f"pred={pred}")
    return ", ".join(parts)


def fmt_recalls(m: dict) -> str:
    per_class = m.get("per_class") or {}
    if not per_class:
        return ""
    values = []
    for name in ["CN", "MCI", "AD"]:
        recall = per_class.get(name, {}).get("recall")
        if recall is not None:
            values.append(f"{name}={recall:.3f}")
    return "recall(" + ", ".join(values) + ")" if values else ""


def best_deep_from_dir(path: Path):
    summary = load_json(path / "summary.json")
    if summary:
        history = summary.get("history") or []
        best_epoch = summary.get("best_epoch")
        best_metrics = summary.get("best_metrics")
        if not best_metrics and history:
            best_row = max(history, key=lambda row: row.get("selection_score", -1e9))
            best_metrics = best_row.get("metrics", {})
            best_epoch = best_row.get("epoch")
        return {
            "epochs": len(history),
            "best_epoch": best_epoch,
            "best_score": summary.get("best_selection_score"),
            "best_metrics": best_metrics or {},
            "last": history[-1] if history else None,
        }
    history = load_json(path / "history.json")
    if history:
        best = max(history, key=lambda row: row["selection_score"])
        return {
            "epochs": len(history),
            "best_epoch": best["epoch"],
            "best_score": best["selection_score"],
            "best_metrics": best["metrics"],
            "last": history[-1],
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = args.v4_root
    manifest = load_json(root / "manifest_v4_summary.json")
    old_v3 = load_json(root.parent / "analysis" / "external_validation_v3_merged.json")
    feature = load_json(root / "atlas_feature_baseline" / "summary.json")
    biomarker = load_json(root / "atlas_feature_biomarkers" / "summary.json")
    cascade = load_json(root / "atlas_cascade_baseline" / "summary.json")
    hybrid = load_json(root / "hybrid_atlas_clinical_baseline" / "summary.json")

    deep = {}
    for path in sorted(root.glob("aranet_v4_*seed42")):
        item = best_deep_from_dir(path)
        if item:
            deep[path.name] = item

    summary = {
        "manifest": manifest,
        "old_v3_external": old_v3.get("ensemble") if old_v3 else None,
        "deep_v4": deep,
        "atlas_feature_baseline": feature,
        "atlas_cascade_baseline": cascade,
        "hybrid_atlas_clinical_baseline": hybrid,
        "atlas_feature_biomarkers": biomarker,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# V4 AD Rescue Progress Summary", ""]
    if old_v3:
        lines += ["## Old v3 external baseline", ""]
        for split, m in old_v3["ensemble"].items():
            lines.append(f"- {split}: {fmt_metric({'n': m.get('n_samples'), 'acc': m.get('accuracy'), 'balanced_acc': m.get('balanced_accuracy_present'), 'macro_auc_ovr': m.get('macro_auc_ovr_valid'), 'cn_retention_rate': m.get('ixi_cn_retention_rate'), 'prediction_distribution': m.get('prediction_distribution')})}")
        lines.append("")
    if feature:
        lines += ["## Atlas feature baseline", ""]
        lines.append(f"Best model: `{feature['best_model']}` score={feature['best_score']:.4f}")
        best = feature["results"][feature["best_model"]]["metrics"]
        for split in ["val", "internal_test", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]:
            if split in best:
                lines.append(f"- {split}: {fmt_metric(best[split])}; {fmt_recalls(best[split])}")
        lines.append("")
    if biomarker:
        lines += ["## Atlas biomarker validation", ""]
        for group in ["aibl_heldout", "aibl_adapt_heldout", "all_labeled_ad", "adni_val_internal_test"]:
            if group in biomarker:
                s = biomarker[group]["ad_key_volume_score"]
                ci = s.get("bootstrap_ci", [])
                lines.append(
                    f"- {group}: AD-key volume score={s.get('ad_key_score'):.3f}, "
                    f"uniform={s.get('uniform_null'):.3f}, delta={s.get('score_minus_uniform'):.3f}, "
                    f"CI=[{ci[0]:.3f}, {ci[1]:.3f}], p={s.get('permutation_p_greater'):.4f}"
                )
        lines.append("")
    if deep:
        lines += ["## Deep v4 status", ""]
        for name, item in deep.items():
            score = item.get("best_score")
            score_text = f"{score:.4f}" if score is not None else "NA"
            lines.append(f"- {name}: epochs={item['epochs']}, best_epoch={item['best_epoch']}, best_score={score_text}")
            for split in ["val", "aibl_heldout", "ixi_external"]:
                m = item["best_metrics"].get(split)
                if m:
                    lines.append(f"  - {split}: {fmt_metric(m)}; {fmt_recalls(m)}")
        lines.append("")
    if cascade:
        lines += ["## Atlas cascade baseline", ""]
        lines.append(f"Best model: `{cascade['best_model']}` score={cascade['best_score']:.4f}")
        best = cascade["results"][cascade["best_model"]]["metrics"]
        for split in ["val", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]:
            if split in best:
                lines.append(f"- {split}: {fmt_metric(best[split])}; {fmt_recalls(best[split])}")
    else:
        lines += ["## Atlas cascade baseline", "", "Still running or not available yet."]
    lines.append("")
    if hybrid:
        lines += ["## Hybrid atlas + clinical baseline", ""]
        for protocol, best_info in hybrid.get("best_by_protocol", {}).items():
            name = best_info.get("name")
            score = best_info.get("score")
            lines.append(f"Protocol `{protocol}` best: `{name}` score={score:.4f}")
            metrics = hybrid["results"][protocol][name]["metrics"]
            for split in ["val", "internal_test", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]:
                if split in metrics:
                    lines.append(f"- {split}: {fmt_metric(metrics[split])}; {fmt_recalls(metrics[split])}")
            lines.append("")
    else:
        lines += ["## Hybrid atlas + clinical baseline", "", "Still running or not available yet.", ""]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
