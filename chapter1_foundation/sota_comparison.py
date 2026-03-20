#!/usr/bin/env python3
"""
SOTA comparison table for ADNI CN/MCI/AD 3-class classification.

Curated literature entries + your v3 results → LaTeX / Markdown table.

Selection criteria for fair comparison:
  - Structural MRI only (no PET, fMRI, or clinical scores as input)
  - 3-class: CN vs MCI vs AD
  - ADNI dataset (any version)
  - Cross-validation or held-out test (not single random split)

Usage:
    python -m chapter1_foundation.sota_comparison \
        --results chapter1_foundation/experiment_results_v3/aggregated.json \
        --output chapter1_foundation/sota_table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


# ── curated SOTA literature ──────────────────────────────────────────────────

LITERATURE = [
    # ── Rigorous CV / proper evaluation (fair comparison) ─────────────
    {
        "method": "3D-CNN",
        "reference": "Korolev et al.",
        "year": 2017,
        "venue": "ISBI",
        "modality": "sMRI (3D)",
        "n_subjects": "ADNI",
        "eval": "5-fold CV",
        "acc": 59.7,
        "bacc": None,
        "auc": None,
        "note": "Baseline 3D-CNN; 3-class",
    },
    {
        "method": "THAN",
        "reference": "Zhang et al.",
        "year": 2022,
        "venue": "Neurocomputing",
        "modality": "sMRI (3D)",
        "n_subjects": "ADNI",
        "eval": "5-fold CV",
        "acc": 62.9,
        "bacc": None,
        "auc": 0.65,
        "note": "Transformer-based hierarchical attention",
    },
    {
        "method": "STNet",
        "reference": "Jia et al.",
        "year": 2023,
        "venue": "MedIA",
        "modality": "sMRI (3D)",
        "n_subjects": "ADNI",
        "eval": "5-fold CV",
        "acc": 71.8,
        "bacc": None,
        "auc": None,
        "note": "Spatial-temporal network; 3-class",
    },
    {
        "method": "LSTM-Robust",
        "reference": "Gao et al.",
        "year": 2023,
        "venue": "MedIA",
        "modality": "sMRI (3D)",
        "n_subjects": "ADNI",
        "eval": "5-fold CV",
        "acc": 76.0,
        "bacc": None,
        "auc": None,
        "note": "Longitudinal LSTM; 3-class",
    },
    {
        "method": "ECAResNet269 + SMOTE/FL",
        "reference": "Alkhathami et al.",
        "year": 2025,
        "venue": "Sci. Rep.",
        "modality": "sMRI (2D coronal)",
        "n_subjects": 1346,
        "eval": "Patient-split",
        "acc": None,
        "bacc": 74.0,
        "auc": None,
        "note": "SMOTE + focal loss; reports BAcc",
    },
    {
        "method": "Ensemble 138 ViT",
        "reference": "Marzban et al.",
        "year": 2024,
        "venue": "Sci. Rep.",
        "modality": "sMRI (ROI 3D)",
        "n_subjects": "ADNI",
        "eval": "5-fold CV",
        "acc": None,
        "bacc": None,
        "auc": None,
        "note": "ROI-based 3D ViT ensemble; interpretable",
    },
    # ── Single-split / potentially inflated (shown for context) ──────
    {
        "method": "3D HCCT",
        "reference": "Majee et al.",
        "year": 2024,
        "venue": "arXiv",
        "modality": "sMRI (3D)",
        "n_subjects": "ADNI",
        "eval": "Single split",
        "acc": 96.1,
        "bacc": None,
        "auc": None,
        "note": "CNN+Transformer hybrid; no CV, likely data leakage",
    },
    {
        "method": "DEMNET",
        "reference": "Murugan et al.",
        "year": 2021,
        "venue": "Neural Comput. Appl.",
        "modality": "sMRI (2D)",
        "n_subjects": "ADNI+Kaggle",
        "eval": "Hold-out",
        "acc": 95.2,
        "bacc": None,
        "auc": None,
        "note": "Single split on Kaggle; no subject-level split",
    },
]


def _fmt(val, pct=False):
    if val is None:
        return "—"
    if pct:
        return f"{val:.1f}"
    return str(val)


def load_own_results(agg_path: Path) -> dict:
    with open(agg_path) as f:
        data = json.load(f)
    own = {}

    for src_key, label in [("individual", ""), ("ensemble", " (ens)")]:
        src = data.get(src_key, {})
        for model_name, stats in src.items():
            if not stats:
                continue
            bacc = stats.get("BAcc", {})
            acc = stats.get("Acc", {})
            auc = stats.get("AUC", {})
            own[model_name + label] = {
                "bacc_mean": bacc.get("mean", 0) * 100,
                "bacc_std": bacc.get("std", 0) * 100,
                "bacc_ci": (bacc.get("ci_lo", 0) * 100, bacc.get("ci_hi", 0) * 100),
                "acc_mean": acc.get("mean", 0) * 100,
                "acc_std": acc.get("std", 0) * 100,
                "auc_mean": auc.get("mean", 0),
                "auc_std": auc.get("std", 0),
                "n_runs": bacc.get("n_runs", bacc.get("n_folds", 0)),
            }
    return own


# ── markdown table ───────────────────────────────────────────────────────────

def generate_markdown(own: Optional[dict] = None) -> str:
    lines = [
        "| Method | Year | Modality | N | Eval | Acc (%) | BAcc (%) | AUC | Note |",
        "|--------|------|----------|---|------|---------|----------|-----|------|",
    ]

    for entry in LITERATURE:
        lines.append(
            f"| {entry['method']} | {entry['year']} | {entry['modality']} "
            f"| {entry['n_subjects']} | {entry['eval']} "
            f"| {_fmt(entry['acc'], True)} | {_fmt(entry['bacc'], True)} "
            f"| {_fmt(entry['auc'], True)} | {entry['note']} |"
        )

    lines.append("|--------|------|----------|---|------|---------|----------|-----|------|")

    if own:
        display_map = {
            "ARA-Net Ensemble": "**ARA-Net Ensemble**",
            "Ours (Atlas+AnatDist)": "ARA-Net (Full)",
            "Ours (no atlas)": "ARA-Net (−Atlas)",
        }
        for key in ["ARA-Net Ensemble",
                     "Ours (Atlas+AnatDist)", "Ours (no atlas)"]:
            if key not in own:
                continue
            o = own[key]
            display = display_map.get(key, key)
            bacc_str = f"{o['bacc_mean']:.1f}±{o['bacc_std']:.1f}"
            acc_str = f"{o['acc_mean']:.1f}±{o['acc_std']:.1f}"
            auc_str = f"{o['auc_mean']:.3f}±{o['auc_std']:.3f}"
            lines.append(
                f"| {display} | 2025 | sMRI (3D) | 2,401 | 6s×5f CV "
                f"| {acc_str} | {bacc_str} | {auc_str} | Ours |"
            )

    return "\n".join(lines)


# ── latex table ──────────────────────────────────────────────────────────────

def generate_latex(own: Optional[dict] = None) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Comparison with recent ADNI 3-class (CN/MCI/AD) classification methods. "
        r"$^\dagger$Multi-modal. $^\ddagger$Single split (no CV). "
        r"Bold: our best model.}",
        r"\label{tab:sota}",
        r"\begin{tabular}{llcccccl}",
        r"\toprule",
        r"Method & Year & Modality & $N$ & Eval & Acc (\%) & BAcc (\%) & AUC \\",
        r"\midrule",
    ]

    for entry in LITERATURE:
        acc = _fmt(entry["acc"], True)
        bacc = _fmt(entry["bacc"], True)
        auc = _fmt(entry["auc"], True)
        n = str(entry["n_subjects"])
        note = ""
        if "Multi-modal" in (entry.get("note") or ""):
            note = r"$^\dagger$"
        eval_lower = (entry.get("eval") or "").lower()
        if "single" in eval_lower or "hold" in eval_lower:
            note = r"$^\ddagger$"

        lines.append(
            f"  {entry['method']}{note} & {entry['year']} & "
            f"{entry['modality']} & {n} & {entry['eval']} & "
            f"{acc} & {bacc} & {auc} \\\\"
        )

    lines.append(r"\midrule")

    if own:
        for key, display, bold in [
            ("ARA-Net Ensemble", r"\textbf{ARA-Net Ensemble}", True),
            ("Ours (Atlas+AnatDist)", "ARA-Net (Full)", False),
            ("Ours (no atlas)", "ARA-Net ($-$Atlas)", False),
        ]:
            if key not in own:
                continue
            o = own[key]
            eval_str = "6s$\\times$5f CV"
            bacc_str = f"{o['bacc_mean']:.1f}$\\pm${o['bacc_std']:.1f}"
            acc_str = f"{o['acc_mean']:.1f}$\\pm${o['acc_std']:.1f}"
            auc_str = f"{o['auc_mean']:.3f}$\\pm${o['auc_std']:.3f}"
            if bold:
                bacc_str = r"\textbf{" + bacc_str + "}"
                acc_str = r"\textbf{" + acc_str + "}"
                auc_str = r"\textbf{" + auc_str + "}"
            lines.append(
                f"  {display} & 2025 & sMRI (3D) & 2,401 & {eval_str} & "
                f"{acc_str} & {bacc_str} & {auc_str} \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SOTA comparison table")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to aggregated.json from aggregate_results.py")
    parser.add_argument("--output", type=str, default="chapter1_foundation/sota_table",
                        help="Output prefix (writes .md and .tex)")
    args = parser.parse_args()

    own = None
    if args.results:
        own = load_own_results(Path(args.results))
        print(f"Loaded own results: {list(own.keys())}")

    md = generate_markdown(own)
    tex = generate_latex(own)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    md_path = out.with_suffix(".md")
    tex_path = out.with_suffix(".tex")

    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table → {md_path}")

    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table → {tex_path}")

    print("\n--- Markdown Preview ---\n")
    print(md)
    print("\n--- LaTeX Preview ---\n")
    print(tex)


if __name__ == "__main__":
    main()
