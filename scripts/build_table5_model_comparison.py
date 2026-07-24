#!/usr/bin/env python3
"""Build the manuscript-ready Table 5 model-comparison summary.

The public repository contains aggregate server-side results, but not the
private row-level MRI/clinical feature cache. This script consolidates the
available aggregate evidence into a single table for manuscript drafting.
"""
from __future__ import annotations

import csv
from pathlib import Path


OUT_CSV = Path("reports/v6_final_model/tables/table5_model_comparison.csv")
OUT_MD = Path("reports/v6_final_model/tables/table5_model_comparison.md")
CONFIG = Path("deployment/final_ensemble_config.json")


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def main() -> None:
    config_size = CONFIG.stat().st_size if CONFIG.exists() else None
    rcspe_size = f"10 scalar params / {config_size:,} bytes JSON" if config_size else "10 scalar params"

    rows = [
        {
            "Model / strategy": "Old v3 ensemble",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.399),
            "AIBL macro AUC": pct(0.597),
            "AD-vs-CN AUC": "N/A",
            "CN/MCI/AD recall": "N/A",
            "IXI CN retention": pct(0.439),
            "OASIS BAcc": "N/A",
            "Model size / parameters": "N/A",
            "Role": "failed external baseline",
        },
        {
            "Model / strategy": "Atlas-only HGB",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.479),
            "AIBL macro AUC": pct(0.732),
            "AD-vs-CN AUC": pct(0.884),
            "CN/MCI/AD recall": "94.4%/15.1%/34.1%",
            "IXI CN retention": pct(0.983),
            "OASIS BAcc": "N/A",
            "Model size / parameters": "N/A",
            "Role": "MRI atlas-only baseline",
        },
        {
            "Model / strategy": "Cascade RF-logreg",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.391),
            "AIBL macro AUC": pct(0.756),
            "AD-vs-CN AUC": pct(0.886),
            "CN/MCI/AD recall": "95.4%/0.0%/22.0%",
            "IXI CN retention": pct(1.000),
            "OASIS BAcc": "N/A",
            "Model size / parameters": "N/A",
            "Role": "two-stage baseline",
        },
        {
            "Model / strategy": "ADNI-only hybrid RF",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.406),
            "AIBL macro AUC": pct(0.753),
            "AD-vs-CN AUC": pct(0.870),
            "CN/MCI/AD recall": "61.4%/60.4%/0.0%",
            "IXI CN retention": "N/A",
            "OASIS BAcc": "N/A",
            "Model size / parameters": "N/A",
            "Role": "ADNI-only zero-shot check",
        },
        {
            "Model / strategy": "v4 atlas+clinical HGB",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.741),
            "AIBL macro AUC": pct(0.942),
            "AD-vs-CN AUC": pct(0.990),
            "CN/MCI/AD recall": "96.4%/52.8%/73.2%",
            "IXI CN retention": pct(0.998),
            "OASIS BAcc": pct(0.322),
            "Model size / parameters": "N/A",
            "Role": "earlier atlas-guided multimodal model",
        },
        {
            "Model / strategy": "Biomarker-enhanced HGB",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.703),
            "AIBL macro AUC": pct(0.942),
            "AD-vs-CN AUC": "N/A",
            "CN/MCI/AD recall": "95.7%/39.6%/75.6%",
            "IXI CN retention": pct(0.997),
            "OASIS BAcc": pct(0.310),
            "Model size / parameters": "N/A",
            "Role": "biomarker sensitivity model",
        },
        {
            "Model / strategy": "Clinical-only RF comparator",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.835),
            "AIBL macro AUC": pct(0.957),
            "AD-vs-CN AUC": pct(0.997),
            "CN/MCI/AD recall": "97.0%/75.5%/78.0%",
            "IXI CN retention": pct(1.000),
            "OASIS BAcc": pct(0.333),
            "Model size / parameters": "N/A",
            "Role": "clinical comparator / upper bound",
        },
        {
            "Model / strategy": "Best single base model",
            "Evaluation unit": "subject",
            "AIBL BAcc": pct(0.756),
            "AIBL macro AUC": pct(0.945),
            "AD-vs-CN AUC": pct(0.994),
            "CN/MCI/AD recall": "95.5%/57.1%/74.1%",
            "IXI CN retention": pct(0.997),
            "OASIS BAcc": pct(0.310),
            "Model size / parameters": "N/A",
            "Role": "best individual stream",
        },
        {
            "Model / strategy": "Full RC-SPE",
            "Evaluation unit": "scan",
            "AIBL BAcc": pct(0.820),
            "AIBL macro AUC": pct(0.939),
            "AD-vs-CN AUC": pct(0.998),
            "CN/MCI/AD recall": "96.4%/64.2%/85.4%",
            "IXI CN retention": pct(1.000),
            "OASIS BAcc": pct(0.334),
            "Model size / parameters": rcspe_size,
            "Role": "scan-level reference",
        },
        {
            "Model / strategy": "ARA-Net / RC-SPE",
            "Evaluation unit": "subject",
            "AIBL BAcc": pct(0.833),
            "AIBL macro AUC": pct(0.937),
            "AD-vs-CN AUC": pct(1.000),
            "CN/MCI/AD recall": "96.1%/68.6%/85.2%",
            "IXI CN retention": pct(1.000),
            "OASIS BAcc": pct(0.334),
            "Model size / parameters": rcspe_size,
            "Role": "locked primary result",
        },
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Table 5 | ARA-Net and Baseline Model Comparison Across Multi-Cohort AD Staging",
        "",
        "| Model / strategy | Evaluation unit | AIBL BAcc | AIBL macro AUC | AD-vs-CN AUC | CN/MCI/AD recall | IXI CN retention | OASIS BAcc | Model size / parameters |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Model / strategy} | {Evaluation unit} | {AIBL BAcc} | {AIBL macro AUC} | {AD-vs-CN AUC} | {CN/MCI/AD recall} | {IXI CN retention} | {OASIS BAcc} | {Model size / parameters} |".format(**row)
        )
    lines.extend([
        "",
        "Note: AIBL is the primary locked external test cohort; IXI is used as a healthy-control specificity check; OASIS is reported as a stress-test limitation, not as a successful external validation. Model checkpoint sizes were not computed for baselines other than ARA-Net / RC-SPE and are therefore reported as N/A."
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {OUT_CSV}")
    print(f"[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
