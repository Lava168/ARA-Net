#!/usr/bin/env python3
"""Generate the v6 core reviewer-evidence matrix.

The matrix consolidates the three formerly fatal reviewer issues:

1. external CN/MCI/AD classification generalization,
2. the invalid attention-only CAS claim, and
3. non-significant direct Braak validation.

It deliberately reads already generated aggregate/public artifacts rather than
row-level prediction files, so the output is safe for the public repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_SUMMARY = Path("reports/v6_final_model/final_rescue_model_summary_public.json")
DEFAULT_NEURO = Path("reports/v4/tables/table4_neurodegeneration.csv")
DEFAULT_CLASSIFICATION = Path("reports/v4/tables/table2_classification.csv")
DEFAULT_CONFIG = Path("deployment/final_ensemble_config.json")
DEFAULT_OUTPUT = Path("reports/v6_final_model/core_reviewer_evidence_matrix.md")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_ci(metric: Mapping[str, object] | None, digits: int = 3) -> str:
    if not metric:
        return "NA"
    return f"{fmt(metric.get('ci_low'), digits)}-{fmt(metric.get('ci_high'), digits)}"


def fmt_recall(metrics: Mapping[str, object]) -> str:
    return "/".join(fmt(metrics.get(f"recall_{label}")) for label in ["CN", "MCI", "AD"])


def by_model_eval(rows: Iterable[dict], model: str, evaluation: str) -> dict:
    for row in rows:
        if row.get("model") == model and row.get("evaluation") == evaluation:
            return row
    raise KeyError(f"Missing row for model={model!r}, evaluation={evaluation!r}")


def by_group(rows: Iterable[dict], group: str) -> dict:
    for row in rows:
        if row.get("group") == group:
            return row
    raise KeyError(f"Missing neurodegeneration row for group={group!r}")


def line_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(summary: dict, neuro_rows: list[dict], class_rows: list[dict], config: dict, sources: list[Path]) -> str:
    subject_metrics = summary["final_model"]["subject_level_metrics"]
    bootstrap = summary["final_model"]["subject_level_bootstrap"]
    aibl = subject_metrics["aibl_heldout"]
    aibl_boot = bootstrap["aibl_heldout"]
    ixi = subject_metrics["ixi_external"]
    oasis = subject_metrics["oasis_external"]
    internal = subject_metrics["internal_test"]
    old_aibl = by_model_eval(class_rows, "Old v3 ensemble", "AIBL external")
    old_ixi = by_model_eval(class_rows, "Old v3 ensemble", "IXI healthy")
    v4_main = by_model_eval(class_rows, "Recommended atlas+clinical HGB", "AIBL heldout")
    neuro_aibl = by_group(neuro_rows, "aibl_heldout")
    neuro_all = by_group(neuro_rows, "all_labeled_ad")
    neuro_adni = by_group(neuro_rows, "adni_val_internal_test")
    cm = aibl["confusion_matrix"]
    internal_cm = internal["confusion_matrix"]

    source_lines = [
        f"- `{path}` sha256:{sha256_prefix(path)}"
        for path in sources
    ]

    evidence_rows = [
        [
            "External CN/MCI/AD classification",
            "Substantially addressed",
            (
                f"Old v3 AIBL BAcc {fmt(old_aibl['balanced_acc'])} -> "
                f"v4 atlas+clinical BAcc {fmt(v4_main['balanced_acc'])} -> "
                f"final subject AIBL BAcc {fmt(aibl['balanced_acc'])} "
                f"(95% CI {fmt_ci(aibl_boot['balanced_acc'])}); "
                f"Acc {fmt(aibl['acc'])}, macro AUC {fmt(aibl['macro_auc_ovr'])}, "
                f"AD-vs-CN AUC {fmt(aibl['ad_vs_cn_auc'])}, recall CN/MCI/AD {fmt_recall(aibl)}."
            ),
            "Lead with locked AIBL subject-level result.",
            "Domain-adapted external heldout, not pure zero-shot transfer.",
        ],
        [
            "Healthy external specificity",
            "Addressed for negative-control use",
            (
                f"Old v3 IXI CN retention {fmt(old_ixi['cn_retention'])}; "
                f"final IXI CN retention {fmt(ixi['cn_retention_rate'])} "
                f"with {int(ixi['n'])} healthy subjects and false-impairment rate {fmt(ixi['false_impairment_rate'])}."
            ),
            "Report IXI as healthy negative-control specificity.",
            "IXI does not provide AD/MCI staging labels.",
        ],
        [
            "OASIS external transfer",
            "Not solved; preserved as limitation",
            (
                f"OASIS subject BAcc {fmt(oasis['balanced_acc'])}, macro AUC {fmt(oasis['macro_auc_ovr'])}, "
                f"AD-vs-CN AUC {fmt(oasis['ad_vs_cn_auc'])}, recall CN/MCI/AD {fmt_recall(oasis)}."
            ),
            "Keep OASIS as stress-test limitation.",
            "Do not claim OASIS validation success.",
        ],
        [
            "CAS validity",
            "Resolved by removing/replacing invalid attention-only CAS",
            (
                f"AIBL AD-key atlas-volume consistency score {fmt(neuro_aibl['ad_key_score'])} "
                f"vs uniform null {fmt(neuro_aibl['uniform_null'])}, delta {fmt(neuro_aibl['delta'])}, "
                f"95% CI {fmt(neuro_aibl['ci_low'])}-{fmt(neuro_aibl['ci_high'])}, "
                f"permutation p={fmt(neuro_aibl['permutation_p'], 4)}."
            ),
            "Replace CAS biomarker wording with atlas structural neurodegeneration consistency.",
            "This supports a structural MRI proxy, not attention maps as biomarkers.",
        ],
        [
            "Braak or substitute biology",
            "Addressed as Braak-alternative proxy, not direct Braak proof",
            (
                f"All labeled AD-key consistency score {fmt(neuro_all['ad_key_score'])} "
                f"vs null {fmt(neuro_all['uniform_null'])}, p={fmt(neuro_all['permutation_p'], 4)}; "
                f"ADNI-only internal check remains non-significant "
                f"(score {fmt(neuro_adni['ad_key_score'])}, p={fmt(neuro_adni['permutation_p'], 4)})."
            ),
            "Use 'MRI neurodegeneration proxy' and remove direct Braak-stage claims.",
            "No neuropathological Braak-stage validation is available.",
        ],
        [
            "MCI/AD error risk",
            "Improved and explicitly quantified",
            (
                f"AIBL confusion rows CN/MCI/AD: {cm[0]}, {cm[1]}, {cm[2]}; "
                f"AD-to-CN errors {cm[2][0]}; MCI recall 95% CI {fmt_ci(aibl_boot['recall_MCI'])}; "
                f"AD recall 95% CI {fmt_ci(aibl_boot['recall_AD'])}."
            ),
            "Report confusion matrix and error-profile figure.",
            "MCI remains the main residual weakness.",
        ],
        [
            "Internal calibration risk",
            "Open limitation",
            (
                f"Internal subject BAcc {fmt(internal['balanced_acc'])}, recall CN/MCI/AD {fmt_recall(internal)}, "
                f"confusion rows CN/MCI/AD: {internal_cm[0]}, {internal_cm[1]}, {internal_cm[2]}."
            ),
            "Discuss internal CN-to-MCI shift as calibration limitation.",
            "Do not overstate universal readiness.",
        ],
    ]

    metrics_rows = [
        ["AIBL heldout subject", str(int(aibl["n"])), fmt(aibl["acc"]), fmt(aibl["balanced_acc"]), fmt(aibl["macro_auc_ovr"]), fmt(aibl["ad_vs_cn_auc"]), fmt_recall(aibl)],
        ["IXI healthy subject", str(int(ixi["n"])), fmt(ixi["acc"]), fmt(ixi["balanced_acc"]), "NA", f"CN retention {fmt(ixi['cn_retention_rate'])}", fmt_recall(ixi)],
        ["OASIS stress subject", str(int(oasis["n"])), fmt(oasis["acc"]), fmt(oasis["balanced_acc"]), fmt(oasis["macro_auc_ovr"]), fmt(oasis["ad_vs_cn_auc"]), fmt_recall(oasis)],
        ["Internal subject", str(int(internal["n"])), fmt(internal["acc"]), fmt(internal["balanced_acc"]), fmt(internal["macro_auc_ovr"]), fmt(internal["ad_vs_cn_auc"]), fmt_recall(internal)],
    ]

    lines = [
        "# Core Reviewer Evidence Matrix",
        "",
        "## Purpose",
        "",
        "This generated matrix consolidates the evidence that the revised ARA-Net work is a substantive rebuild rather than a small v3 patch. It focuses on the three critical issues: external classification, invalid CAS, and non-significant Braak validation.",
        "",
        "## Generated From",
        "",
        *source_lines,
        "",
        "## Locked Model",
        "",
        f"- Model: {config['name']} ({config['version']})",
        f"- Classes: {', '.join(config['classes'])}",
        f"- Primary endpoint: AIBL locked heldout subject-level CN/MCI/AD staging.",
        f"- Clinical-use boundary: {config['clinical_use_notice']}",
        "",
        "## Requirement-Level Evidence",
        "",
        *line_table(
            ["reviewer issue", "current status", "quantitative evidence", "manuscript use", "claim boundary"],
            evidence_rows,
        ),
        "",
        "## Primary Metrics Snapshot",
        "",
        *line_table(
            ["cohort", "n", "Acc", "BAcc", "macro AUC", "AD-vs-CN / specificity", "recall CN/MCI/AD"],
            metrics_rows,
        ),
        "",
        "## Reviewer-Safe Claim",
        "",
        "The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance and IXI healthy specificity. The old attention-only CAS and direct Braak claims should be removed; the biological claim should be limited to an atlas-region MRI neurodegeneration proxy. OASIS transfer and internal calibration remain explicit limitations.",
        "",
        "## Manuscript Actions",
        "",
        "- Lead the Results with the final AIBL subject-level endpoint, bootstrap intervals, and confusion matrix.",
        "- State that IXI is a healthy negative-control specificity analysis.",
        "- Keep OASIS as an honest stress-test limitation.",
        "- Replace CAS/Braak language with atlas structural neurodegeneration consistency language.",
        "- Include MCI/AD error analysis and avoid clinical-deployment claims.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--neurodegeneration", type=Path, default=DEFAULT_NEURO)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sources = [args.summary, args.neurodegeneration, args.classification, args.config]
    summary = load_json(args.summary)
    neuro_rows = load_csv(args.neurodegeneration)
    class_rows = load_csv(args.classification)
    config = load_json(args.config)
    report = build_report(summary, neuro_rows, class_rows, config, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
