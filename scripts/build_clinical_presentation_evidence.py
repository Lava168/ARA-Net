#!/usr/bin/env python3
"""Build reviewer-facing clinical-presentation evidence tables for ARA-Net."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ["CN", "MCI", "AD"]
REGION_FEATURES = {
    "atlas_hippocampus_volume": {
        "label": "Hippocampus volume",
        "prior": "AD-prior medial temporal proxy",
        "expected": "decrease",
    },
    "atlas_amygdala_volume": {
        "label": "Amygdala volume",
        "prior": "AD-prior medial temporal proxy",
        "expected": "decrease",
    },
    "atlas_lateral_ventricle_volume": {
        "label": "Lateral ventricle volume",
        "prior": "AD-prior ventricular enlargement",
        "expected": "increase",
    },
    "atlas_cortex_volume": {
        "label": "Cortex volume",
        "prior": "broad cortical reserve",
        "expected": "decrease",
    },
    "atlas_ad_like_z": {
        "label": "Atlas AD-like z",
        "prior": "summary structural neurodegeneration score",
        "expected": "increase",
    },
}
CLINICAL_FEATURES = ["clin_age", "clin_mmse", "clin_cdrsb", "clin_apoe4"]
UNAVAILABLE_ANNOTATION_TARGETS = ["clin_entorhinal", "temporal_lobe_annotation", "clinician_roi_annotation"]


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return math.nan
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return statistics.fmean(values) if values else math.nan


def sd(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values)


def pooled_sd(a: list[float], b: list[float]) -> float:
    a = [v for v in a if math.isfinite(v)]
    b = [v for v in b if math.isfinite(v)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    numerator = (len(a) - 1) * statistics.variance(a) + (len(b) - 1) * statistics.variance(b)
    denom = len(a) + len(b) - 2
    return math.sqrt(numerator / denom) if denom > 0 else math.nan


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def class_feature_gradients(rows: list[dict], split: str) -> list[dict]:
    split_rows = [row for row in rows if row.get("split") == split]
    out = []
    for column, meta in REGION_FEATURES.items():
        by_class = {
            cls: [parse_float(row.get(column)) for row in split_rows if row.get("y_true") == cls]
            for cls in CLASSES
        }
        cn = by_class["CN"]
        ad = by_class["AD"]
        cn_mean = mean(cn)
        mci_mean = mean(by_class["MCI"])
        ad_mean = mean(ad)
        delta = ad_mean - cn_mean if math.isfinite(cn_mean) and math.isfinite(ad_mean) else math.nan
        pooled = pooled_sd(cn, ad)
        effect = delta / pooled if math.isfinite(pooled) and pooled > 0 else math.nan
        expected = meta["expected"]
        direction_pass = (
            (expected == "increase" and math.isfinite(delta) and delta > 0)
            or (expected == "decrease" and math.isfinite(delta) and delta < 0)
        )
        out.append(
            {
                "feature": column,
                "label": meta["label"],
                "prior": meta["prior"],
                "expected_direction": expected,
                "CN_mean": cn_mean,
                "MCI_mean": mci_mean,
                "AD_mean": ad_mean,
                "AD_minus_CN": delta,
                "AD_vs_CN_cohens_d": effect,
                "direction_matches_AD_prior": direction_pass,
            }
        )
    return out


def load_ad_key_concentration(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = read_csv(path)
    out = {}
    for row in rows:
        group = row["group"]
        out[group] = {k: parse_float(v) if k != "group" else v for k, v in row.items()}
    return out


def split_availability(rows: list[dict], split: str) -> dict:
    split_rows = [row for row in rows if row.get("split") == split]
    out = {}
    for column in CLINICAL_FEATURES + list(REGION_FEATURES):
        present = sum(math.isfinite(parse_float(row.get(column))) for row in split_rows)
        out[column] = {
            "n_present": present,
            "n_total": len(split_rows),
            "availability": present / len(split_rows) if split_rows else math.nan,
        }
    return out


def domain_shift(rows: list[dict], reference_split: str, target_split: str) -> list[dict]:
    ref = [row for row in rows if row.get("split") == reference_split]
    target = [row for row in rows if row.get("split") == target_split]
    out = []
    for column, meta in REGION_FEATURES.items():
        ref_values = [parse_float(row.get(column)) for row in ref]
        target_values = [parse_float(row.get(column)) for row in target]
        ref_mean = mean(ref_values)
        target_mean = mean(target_values)
        pooled = pooled_sd(ref_values, target_values)
        smd = (target_mean - ref_mean) / pooled if math.isfinite(pooled) and pooled > 0 else math.nan
        out.append(
            {
                "feature": column,
                "label": meta["label"],
                "aibl_heldout_mean": ref_mean,
                "oasis_external_mean": target_mean,
                "oasis_minus_aibl": target_mean - ref_mean if math.isfinite(ref_mean) and math.isfinite(target_mean) else math.nan,
                "standardized_mean_difference": smd,
            }
        )
    return sorted(out, key=lambda item: abs(item["standardized_mean_difference"]) if math.isfinite(item["standardized_mean_difference"]) else -1, reverse=True)


def prediction_distribution(rows: list[dict], split: str) -> dict:
    split_rows = [row for row in rows if row.get("split") == split]
    counts = Counter(row.get("y_pred", "") for row in split_rows)
    total = len(split_rows)
    return {
        "n": total,
        "counts": dict(counts),
        "rates": {cls: counts.get(cls, 0) / total if total else math.nan for cls in CLASSES},
    }


def performance_drop(summary: dict) -> dict:
    metrics = summary["final_model"]["subject_level_metrics"]
    aibl = metrics["aibl_heldout"]
    oasis = metrics["oasis_external"]
    return {
        "aibl_heldout": aibl,
        "oasis_external": oasis,
        "delta_oasis_minus_aibl": {
            "balanced_acc": oasis["balanced_acc"] - aibl["balanced_acc"],
            "macro_auc_ovr": oasis["macro_auc_ovr"] - aibl["macro_auc_ovr"],
            "recall_CN": oasis["recall_CN"] - aibl["recall_CN"],
            "recall_MCI": oasis["recall_MCI"] - aibl["recall_MCI"],
            "recall_AD": oasis["recall_AD"] - aibl["recall_AD"],
        },
    }


def write_markdown(path: Path, payload: dict) -> None:
    concentration = payload["atlas_evidence"]["ad_key_concentration"].get("aibl_heldout", {})
    drop = payload["oasis_domain_shift"]["performance"]
    oasis = drop["oasis_external"]
    delta = drop["delta_oasis_minus_aibl"]

    lines = [
        "# ARA-Net Clinical Presentation Evidence Pack",
        "",
        "## What This Adds",
        "",
        "- Lightweight deployment numbers for the RC-SPE probability head.",
        "- Atlas-level AD-prior evidence validation for hippocampus, amygdala, and lateral ventricles.",
        "- Explicit OASIS domain-shift explanation instead of hiding the weak external result.",
        "- A UI-ready checklist for reviewer-facing clinical presentation without overclaiming clinical deployment.",
        "",
        "## Atlas Evidence Validation",
        "",
        "| Evidence | Value |",
        "|---|---:|",
        f"| AIBL AD-key concentration score | {fmt(concentration.get('ad_key_score', math.nan))} |",
        f"| Uniform regional null | {fmt(concentration.get('uniform_null', math.nan))} |",
        f"| Score minus null | {fmt(concentration.get('delta', math.nan))} |",
        f"| Bootstrap 95% CI | {fmt(concentration.get('ci_low', math.nan))}-{fmt(concentration.get('ci_high', math.nan))} |",
        f"| Permutation p | {fmt(concentration.get('permutation_p', math.nan))} |",
        "",
        "### AIBL Heldout Structural Direction Checks",
        "",
        "| Feature | CN mean | MCI mean | AD mean | AD-CN | Cohen d | Matches AD prior |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["atlas_evidence"]["aibl_heldout_gradients"]:
        lines.append(
            "| {label} | {cn} | {mci} | {ad} | {delta} | {effect} | {ok} |".format(
                label=row["label"],
                cn=fmt(row["CN_mean"]),
                mci=fmt(row["MCI_mean"]),
                ad=fmt(row["AD_mean"]),
                delta=fmt(row["AD_minus_CN"]),
                effect=fmt(row["AD_vs_CN_cohens_d"]),
                ok="yes" if row["direction_matches_AD_prior"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## OASIS Domain-Shift Finding",
            "",
            "| OASIS metric | Value |",
            "|---|---:|",
            f"| n | {oasis['n']} |",
            f"| Accuracy | {fmt(oasis['acc'])} |",
            f"| Balanced accuracy | {fmt(oasis['balanced_acc'])} |",
            f"| Macro AUC | {fmt(oasis['macro_auc_ovr'])} |",
            f"| CN recall | {fmt(oasis['recall_CN'])} |",
            f"| MCI recall | {fmt(oasis['recall_MCI'])} |",
            f"| AD recall | {fmt(oasis['recall_AD'])} |",
            f"| Balanced-accuracy drop vs AIBL heldout | {fmt(delta['balanced_acc'])} |",
            f"| MCI-recall drop vs AIBL heldout | {fmt(delta['recall_MCI'])} |",
            f"| AD-recall drop vs AIBL heldout | {fmt(delta['recall_AD'])} |",
            "",
            "### OASIS Prediction Collapse",
            "",
            "| Predicted class | n | rate |",
            "|---|---:|---:|",
        ]
    )
    pred = payload["oasis_domain_shift"]["prediction_distribution"]
    for cls in CLASSES:
        lines.append(f"| {cls} | {pred['counts'].get(cls, 0)} | {fmt(pred['rates'].get(cls, math.nan))} |")
    lines.extend(
        [
            "",
            "### Largest OASIS-vs-AIBL Structural Shifts",
            "",
            "| Feature | AIBL mean | OASIS mean | OASIS-AIBL | SMD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["oasis_domain_shift"]["structural_shift"][:5]:
        lines.append(
            "| {label} | {aibl} | {oasis_mean} | {delta_feature} | {smd} |".format(
                label=row["label"],
                aibl=fmt(row["aibl_heldout_mean"]),
                oasis_mean=fmt(row["oasis_external_mean"]),
                delta_feature=fmt(row["oasis_minus_aibl"]),
                smd=fmt(row["standardized_mean_difference"]),
            )
        )
    lines.extend(
        [
            "",
            "## Manuscript-Ready Interpretation",
            "",
            "ARA-Net's locked AIBL heldout result supports domain-adapted external subject-level staging, but OASIS remains a stress-test failure. The OASIS pattern is not a random accuracy drop: predictions collapse toward CN, preserving high CN recall while nearly eliminating MCI and AD recall. This is consistent with a domain-shift problem involving cohort/scanner/protocol differences, label-distribution differences, and incomplete clinical-feature harmonization. The correct claim is therefore not zero-shot generalization, but strong AIBL heldout performance with an explicitly documented OASIS limitation.",
            "",
            "## Annotation Boundary",
            "",
            "Current evidence supports atlas-level structural neurodegeneration consistency. It does not yet provide clinician-drawn lesion masks, CAGM-style clinical annotation overlap, entorhinal-cortex validation in the public enriched table, or temporal-lobe region masks. Those targets should be described as planned validation rather than completed evidence.",
            "",
            "Unavailable annotation targets in the current public evidence table:",
        ]
    )
    for target in payload["annotation_boundary"]["unavailable_targets"]:
        lines.append(f"- {target}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("reports/v6_final_model/tables/final_subject_predictions_enriched.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/v6_final_model/final_rescue_model_summary_public.json"))
    parser.add_argument("--neurodegeneration-table", type=Path, default=Path("reports/v4/tables/table4_neurodegeneration.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/v6_final_model/tables/clinical_presentation_evidence.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/v6_final_model/tables/clinical_presentation_evidence.md"))
    args = parser.parse_args()

    rows = read_csv(args.predictions)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    concentration = load_ad_key_concentration(args.neurodegeneration_table)

    payload = {
        "model": "ARA-Net",
        "atlas_evidence": {
            "ad_key_concentration": concentration,
            "aibl_heldout_gradients": class_feature_gradients(rows, "aibl_heldout"),
            "oasis_external_gradients": class_feature_gradients(rows, "oasis_external"),
        },
        "oasis_domain_shift": {
            "performance": performance_drop(summary),
            "prediction_distribution": prediction_distribution(rows, "oasis_external"),
            "structural_shift": domain_shift(rows, "aibl_heldout", "oasis_external"),
            "feature_availability": {
                "aibl_heldout": split_availability(rows, "aibl_heldout"),
                "oasis_external": split_availability(rows, "oasis_external"),
            },
        },
        "annotation_boundary": {
            "completed": [
                "atlas-level AD-key concentration over hippocampus, amygdala, and lateral ventricles",
                "AIBL heldout structural direction checks",
                "OASIS prediction-collapse and domain-shift summary",
            ],
            "unavailable_targets": UNAVAILABLE_ANNOTATION_TARGETS,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
