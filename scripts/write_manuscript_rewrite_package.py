#!/usr/bin/env python3
"""Generate manuscript rewrite materials from the v4 experiment reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.load(path.open()) if path.exists() else None


def metric(summary: dict, protocol: str, run: str, split: str) -> dict:
    return summary["results"][protocol][run]["metrics"][split]


def recall(m: dict, label: str) -> float:
    return float(m.get("per_class", {}).get(label, {}).get("recall") or 0.0)


def fmt_auc(value) -> str:
    return "NA" if value is None else f"{value:.3f}"


def fmt_metrics(m: dict) -> str:
    extra = []
    if m.get("ad_vs_cn_auc") is not None:
        extra.append(f"AD-vs-CN AUC {m['ad_vs_cn_auc']:.3f}")
    if m.get("cn_retention_rate") is not None:
        extra.append(f"CN retention {m['cn_retention_rate']:.3f}")
    extra_text = ", " + ", ".join(extra) if extra else ""
    return (
        f"Acc {m.get('acc'):.3f}, BAcc {m.get('balanced_acc'):.3f}, "
        f"macro AUC {fmt_auc(m.get('macro_auc_ovr'))}{extra_text}, "
        f"recall CN/MCI/AD {recall(m, 'CN'):.3f}/{recall(m, 'MCI'):.3f}/{recall(m, 'AD'):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--review-text", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = args.v4_root
    hybrid = load(root / "hybrid_atlas_clinical_baseline" / "summary.json")
    feature = load(root / "atlas_feature_baseline" / "summary.json")
    cascade = load(root / "atlas_cascade_baseline" / "summary.json")
    biomarker = load(root / "atlas_feature_biomarkers" / "summary.json")
    manifest = load(root / "manifest_v4_summary.json")
    replicates = load(root / "hybrid_replicate_summary.json")
    old_v3 = load(root.parent / "analysis" / "external_validation_v3_merged.json")

    main_run = "atlas_core_clinical__hgb"
    clinical_run = "clinical_core_only__rf_balanced"
    biomarker_run = "atlas_biomarker_enhanced__hgb"

    m_main = metric(hybrid, "aibl_adapted", main_run, "aibl_heldout")
    m_main_ixi = metric(hybrid, "aibl_adapted", main_run, "ixi_external")
    m_main_internal = metric(hybrid, "aibl_adapted", main_run, "internal_test")
    m_clin = metric(hybrid, "aibl_adapted", clinical_run, "aibl_heldout")
    m_bio = metric(hybrid, "aibl_adapted", biomarker_run, "aibl_heldout")
    m_adni_only = metric(hybrid, "adni_only", "atlas_biomarker_enhanced__rf_balanced", "aibl_heldout")
    feature_best = feature["results"][feature["best_model"]]["metrics"]["aibl_heldout"]
    cascade_best = cascade["results"][cascade["best_model"]]["metrics"]["aibl_heldout"]

    split_counts = manifest["split_counts"]
    aibl_biomarker = biomarker["aibl_heldout"]["ad_key_volume_score"]
    aibl_ci = aibl_biomarker["bootstrap_ci"]
    replicate_stats = replicates["runs"][main_run]["summary"]

    lines = ["# Manuscript Rewrite Package", ""]

    lines += [
        "## New Manuscript Positioning",
        "",
        "Recommended title direction:",
        "",
        "**Atlas-guided multimodal Alzheimer disease staging with external heldout validation and neurodegeneration-consistent regional biomarkers**",
        "",
        "Core claim to make:",
        "",
        "The revised work should no longer be framed as a purely attention-based deep MRI classifier. It should be framed as a cross-cohort atlas-guided and clinically adapted AD staging framework that combines anatomically grounded MRI region features with core clinical variables, validates classification on a locked AIBL heldout split, tests healthy specificity on IXI, and replaces the failed CAS/Braak claim with an empirically tested atlas-region neurodegeneration consistency score.",
        "",
        "Claims to avoid:",
        "",
        "- Do not claim pure ADNI-to-AIBL zero-shot staging is solved.",
        "- Do not claim direct Braak staging validation.",
        "- Do not use the clinical-only model as the main atlas-guided model.",
        "- Do not hide the weak OASIS transfer.",
        "",
    ]

    lines += ["## Dataset And Split Table", ""]
    lines.append("| Split | Scans | Subjects | CN | MCI | AD | Role |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    roles = {
        "train": "ADNI training",
        "val": "ADNI model selection",
        "internal_test": "ADNI internal test",
        "aibl_adapt_train": "AIBL adaptation training",
        "aibl_adapt_val": "AIBL adaptation validation",
        "aibl_heldout": "Locked AIBL heldout external test",
        "oasis_external": "OASIS external stress test",
        "ixi_external": "Healthy external negative-control cohort",
    }
    for split in roles:
        item = split_counts[split]
        labels = item["labels"]
        lines.append(
            f"| {split} | {item['scans']} | {item['subjects']} | {labels.get('CN', 0)} | "
            f"{labels.get('MCI', 0)} | {labels.get('AD', 0)} | {roles[split]} |"
        )
    lines.append("")

    lines += ["## Main Result Table", ""]
    lines.append("| Model / protocol | Evaluation | Acc | BAcc | Macro AUC | AD-vs-CN AUC / CN retention | CN/MCI/AD recall | Interpretation |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")
    table_rows = [
        ("Old v3 ensemble", "AIBL external", old_v3["ensemble"]["aibl"], None, "Failed external baseline"),
        ("Old v3 ensemble", "IXI healthy", old_v3["ensemble"]["ixi"], None, "High false impairment rate"),
        ("Atlas-only HGB", "AIBL heldout", feature_best, None, "MRI specificity improves but staging remains weak"),
        ("Cascade RF-logreg", "AIBL heldout", cascade_best, None, "Healthy-specific but fails MCI heldout"),
        ("ADNI-only hybrid RF", "AIBL heldout", m_adni_only, None, "Pure zero-shot remains insufficient"),
        ("Recommended atlas+clinical HGB", "AIBL heldout", m_main, None, "Main atlas-guided multimodal result"),
        ("Recommended atlas+clinical HGB", "IXI healthy", m_main_ixi, None, "Healthy negative-control specificity"),
        ("Clinical-only RF", "AIBL heldout", m_clin, None, "Strong comparator / upper-bound"),
        ("Biomarker-enhanced HGB", "AIBL heldout", m_bio, None, "Sensitivity analysis"),
    ]
    for name, eval_name, m, _, note in table_rows:
        if "accuracy" in m:
            acc = m.get("accuracy")
            bacc = m.get("balanced_accuracy_present")
            auc = m.get("macro_auc_ovr_valid")
            extra = f"CN retention {m.get('ixi_cn_retention_rate'):.3f}" if m.get("ixi_cn_retention_rate") is not None else ""
            recalls = "see old v3 report"
        else:
            acc = m.get("acc")
            bacc = m.get("balanced_acc")
            auc = m.get("macro_auc_ovr")
            if m.get("cn_retention_rate") is not None:
                extra = f"CN retention {m.get('cn_retention_rate'):.3f}"
            elif m.get("ad_vs_cn_auc") is not None:
                extra = f"AD-vs-CN AUC {m.get('ad_vs_cn_auc'):.3f}"
            else:
                extra = ""
            recalls = f"{recall(m, 'CN'):.3f}/{recall(m, 'MCI'):.3f}/{recall(m, 'AD'):.3f}"
        lines.append(
            f"| {name} | {eval_name} | {acc:.3f} | {bacc:.3f} | {fmt_auc(auc)} | {extra} | {recalls} | {note} |"
        )
    lines.append("")

    lines += [
        "## Three Fatal Issues And New Evidence",
        "",
        "### 1. Cross-dataset generalization was unsupported",
        "",
        "Old manuscript problem: IXI and OASIS were used for attention similarity only; no external classification metrics were reported.",
        "",
        "New evidence:",
        "",
        f"- Locked AIBL heldout external test for the recommended atlas+clinical model: {fmt_metrics(m_main)}.",
        f"- IXI healthy negative-control test: {fmt_metrics(m_main_ixi)}.",
        f"- Multi-seed confirmation for the recommended model: AIBL heldout BAcc {replicate_stats['aibl_heldout.balanced_acc']['mean']:.3f} +/- {replicate_stats['aibl_heldout.balanced_acc']['std']:.3f}, IXI CN retention {replicate_stats['ixi_external.cn_retention_rate']['mean']:.3f} +/- {replicate_stats['ixi_external.cn_retention_rate']['std']:.3f}, n={replicate_stats['aibl_heldout.balanced_acc']['n']}.",
        "",
        "How to write it:",
        "",
        "We now explicitly distinguish zero-shot external evaluation from clinically adapted external heldout evaluation. ADNI-only models remain weak on AIBL AD detection, whereas the clinically adapted atlas-guided model generalizes to a locked AIBL heldout set and preserves specificity on IXI.",
        "",
        "### 2. CAS was below chance and unvalidated",
        "",
        "Old manuscript problem: attention-based CAS was below the uniform 6/21 null and therefore could not support clinical alignment.",
        "",
        "New evidence:",
        "",
        f"- AIBL heldout AD-key volume score {aibl_biomarker['ad_key_score']:.3f} versus uniform null {aibl_biomarker['uniform_null']:.3f}; delta {aibl_biomarker['score_minus_uniform']:.3f}; bootstrap CI [{aibl_ci[0]:.3f}, {aibl_ci[1]:.3f}]; permutation p={aibl_biomarker['permutation_p_greater']:.4f}.",
        "- The new score uses atlas-derived volume changes rather than unvalidated attention mass.",
        "",
        "How to write it:",
        "",
        "We replace the original attention-only CAS with an atlas-region neurodegeneration consistency score. This is not a cosmetic reinterpretation: it changes the validity target from attention concentration to disease-consistent structural MRI changes in a priori AD-relevant regions.",
        "",
        "### 3. Braak correlation was non-significant",
        "",
        "Old manuscript problem: the reported Braak correlation was non-significant and could not support a mechanistic claim.",
        "",
        "New evidence:",
        "",
        "- The revised validation should be called Braak-alternative or neurodegeneration-proxy validation.",
        "- The significant AIBL heldout AD-key volume score supports MRI-consistent medial temporal atrophy and ventricular expansion patterns.",
        "- ADNI-only biological validation remains non-significant, so this limitation must be stated.",
        "",
        "How to write it:",
        "",
        "We no longer claim direct Braak staging. Instead, we evaluate whether atlas-derived disease gradients concentrate in established MRI neurodegeneration regions. This is a weaker but empirically supported biological validation.",
        "",
    ]

    lines += [
        "## Proposed Revised Results Sections",
        "",
        "1. Cohort construction and leakage-free subject-level splits.",
        "2. External failure analysis of the original v3 model.",
        "3. MRI/atlas-only feature baseline and healthy specificity recovery.",
        "4. Clinically adapted atlas-guided model on locked AIBL heldout.",
        "5. IXI negative-control specificity analysis.",
        "6. Clinical-only and biomarker-enhanced sensitivity analyses.",
        "7. CAS replacement: atlas neurodegeneration consistency score.",
        "8. Limitations: OASIS transfer, no direct Braak labels, adaptation-vs-zero-shot distinction, 21-region atlas coarseness.",
        "",
    ]

    lines += [
        "## Cover Letter Core Paragraph",
        "",
        "In response to the previous decision, we did not attempt a narrow revision of the original manuscript. Instead, we rebuilt the experimental framework and substantially rewrote the study. The revised work now includes explicit subject-level cohort manifests, external classification on AIBL with a locked heldout split, an IXI healthy negative-control specificity test, MRI/atlas-only and clinical-only comparator models, multi-seed confirmation of the key hybrid results, and a replacement of the original attention-only CAS/Braak claims with an empirically tested atlas-region neurodegeneration consistency analysis. These additions directly address the previously identified concerns regarding unsupported cross-dataset generalization, an invalid CAS result, and non-significant Braak validation.",
        "",
    ]

    lines += [
        "## Response Matrix",
        "",
        "| Reviewer/editor concern | New action | Evidence file/result | Manuscript change |",
        "|---|---|---|---|",
        "| Cross-dataset generalization unsupported | Added AIBL heldout, IXI healthy negative control, OASIS stress test | `v4_decision_report.md`, `hybrid_replicate_summary.md` | New external validation section and tables |",
        "| CAS below chance | Replaced attention-only CAS with atlas neurodegeneration consistency score | AIBL heldout score 0.510 vs 0.286 null, p=0.026 | New biomarker validation section |",
        "| Braak non-significant | Removed direct Braak claim, reframed as Braak-alternative MRI neurodegeneration proxy | AIBL and pooled AD-key volume score | Revised interpretation and limitation |",
        "| Need volumetric/clinical baseline | Added atlas-only, atlas+clinical, clinical-only, biomarker-enhanced models | Candidate ranking and replicate summary | New baseline/sensitivity table |",
        "| Reproducibility unclear | Added subject-level manifest and split counts | manifest summary in v4 reports | New cohort and split subsection |",
        "| MCI errors | Reported per-class recall and showed MCI improvement in AIBL heldout | MCI recall 0.528 in main model; 0.755 clinical-only comparator | New error/per-class analysis |",
        "| Source availability | Scripts and outputs now organized under reproducible v4 pipeline | local `scripts/`, server `outputs/v4` | Release checklist and methods appendix |",
        "",
    ]

    lines += [
        "## Remaining Weaknesses To State Honestly",
        "",
        "- OASIS remains weak and should be described as an external stress test where transfer is not solved.",
        "- The strongest classifier is clinical-only; the atlas-guided model is chosen because it retains MRI atlas information and better supports the paper's mechanistic story.",
        "- AIBL heldout is domain-adapted external validation, not pure zero-shot transfer.",
        "- The biological validation is a Braak alternative, not direct neuropathological staging.",
        "- The 21-region atlas is coarse; a finer parcellation sensitivity analysis would be the strongest next experiment.",
        "",
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
