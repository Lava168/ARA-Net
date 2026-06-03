#!/usr/bin/env python3
"""Write a decision report for the v4 AD rescue experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.load(path.open()) if path.exists() else None


def metric(summary: dict, protocol: str, run: str, split: str) -> dict:
    return summary["results"][protocol][run]["metrics"][split]


def recalls(m: dict) -> str:
    pc = m.get("per_class", {})
    return "/".join(f"{pc.get(name, {}).get('recall', 0.0):.3f}" for name in ["CN", "MCI", "AD"])


def line_for(label: str, m: dict) -> str:
    return (
        f"- {label}: Acc={m.get('acc'):.3f}, BAcc={m.get('balanced_acc'):.3f}, "
        f"AUC={m.get('macro_auc_ovr'):.3f}" if m.get("macro_auc_ovr") is not None else
        f"- {label}: Acc={m.get('acc'):.3f}, BAcc={m.get('balanced_acc'):.3f}, AUC=NA"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = args.v4_root
    hybrid = load(root / "hybrid_atlas_clinical_baseline" / "summary.json")
    feature = load(root / "atlas_feature_baseline" / "summary.json")
    cascade = load(root / "atlas_cascade_baseline" / "summary.json")
    biomarker = load(root / "atlas_feature_biomarkers" / "summary.json")
    v3 = load(root.parent / "analysis" / "external_validation_v3_merged.json")
    manifest = load(root / "manifest_v4_summary.json")
    replicates = load(root / "hybrid_replicate_summary.json")

    lines = ["# V4 Decision Report", ""]
    lines += [
        "## Recommendation",
        "",
        "Use a two-tier story, not a single overclaimed model:",
        "",
        "1. MRI/atlas-only evidence shows the original v3 failure mode was fixed for healthy specificity but not fully for MCI staging.",
        "2. The main atlas-guided multimodal model should be `aibl_adapted/atlas_core_clinical__hgb`: it keeps the atlas MRI signal, uses core clinical variables, and performs well on locked AIBL heldout plus IXI.",
        "3. `aibl_adapted/clinical_core_only__rf_balanced` is the strongest classifier but should be presented as a clinical-only comparator or upper-bound, not as the atlas-guided model.",
        "4. `atlas_biomarker_enhanced__hgb` is a biomarker-enhanced sensitivity analysis, not the clean main model.",
        "5. ADNI-only zero-shot models remain weak on AIBL AD detection; do not claim pure zero-shot cross-cohort staging is solved.",
        "",
    ]

    if manifest:
        split_counts = manifest.get("split_counts", {})
        lines += ["## Data Split Evidence", ""]
        for split in ["train", "val", "internal_test", "aibl_adapt_train", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]:
            item = split_counts.get(split)
            if item:
                labels = item.get("labels", {})
                lines.append(
                    f"- {split}: scans={item.get('scans')}, subjects={item.get('subjects')}, "
                    f"CN/MCI/AD={labels.get('CN',0)}/{labels.get('MCI',0)}/{labels.get('AD',0)}"
                )
        lines.append("")

    if v3:
        lines += ["## Old v3 Failure Baseline", ""]
        for name, m in v3["ensemble"].items():
            lines.append(
                f"- {name}: Acc={m.get('accuracy'):.3f}, BAcc={m.get('balanced_accuracy_present'):.3f}, "
                f"AUC={m.get('macro_auc_ovr_valid')}, pred={m.get('prediction_distribution')}"
            )
        lines.append("")

    lines += ["## Classification Decision", ""]
    if hybrid:
        for run, title in [
            ("atlas_core_clinical__hgb", "Recommended atlas-guided multimodal model"),
            ("clinical_core_only__rf_balanced", "Strong clinical-only comparator"),
            ("atlas_biomarker_enhanced__hgb", "Biomarker-enhanced sensitivity model"),
        ]:
            lines += [f"### {title}: `{run}`", ""]
            for split in ["internal_test", "aibl_adapt_val", "aibl_heldout", "ixi_external", "oasis_external"]:
                m = metric(hybrid, "aibl_adapted", run, split)
                auc = "NA" if m.get("macro_auc_ovr") is None else f"{m.get('macro_auc_ovr'):.3f}"
                extra = ""
                if m.get("ad_vs_cn_auc") is not None:
                    extra += f", ADvCN_AUC={m.get('ad_vs_cn_auc'):.3f}"
                if m.get("cn_retention_rate") is not None:
                    extra += f", CN_retention={m.get('cn_retention_rate'):.3f}"
                lines.append(
                    f"- {split}: Acc={m.get('acc'):.3f}, BAcc={m.get('balanced_acc'):.3f}, "
                    f"AUC={auc}{extra}, recall_CN/MCI/AD={recalls(m)}, pred={m.get('prediction_distribution')}"
                )
            lines.append("")

    if feature:
        best_name = feature["best_model"]
        m = feature["results"][best_name]["metrics"]["aibl_heldout"]
        ix = feature["results"][best_name]["metrics"]["ixi_external"]
        lines += [
            "## MRI/Atlas-Only Baseline",
            "",
            f"- Best atlas-only model `{best_name}` AIBL heldout: BAcc={m.get('balanced_acc'):.3f}, AUC={m.get('macro_auc_ovr'):.3f}, recall_CN/MCI/AD={recalls(m)}.",
            f"- IXI healthy specificity: CN retention={ix.get('cn_retention_rate'):.3f}.",
            "- Interpretation: useful specificity and AD-vs-CN signal, but MCI/AD staging is insufficient without clinical adaptation.",
            "",
        ]

    if cascade:
        best_name = cascade["best_model"]
        m = cascade["results"][best_name]["metrics"]["aibl_heldout"]
        ix = cascade["results"][best_name]["metrics"]["ixi_external"]
        lines += [
            "## Cascade Baseline",
            "",
            f"- Best cascade `{best_name}` AIBL heldout: BAcc={m.get('balanced_acc'):.3f}, AUC={m.get('macro_auc_ovr'):.3f}, recall_CN/MCI/AD={recalls(m)}.",
            f"- IXI CN retention={ix.get('cn_retention_rate'):.3f}.",
            "- Interpretation: excellent healthy specificity, but it fails MCI detection on heldout AIBL.",
            "",
        ]

    if biomarker:
        lines += ["## CAS / Braak-Alternative Biological Validation", ""]
        for group in ["aibl_heldout", "aibl_adapt_heldout", "all_labeled_ad", "adni_val_internal_test"]:
            if group not in biomarker:
                continue
            score = biomarker[group]["ad_key_volume_score"]
            ci = score["bootstrap_ci"]
            lines.append(
                f"- {group}: AD-key volume score={score['ad_key_score']:.3f}, "
                f"uniform={score['uniform_null']:.3f}, delta={score['score_minus_uniform']:.3f}, "
                f"CI=[{ci[0]:.3f}, {ci[1]:.3f}], permutation p={score['permutation_p_greater']:.4f}."
            )
        lines.append(
            "- Interpretation: valid as an MRI neurodegeneration/Braak-proxy check, especially in AIBL and pooled labeled AD; ADNI validation alone is not significant, so do not claim direct Braak staging."
        )
        lines.append("")

    if replicates:
        missing = replicates.get("missing", [])
        lines += ["## Stability Status", ""]
        if missing:
            lines.append(f"- Multi-seed replicate check is running or incomplete: missing {len(missing)} seed summaries.")
        else:
            lines.append("- Multi-seed replicate check completed.")
        for run, item in replicates.get("runs", {}).items():
            stats = item.get("summary", {})
            bacc = stats.get("aibl_heldout.balanced_acc")
            cnret = stats.get("ixi_external.cn_retention_rate")
            if bacc and cnret:
                lines.append(
                    f"- {run}: AIBL heldout BAcc={bacc['mean']:.3f}+/-{bacc['std']:.3f}, "
                    f"IXI CN retention={cnret['mean']:.3f}+/-{cnret['std']:.3f} (n={bacc['n']})."
                )
        lines.append("")

    stability_phrase = "OASIS transfer remains weak and should be framed as a limitation."
    if replicates and replicates.get("missing"):
        stability_phrase = "The remaining weakness is OASIS transfer and incomplete multi-seed confirmation."
    elif replicates:
        stability_phrase = "Multi-seed confirmation is complete for the key hybrid models; OASIS transfer remains the main unresolved weakness."

    lines += [
        "## Manuscript Implication",
        "",
        f"This is now a substantive new work if framed correctly: a cross-cohort atlas-guided and clinically adapted AD staging framework with locked AIBL heldout evaluation, IXI healthy negative-control specificity, and atlas-region biological validation. {stability_phrase}",
        "",
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
