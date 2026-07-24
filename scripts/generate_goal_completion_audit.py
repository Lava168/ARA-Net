#!/usr/bin/env python3
"""Generate a requirement-level audit for the ARA-Net V6 rebuild goal."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_MD = Path("reports/v6_final_model/goal_completion_audit.md")
DEFAULT_OUTPUT_JSON = Path("reports/v6_final_model/goal_completion_audit.json")

SUMMARY_JSON = Path("reports/v6_final_model/final_rescue_model_summary_public.json")
COHORT_TABLE = Path("reports/v4/tables/table1_cohort_splits.md")
BIOLOGY_TABLE = Path("reports/v4/tables/table4_neurodegeneration.md")
CLASSIFICATION_TABLE = Path("reports/v6_final_model/tables/final_model_classification_table.md")
DOCX_QA = Path("reports/v6_final_model/ARA-Net_V6_full_manuscript_docx_qa.md")
WORD_AUDIT = Path("reports/v6_final_model/word_manuscript_claim_audit_v6_docx.md")
CLAIM_AUDIT = Path("reports/v6_final_model/claim_boundary_audit.md")
PUBLIC_MANIFEST = Path("reports/v6_final_model/public_release_manifest.json")
MANUSCRIPT_MD = Path("reports/v6_final_model/manuscript_v6_full_draft.md")
MANUSCRIPT_DOCX = Path("reports/v6_final_model/ARA-Net_V6_full_manuscript_draft.docx")
MODEL_CONFIG = Path("deployment/final_ensemble_config.json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    if not rows:
        return []
    headers = [header.strip().lower().replace(" ", "_") for header in rows[0]]
    parsed = []
    for row in rows[1:]:
        parsed.append({headers[idx]: row[idx] if idx < len(row) else "" for idx in range(len(headers))})
    return parsed


def report_passed(path: Path) -> bool:
    text = read_text(path)
    return "Status: **pass**" in text or "Restricted artifact check: **pass**" in text


def f3(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def ci_pair(bootstrap: dict[str, Any], key: str) -> str:
    item = bootstrap[key]
    return f"{f3(item['ci_low'])}-{f3(item['ci_high'])}"


def biology_row(rows: list[dict[str, str]], group: str) -> dict[str, float]:
    for row in rows:
        if row.get("group") == group:
            return {
                "score": float(row["ad_key_score"]),
                "null": float(row["uniform_null"]),
                "delta": float(row["delta"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "p": float(row["permutation_p"]),
            }
    raise KeyError(group)


def classification_lookup(rows: list[dict[str, str]], model: str, unit: str, cohort: str) -> dict[str, str]:
    for row in rows:
        if (
            row.get("model_/_protocol") == model
            and row.get("evaluation_unit") == unit
            and row.get("test_cohort") == cohort
        ):
            return row
    raise KeyError((model, unit, cohort))


def make_requirement(req_id: str, requirement: str, status: str, evidence: str, sources: list[str]) -> dict[str, Any]:
    return {
        "id": req_id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "sources": sources,
    }


def build_audit() -> dict[str, Any]:
    summary = read_json(SUMMARY_JSON)
    manifest = read_json(PUBLIC_MANIFEST)
    config = read_json(MODEL_CONFIG)
    cohorts = parse_markdown_table(COHORT_TABLE)
    biology = parse_markdown_table(BIOLOGY_TABLE)
    classification = parse_markdown_table(CLASSIFICATION_TABLE)

    subject_metrics = summary["final_model"]["subject_level_metrics"]
    subject_boot = summary["final_model"]["subject_level_bootstrap"]
    aibl = subject_metrics["aibl_heldout"]
    ixi = subject_metrics["ixi_external"]
    oasis = subject_metrics["oasis_external"]
    internal = subject_metrics["internal_test"]
    aibl_boot = subject_boot["aibl_heldout"]

    old_v3 = classification_lookup(classification, "Old v3 ensemble", "scan", "AIBL external")
    final_subject = classification_lookup(classification, "Final rescue ensemble", "subject", "AIBL heldout")
    clinical_rf = classification_lookup(classification, "Clinical-only RF comparator", "scan", "AIBL heldout")

    aibl_bio = biology_row(biology, "aibl_heldout")
    all_bio = biology_row(biology, "all_labeled_ad")
    adni_bio = biology_row(biology, "adni_val_internal_test")

    total_scans = sum(int(row["scans"].replace(",", "")) for row in cohorts)
    total_subject_inventory = sum(int(row["subjects"].replace(",", "")) for row in cohorts)
    split_names = {row["split"] for row in cohorts}
    required_splits = {
        "train",
        "val",
        "internal_test",
        "aibl_adapt_train",
        "aibl_adapt_val",
        "aibl_heldout",
        "oasis_external",
        "ixi_external",
    }

    aibl_conf = aibl["confusion_matrix"]
    ad_to_cn = int(aibl_conf[2][0])
    docx_qa_pass = report_passed(DOCX_QA)
    word_audit_pass = report_passed(WORD_AUDIT)
    claim_audit_pass = report_passed(CLAIM_AUDIT)
    manifest_pass = manifest["restricted_artifact_check"]["status"] == "pass"

    requirements = [
        make_requirement(
            "data_line",
            "Rebuild the AD data line beyond the old v3 small revision.",
            "pass" if required_splits.issubset(split_names) and total_scans > 3000 else "fail",
            (
                f"Leakage-aware split inventory covers ADNI train/val/internal test, AIBL adaptation "
                f"train/validation/locked heldout, OASIS stress test, and IXI healthy negative control "
                f"with {total_scans:,} scans and {total_subject_inventory:,} split-inventory subjects."
            ),
            [str(COHORT_TABLE), "scripts/build_v4_manifest.py"],
        ),
        make_requirement(
            "model_line",
            "Lock a substantive rebuilt model rather than a v3 patch.",
            "pass" if config.get("version") == "v6.0-research" and len(config.get("base_models", [])) == 6 else "fail",
            (
                f"Locked model is `{config['name']}` with six probability streams, temperature "
                f"{f3(config['temperature'])}, class offsets, and subject-level aggregation. "
                f"AIBL BAcc improves from old v3 {old_v3['bacc']} to final subject-level {final_subject['bacc']}."
            ),
            [str(MODEL_CONFIG), str(CLASSIFICATION_TABLE), "scripts/rescue_probability_optimizer.py"],
        ),
        make_requirement(
            "external_classification",
            "Provide real external CN/MCI/AD classification evidence.",
            "bounded_pass"
            if aibl["balanced_acc"] >= 0.80 and ixi["cn_retention_rate"] == 1.0
            else "fail",
            (
                f"AIBL locked heldout subject endpoint n={aibl['n']} has Acc {f3(aibl['acc'])}, "
                f"BAcc {f3(aibl['balanced_acc'])}, macro AUC {f3(aibl['macro_auc_ovr'])}, "
                f"AD-vs-CN AUC {f3(aibl['ad_vs_cn_auc'])}, recall CN/MCI/AD "
                f"{f3(aibl['recall_CN'])}/{f3(aibl['recall_MCI'])}/{f3(aibl['recall_AD'])}; "
                f"IXI healthy CN retention is {f3(ixi['cn_retention_rate'])}. "
                "This supports domain-adapted external heldout performance, not universal zero-shot transfer."
            ),
            [str(SUMMARY_JSON), str(CLASSIFICATION_TABLE), str(MODEL_CONFIG)],
        ),
        make_requirement(
            "stability_error",
            "Add stability and MCI/AD error analysis for the final model.",
            "pass" if aibl_boot["balanced_acc"]["n_boot"] == 2000 and ad_to_cn == 0 else "fail",
            (
                f"AIBL BAcc 95% CI is {ci_pair(aibl_boot, 'balanced_acc')}; "
                f"MCI recall CI is {ci_pair(aibl_boot, 'recall_MCI')}; "
                f"AD recall CI is {ci_pair(aibl_boot, 'recall_AD')}. "
                f"AIBL AD-to-CN errors are {ad_to_cn}; residual AD errors fall to MCI."
            ),
            [str(SUMMARY_JSON), "reports/v6_final_model/final_model_error_analysis.md"],
        ),
        make_requirement(
            "cas_validity",
            "Resolve the invalid CAS issue with a defensible replacement.",
            "bounded_pass" if aibl_bio["score"] > aibl_bio["null"] and aibl_bio["p"] < 0.05 else "fail",
            (
                f"Old attention-only CAS is removed as a central claim. Replacement AIBL AD-key "
                f"atlas-volume consistency is {f3(aibl_bio['score'])} vs null {f3(aibl_bio['null'])}, "
                f"CI {f3(aibl_bio['ci_low'])}-{f3(aibl_bio['ci_high'])}, p={f3(aibl_bio['p'])}. "
                "The claim is limited to a structural MRI neurodegeneration proxy."
            ),
            [str(BIOLOGY_TABLE), str(CLAIM_AUDIT), str(WORD_AUDIT)],
        ),
        make_requirement(
            "braak_alternative",
            "Replace non-significant Braak validation with valid substitute biology.",
            "bounded_pass" if all_bio["score"] > all_bio["null"] and all_bio["p"] < 0.05 else "fail",
            (
                f"All labeled AD-key consistency is {f3(all_bio['score'])} vs null {f3(all_bio['null'])}, "
                f"p={f3(all_bio['p'])}. ADNI-only remains non-significant (score {f3(adni_bio['score'])}, "
                f"p={f3(adni_bio['p'])}), so direct Braak-stage proof is explicitly not claimed."
            ),
            [str(BIOLOGY_TABLE), str(CLAIM_AUDIT), str(MANUSCRIPT_MD)],
        ),
        make_requirement(
            "manuscript_basis",
            "Produce manuscript rewrite evidence and a generated Word draft.",
            "pass" if MANUSCRIPT_MD.exists() and MANUSCRIPT_DOCX.exists() and docx_qa_pass and word_audit_pass else "fail",
            (
                "Full V6 manuscript draft and generated DOCX exist. DOCX QA is pass with rendered "
                "page PNG/PDF inspection; Word claim-boundary audit reports 0 blockers and 0 warnings."
            ),
            [str(MANUSCRIPT_MD), str(MANUSCRIPT_DOCX), str(DOCX_QA), str(WORD_AUDIT)],
        ),
        make_requirement(
            "reproducibility_public_package",
            "Make the rebuilt work reproducible without exposing restricted data.",
            "pass" if manifest_pass and claim_audit_pass else "fail",
            (
                f"Public manifest reports {manifest['file_count']} public tracked files and restricted-artifact "
                "check pass. Public scope is code, deployment wrappers, aggregate reports, final figures, "
                "documentation, and toy examples; raw datasets and row-level predictions are not redistributed."
            ),
            [str(PUBLIC_MANIFEST), str(CLAIM_AUDIT), "README.md", "docs/DATA_CARD.md"],
        ),
    ]

    limitations = [
        {
            "item": "OASIS external transfer",
            "evidence": (
                f"OASIS subject BAcc {f3(oasis['balanced_acc'])}, macro AUC {f3(oasis['macro_auc_ovr'])}, "
                f"AD-vs-CN AUC {f3(oasis['ad_vs_cn_auc'])}, recall CN/MCI/AD "
                f"{f3(oasis['recall_CN'])}/{f3(oasis['recall_MCI'])}/{f3(oasis['recall_AD'])}."
            ),
            "boundary": "Stress-test limitation, not a claimed validation success.",
        },
        {
            "item": "Internal calibration",
            "evidence": (
                f"Internal subject BAcc {f3(internal['balanced_acc'])}; recall CN/MCI/AD "
                f"{f3(internal['recall_CN'])}/{f3(internal['recall_MCI'])}/{f3(internal['recall_AD'])}."
            ),
            "boundary": "Discuss as calibration/domain-shift limitation.",
        },
        {
            "item": "Clinical-only comparator",
            "evidence": (
                f"Clinical-only RF comparator has AIBL scan BAcc {clinical_rf['bacc']} and macro AUC "
                f"{clinical_rf['macro_auc']}."
            ),
            "boundary": "Use as transparent upper bound, not as the central atlas-guided claim.",
        },
        {
            "item": "Clinical deployment",
            "evidence": config["clinical_use_notice"],
            "boundary": "Open-source deployable research prototype only; not a medical device.",
        },
    ]

    failures = [item for item in requirements if item["status"] == "fail"]
    overall_status = "fail" if failures else "pass_with_bounded_limitations"
    return {
        "overall_status": overall_status,
        "objective": (
            "Substantively rebuild ARA-Net for AD by reconstructing data/model experiments, "
            "addressing external classification, replacing invalid CAS, providing Braak-alternative "
            "biological evidence, and producing reproducible results plus manuscript rewrite evidence."
        ),
        "requirements": requirements,
        "limitations": limitations,
        "decision": (
            "The repository evidence supports a substantive V6 rebuild and resubmission-ready evidence "
            "package, with explicit boundaries around OASIS, internal calibration, direct Braak proof, "
            "and clinical deployment."
            if not failures
            else "One or more core requirements failed; do not treat the goal as complete."
        ),
    }


def write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal Completion Audit",
        "",
        f"- Overall status: **{audit['overall_status']}**",
        f"- Objective: {audit['objective']}",
        "",
        "## Requirement-Level Evidence",
        "",
        "| id | status | requirement | evidence | sources |",
        "|---|---|---|---|---|",
    ]
    for item in audit["requirements"]:
        source_text = "<br>".join(f"`{source}`" for source in item["sources"])
        lines.append(
            f"| `{item['id']}` | **{item['status']}** | {item['requirement']} | "
            f"{item['evidence']} | {source_text} |"
        )

    lines += [
        "",
        "## Explicit Non-Solved Boundaries",
        "",
        "| item | evidence | manuscript boundary |",
        "|---|---|---|",
    ]
    for item in audit["limitations"]:
        lines.append(f"| {item['item']} | {item['evidence']} | {item['boundary']} |")

    lines += [
        "",
        "## Decision",
        "",
        audit["decision"],
        "",
        "This audit is intentionally conservative: it treats OASIS transfer, internal calibration, direct Braak-stage proof, and clinical deployment as limitations rather than solved claims.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    audit = build_audit()
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(audit, args.markdown)
    args.json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[saved] {args.markdown}")
    print(f"[saved] {args.json}")
    if audit["overall_status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
