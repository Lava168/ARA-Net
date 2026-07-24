# Goal Completion Audit

- Overall status: **pass_with_bounded_limitations**
- Objective: Substantively rebuild ARA-Net for AD by reconstructing data/model experiments, addressing external classification, replacing invalid CAS, providing Braak-alternative biological evidence, and producing reproducible results plus manuscript rewrite evidence.

## Requirement-Level Evidence

| id | status | requirement | evidence | sources |
|---|---|---|---|---|
| `data_line` | **pass** | Rebuild the AD data line beyond the old v3 small revision. | Leakage-aware split inventory covers ADNI train/val/internal test, AIBL adaptation train/validation/locked heldout, OASIS stress test, and IXI healthy negative control with 4,388 scans and 2,023 split-inventory subjects. | `reports/v4/tables/table1_cohort_splits.md`<br>`scripts/build_v4_manifest.py` |
| `model_line` | **pass** | Lock a substantive rebuilt model rather than a v3 patch. | Locked model is `ARA-Net V6 subject-level rescue ensemble` with six probability streams, temperature 0.672, class offsets, and subject-level aggregation. AIBL BAcc improves from old v3 0.399 to final subject-level 0.833. | `deployment/final_ensemble_config.json`<br>`reports/v6_final_model/tables/final_model_classification_table.md`<br>`scripts/rescue_probability_optimizer.py` |
| `external_classification` | **bounded_pass** | Provide real external CN/MCI/AD classification evidence. | AIBL locked heldout subject endpoint n=216 has Acc 0.903, BAcc 0.833, macro AUC 0.937, AD-vs-CN AUC 1.000, recall CN/MCI/AD 0.961/0.686/0.852; IXI healthy CN retention is 1.000. This supports domain-adapted external heldout performance, not universal zero-shot transfer. | `reports/v6_final_model/final_rescue_model_summary_public.json`<br>`reports/v6_final_model/tables/final_model_classification_table.md`<br>`deployment/final_ensemble_config.json` |
| `stability_error` | **pass** | Add stability and MCI/AD error analysis for the final model. | AIBL BAcc 95% CI is 0.759-0.899; MCI recall CI is 0.531-0.839; AD recall CI is 0.710-0.966. AIBL AD-to-CN errors are 0; residual AD errors fall to MCI. | `reports/v6_final_model/final_rescue_model_summary_public.json`<br>`reports/v6_final_model/final_model_error_analysis.md` |
| `cas_validity` | **bounded_pass** | Resolve the invalid CAS issue with a defensible replacement. | Old attention-only CAS is removed as a central claim. Replacement AIBL AD-key atlas-volume consistency is 0.510 vs null 0.286, CI 0.479-0.526, p=0.026. The claim is limited to a structural MRI neurodegeneration proxy. | `reports/v4/tables/table4_neurodegeneration.md`<br>`reports/v6_final_model/claim_boundary_audit.md`<br>`reports/v6_final_model/word_manuscript_claim_audit_v6_docx.md` |
| `braak_alternative` | **bounded_pass** | Replace non-significant Braak validation with valid substitute biology. | All labeled AD-key consistency is 0.426 vs null 0.286, p=0.021. ADNI-only remains non-significant (score 0.342, p=0.184), so direct Braak-stage proof is explicitly not claimed. | `reports/v4/tables/table4_neurodegeneration.md`<br>`reports/v6_final_model/claim_boundary_audit.md`<br>`reports/v6_final_model/manuscript_v6_full_draft.md` |
| `manuscript_basis` | **pass** | Produce manuscript rewrite evidence and a generated Word draft. | Full V6 manuscript draft and generated DOCX exist. DOCX QA is pass with rendered page PNG/PDF inspection; Word claim-boundary audit reports 0 blockers and 0 warnings. | `reports/v6_final_model/manuscript_v6_full_draft.md`<br>`reports/v6_final_model/ARA-Net_V6_full_manuscript_draft.docx`<br>`reports/v6_final_model/ARA-Net_V6_full_manuscript_docx_qa.md`<br>`reports/v6_final_model/word_manuscript_claim_audit_v6_docx.md` |
| `reproducibility_public_package` | **pass** | Make the rebuilt work reproducible without exposing restricted data. | Public manifest reports 118 public tracked files and restricted-artifact check pass. Public scope is code, deployment wrappers, aggregate reports, final figures, documentation, and toy examples; raw datasets and row-level predictions are not redistributed. | `reports/v6_final_model/public_release_manifest.json`<br>`reports/v6_final_model/claim_boundary_audit.md`<br>`README.md`<br>`docs/DATA_CARD.md` |

## Explicit Non-Solved Boundaries

| item | evidence | manuscript boundary |
|---|---|---|
| OASIS external transfer | OASIS subject BAcc 0.334, macro AUC 0.554, AD-vs-CN AUC 0.371, recall CN/MCI/AD 0.966/0.034/0.000. | Stress-test limitation, not a claimed validation success. |
| Internal calibration | Internal subject BAcc 0.448; recall CN/MCI/AD 0.241/0.553/0.550. | Discuss as calibration/domain-shift limitation. |
| Clinical-only comparator | Clinical-only RF comparator has AIBL scan BAcc 0.835 and macro AUC 0.957. | Use as transparent upper bound, not as the central atlas-guided claim. |
| Clinical deployment | Not a medical device. Not cleared or approved for clinical use. | Open-source deployable research prototype only; not a medical device. |

## Decision

The repository evidence supports a substantive V6 rebuild and resubmission-ready evidence package, with explicit boundaries around OASIS, internal calibration, direct Braak proof, and clinical deployment.

This audit is intentionally conservative: it treats OASIS transfer, internal calibration, direct Braak-stage proof, and clinical deployment as limitations rather than solved claims.
