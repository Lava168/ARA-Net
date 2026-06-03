# Public Release Manifest

- Public tracked file count: 115
- Restricted artifact check: **pass**
- Public scope: code, deployment wrapper, aggregate reports, final figures, documentation, and toy probability examples.
- Not redistributed: raw ADNI/AIBL/OASIS/IXI data, private clinical spreadsheets, row-level subject/scan predictions, MRI volumes, and model checkpoints.
- Manifest note: Manifest output files and tracked restricted artifacts are excluded from the public file list.

## Category Counts

| category | files |
|---|---:|
| aggregate_reports | 48 |
| analysis_code | 30 |
| deployment | 4 |
| documentation | 8 |
| environment | 2 |
| examples | 1 |
| figures | 18 |
| frontend | 3 |
| other | 1 |

## Reproduction Commands

| name | command | scope |
|---|---|---|
| Research CLI smoke test | `python deployment/research_inference.py --input-csv examples/probability_input_example.csv --output examples/predictions_subject.csv --unit subject` | Checks the public probability-ensemble wrapper. |
| Core reviewer evidence matrix | `python scripts/generate_core_reviewer_evidence_matrix.py` | Regenerates the external classification / CAS replacement / Braak-alternative evidence matrix. |
| Claim boundary audit | `python scripts/audit_claim_boundaries.py` | Scans public Git-tracked files for unsupported Braak/CAS/OASIS/zero-shot/clinical-deployment overclaims. |
| Public release manifest | `python scripts/generate_public_release_manifest.py` | Regenerates the public-file manifest and restricted-artifact check. |
| Final figures | `python scripts/generate_v6_final_figures.py --summary reports/v6_final_model/final_rescue_model_summary_public.json --table2 reports/v4/tables/table2_classification.csv --table-dir reports/v6_final_model/tables --out-dir reports/v6_final_model/figures` | Regenerates v6 aggregate manuscript figures after upstream result files exist. |

## Restricted Artifact Check

No tracked files matched the restricted row-level/data/model-artifact patterns.

## Public Files

### aggregate_reports

| path | bytes | sha256 | role |
|---|---:|---|---|
| `reports/v4/complete_manuscript_v4.md` | 23959 | `82f46f4e0327` | Public repository file. |
| `reports/v4/cover_letter_rebuild_draft.md` | 3516 | `763e096a4aa7` | Public repository file. |
| `reports/v4/figure_table_todo.md` | 4331 | `2845a3387b19` | Public repository file. |
| `reports/v4/figures/figure_captions.json` | 951 | `8f6abd09ca61` | Public repository file. |
| `reports/v4/full_manuscript_skeleton.md` | 15918 | `dd7b380e9499` | Public repository file. |
| `reports/v4/hybrid_candidate_ranking.md` | 5335 | `48d82ce9c1ea` | Public repository file. |
| `reports/v4/hybrid_replicate_summary.md` | 1635 | `7744bc7c2c51` | Public repository file. |
| `reports/v4/manuscript_insertable_sections.md` | 5054 | `86fa3b055d24` | Public repository file. |
| `reports/v4/manuscript_rewrite_package.md` | 8951 | `3f554379f439` | Public repository file. |
| `reports/v4/review_response_rebuild_matrix.md` | 7179 | `352657a18872` | Public repository file. |
| `reports/v4/tables/table1_cohort_splits.csv` | 483 | `067337b3681f` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table1_cohort_splits.md` | 648 | `9adfed804f98` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table2_classification.csv` | 1645 | `9eac4ffe8c26` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table2_classification.md` | 1278 | `ac486d8b4fe9` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table3_replicates.csv` | 411 | `d59f246389f4` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table3_replicates.md` | 393 | `64b632695b36` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table4_neurodegeneration.csv` | 608 | `d1ecf85ce88d` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/tables/table4_neurodegeneration.md` | 391 | `feba23341865` | Aggregate v4 rebuild table used by final evidence reports. |
| `reports/v4/v4_decision_report.md` | 6022 | `c0dfe7334088` | Public repository file. |
| `reports/v4/v4_progress_summary.md` | 7219 | `2a20c097f393` | Public repository file. |
| `reports/v5_rescue/rescue_interim_metrics.json` | 13418 | `515cd64884fc` | Public repository file. |
| `reports/v5_rescue/rescue_probability_no_oasis_tune.md` | 5733 | `7bb3a684ab2d` | Public repository file. |
| `reports/v5_rescue/rescue_probability_subject_quick_no_oasis_tune.md` | 4711 | `d36630555756` | Public repository file. |
| `reports/v5_rescue/rescue_probability_targeted_no_oasis_tune.md` | 4734 | `ac50bb888bf6` | Public repository file. |
| `reports/v5_rescue/rescue_status.md` | 3672 | `6194263d1267` | Public repository file. |
| `reports/v6_final_model/ARA-Net_V6_full_manuscript_docx_qa.md` | 1172 | `2e56c5a068ba` | Structural QA report for the generated V6 DOCX. |
| `reports/v6_final_model/ARA-Net_V6_full_manuscript_draft.docx` | 49090 | `84945dbf695e` | Generated V6 Word manuscript replacement draft. |
| `reports/v6_final_model/claim_boundary_audit.md` | 31166 | `7268976ec7e5` | Public claim-boundary audit for reviewer-safe wording. |
| `reports/v6_final_model/clinical_translation_roadmap.md` | 3166 | `1cc4f087bda8` | Public repository file. |
| `reports/v6_final_model/core_reviewer_evidence_matrix.md` | 4913 | `2197d755812e` | Three-core-issue reviewer evidence matrix. |
| `reports/v6_final_model/deployment_code_clinical_response.md` | 4896 | `ae28cdc7bb21` | Public repository file. |
| `reports/v6_final_model/final_figure_blueprint.md` | 9566 | `779342f52391` | Panel-by-panel final figure blueprint. |
| `reports/v6_final_model/final_model_error_analysis.md` | 3049 | `63403ec1536f` | Public repository file. |
| `reports/v6_final_model/final_rescue_model_lock_report.md` | 3391 | `e15c745c0f69` | Public repository file. |
| `reports/v6_final_model/final_rescue_model_summary_public.json` | 59167 | `1248b2f9da44` | Public aggregate final-model metrics and bootstrap evidence. |
| `reports/v6_final_model/final_submission_checklist.md` | 5454 | `c79d9961ba38` | Public repository file. |
| `reports/v6_final_model/final_submission_closure_packet.md` | 6559 | `e45dea92d2ab` | Final manuscript-integration packet for figures, OASIS handling, citations, and terminology. |
| `reports/v6_final_model/manuscript_v6_full_draft.md` | 24983 | `a326ba374161` | Full V6 manuscript draft for replacing the old Word manuscript body. |
| `reports/v6_final_model/manuscript_v6_rewrite_package.md` | 15586 | `5af34abdf0dd` | Public repository file. |
| `reports/v6_final_model/reviewer_response_v6_matrix.md` | 4745 | `579a274ef68b` | Public repository file. |
| `reports/v6_final_model/tables/aibl_heldout_confusion_transitions.csv` | 294 | `21d4b97d0233` | Aggregate final v6 table. |
| `reports/v6_final_model/tables/aibl_heldout_error_group_features.csv` | 3549 | `da0476e96fe4` | Aggregate final v6 table. |
| `reports/v6_final_model/tables/final_model_classification_table.md` | 1910 | `737ecbcb91a8` | Aggregate final v6 table. |
| `reports/v6_final_model/tables/internal_test_confusion_transitions.csv` | 263 | `061897c6e6c2` | Aggregate final v6 table. |
| `reports/v6_final_model/tables/internal_test_error_group_features.csv` | 3425 | `c781b1d4e33e` | Aggregate final v6 table. |
| `reports/v6_final_model/word_manuscript_claim_audit.md` | 34247 | `5775322eb05f` | Generated claim-boundary audit for the uploaded Word manuscript. |
| `reports/v6_final_model/word_manuscript_claim_audit_v6_docx.md` | 3943 | `806a1c2f75ed` | Word claim-boundary audit for the generated V6 DOCX. |
| `reports/v6_final_model/word_manuscript_rewrite_map.md` | 9051 | `9419af33a251` | Section-level map for rewriting the old Word manuscript into the V6 submission. |

### analysis_code

| path | bytes | sha256 | role |
|---|---:|---|---|
| `scripts/analyze_atlas_feature_biomarkers.py` | 7071 | `24f8d4e2b024` | Reproducible analysis or report-generation script. |
| `scripts/audit_ad_data.py` | 6283 | `9bd03c8ad6ad` | Reproducible analysis or report-generation script. |
| `scripts/audit_claim_boundaries.py` | 9474 | `a327a8060be7` | Public overclaim audit script. |
| `scripts/audit_word_manuscript_claims.py` | 8520 | `df03e159af5e` | Word manuscript claim-boundary audit script. |
| `scripts/build_oasis_locked_manifest.py` | 4947 | `d16cc56825fd` | Reproducible analysis or report-generation script. |
| `scripts/build_v4_manifest.py` | 8732 | `35605c2519df` | Reproducible analysis or report-generation script. |
| `scripts/build_v6_manuscript_docx.py` | 12380 | `292a2bcb436c` | DOCX builder for the V6 manuscript replacement draft. |
| `scripts/evaluate_external_ad_datasets.py` | 31196 | `01f333bac625` | Reproducible analysis or report-generation script. |
| `scripts/export_manuscript_tables.py` | 7116 | `18b65986b620` | Reproducible analysis or report-generation script. |
| `scripts/extract_v4_attention_biomarkers.py` | 9777 | `d88d66b9c65c` | Reproducible analysis or report-generation script. |
| `scripts/final_rescue_model_package.py` | 36667 | `1d89b8f65004` | Reproducible analysis or report-generation script. |
| `scripts/generate_core_reviewer_evidence_matrix.py` | 12221 | `61e240505824` | Reproducible analysis or report-generation script. |
| `scripts/generate_manuscript_figures.py` | 26774 | `b6a9eae6f5a8` | Reproducible analysis or report-generation script. |
| `scripts/generate_public_release_manifest.py` | 11691 | `e8cf50842e97` | Reproducible analysis or report-generation script. |
| `scripts/generate_v6_final_figures.py` | 13200 | `ba1ff4f804ec` | Reproducible analysis or report-generation script. |
| `scripts/launch_external_validation_shards.sh` | 949 | `57072603751b` | Reproducible analysis or report-generation script. |
| `scripts/merge_external_validation_shards.py` | 9540 | `6096bd336cdc` | Reproducible analysis or report-generation script. |
| `scripts/qa_v6_manuscript_docx.py` | 6243 | `1990e2c7e19f` | Structural DOCX QA script for the V6 manuscript replacement draft. |
| `scripts/rank_hybrid_candidates.py` | 3320 | `c0007484594a` | Reproducible analysis or report-generation script. |
| `scripts/rescue_probability_optimizer.py` | 23596 | `a8d5ce00b5a3` | Reproducible analysis or report-generation script. |
| `scripts/summarize_hybrid_replicates.py` | 5774 | `78e281e7bc9c` | Reproducible analysis or report-generation script. |
| `scripts/summarize_v4_results.py` | 7740 | `560b2ef1dda5` | Reproducible analysis or report-generation script. |
| `scripts/train_atlas_cascade_baseline.py` | 9079 | `985a31965067` | Reproducible analysis or report-generation script. |
| `scripts/train_atlas_feature_baseline.py` | 15276 | `da10fa0a2634` | Reproducible analysis or report-generation script. |
| `scripts/train_hybrid_atlas_clinical_baseline.py` | 19673 | `76de733f05b8` | Reproducible analysis or report-generation script. |
| `scripts/train_rescue_hybrid_search.py` | 24236 | `b72d6dcb9aa6` | Reproducible analysis or report-generation script. |
| `scripts/train_v4_external_generalization.py` | 33397 | `72643927f7d2` | Reproducible analysis or report-generation script. |
| `scripts/write_complete_manuscript_v4.py` | 5768 | `1f27867e8713` | Reproducible analysis or report-generation script. |
| `scripts/write_manuscript_rewrite_package.py` | 14792 | `48a0ff6d4e95` | Reproducible analysis or report-generation script. |
| `scripts/write_v4_decision_report.py` | 9008 | `2c2bf24ceba8` | Reproducible analysis or report-generation script. |

### deployment

| path | bytes | sha256 | role |
|---|---:|---|---|
| `Dockerfile` | 372 | `9cacd6791aa5` | Public repository file. |
| `deployment/final_ensemble_config.json` | 1663 | `11f8d381391a` | Locked deployable ensemble configuration. |
| `deployment/research_api.py` | 7058 | `b96e4f15cd4b` | Research inference/deployment wrapper. |
| `deployment/research_inference.py` | 8194 | `dccc5f453997` | Research inference/deployment wrapper. |

### documentation

| path | bytes | sha256 | role |
|---|---:|---|---|
| `LICENSE` | 1064 | `5f7e61045d8d` | Public repository file. |
| `README.md` | 7279 | `b48ebf1d4aca` | Public repository file. |
| `docs/CLINICAL_VALIDATION_PROTOCOL.md` | 2492 | `41d335eb58ad` | Public documentation and claim-boundary document. |
| `docs/DATA_CARD.md` | 1682 | `01c0c7d68111` | Public documentation and claim-boundary document. |
| `docs/MODEL_CARD.md` | 2447 | `0505dd364222` | Public documentation and claim-boundary document. |
| `docs/OPEN_SOURCE_AND_DEPLOYMENT.md` | 1571 | `ffaa9327d643` | Public documentation and claim-boundary document. |
| `docs/REGULATORY_NOTES.md` | 2174 | `ae8196036c9f` | Public documentation and claim-boundary document. |
| `reports/v6_final_model/README.md` | 3374 | `54c45f60919d` | Public repository file. |

### environment

| path | bytes | sha256 | role |
|---|---:|---|---|
| `requirements-deploy.txt` | 12 | `160fed6c8729` | Public repository file. |
| `requirements.txt` | 68 | `55d7f845ad23` | Public repository file. |

### examples

| path | bytes | sha256 | role |
|---|---:|---|---|
| `examples/probability_input_example.csv` | 1367 | `905ed8c0ce97` | Public repository file. |

### figures

| path | bytes | sha256 | role |
|---|---:|---|---|
| `reports/v4/figures/figure1_revised_study_design.pdf` | 35143 | `df2af177861a` | Public repository file. |
| `reports/v4/figures/figure1_revised_study_design.png` | 654328 | `1e94e9ba758a` | Public repository file. |
| `reports/v4/figures/figure2_external_classification_improvement.pdf` | 27744 | `6c3bea0849f0` | Public repository file. |
| `reports/v4/figures/figure2_external_classification_improvement.png` | 243890 | `74da4fb269d9` | Public repository file. |
| `reports/v4/figures/figure3_aibl_confusion_matrices.pdf` | 31013 | `9ad75e2e2f20` | Public repository file. |
| `reports/v4/figures/figure3_aibl_confusion_matrices.png` | 198476 | `662650760106` | Public repository file. |
| `reports/v4/figures/figure4_neurodegeneration_consistency.pdf` | 31046 | `f8f7fe2c13ac` | Public repository file. |
| `reports/v4/figures/figure4_neurodegeneration_consistency.png` | 320777 | `df2bdfe59a27` | Public repository file. |
| `reports/v4/figures/figure5_oasis_stress_test.pdf` | 26417 | `9fe1de050f33` | Public repository file. |
| `reports/v4/figures/figure5_oasis_stress_test.png` | 198920 | `2de3ae318fd7` | Public repository file. |
| `reports/v6_final_model/figures/figure2_final_external_rescue.pdf` | 29776 | `76a2189dbea9` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure2_final_external_rescue.png` | 233499 | `cdd6b74a6738` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure3_final_subject_confusion.pdf` | 31352 | `cbd871cb3e48` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure3_final_subject_confusion.png` | 185582 | `69b5cd4fe826` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure4_final_bootstrap_stability.pdf` | 16256 | `2970688abf42` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure4_final_bootstrap_stability.png` | 115883 | `70b71f3e8f3b` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure5_final_error_profiles.pdf` | 29280 | `74c7a72a1979` | Final v6 manuscript figure. |
| `reports/v6_final_model/figures/figure5_final_error_profiles.png` | 210377 | `75e7982f84cc` | Final v6 manuscript figure. |

### frontend

| path | bytes | sha256 | role |
|---|---:|---|---|
| `frontend/app.js` | 16407 | `73c7337c3384` | Browser research console asset. |
| `frontend/index.html` | 9319 | `5e10d6e45ed5` | Browser research console asset. |
| `frontend/styles.css` | 12864 | `b5357a132d07` | Browser research console asset. |

### other

| path | bytes | sha256 | role |
|---|---:|---|---|
| `.gitignore` | 639 | `d71d37836bc0` | Public repository file. |
