# V6 Final Model Public Evidence Package

This folder contains the public, aggregate evidence package for the revised ARA-Net work.

## Main Evidence Files

- `final_rescue_model_lock_report.md`: locked final model, subject-level endpoint, bootstrap intervals, and claim boundary.
- `../v6_algorithm_innovation/algorithm_innovation_evidence.md`: RC-SPE ablation, calibration, risk-profile, and leave-one-model-out evidence.
- `core_reviewer_evidence_matrix.md`: generated matrix for the three core revision issues: external classification, CAS replacement, and Braak-alternative biological validation.
- `goal_completion_audit.md`: requirement-level audit for the full V6 rebuild objective, including bounded limitations.
- `final_figure_blueprint.md`: planned main and supplementary figure set with panel-by-panel content.
- `manuscript_v6_full_draft.md`: full V6 manuscript draft for replacing the old Word manuscript body.
- `ARA-Net_V6_full_manuscript_draft.docx`: generated V6 Word manuscript replacement draft.
- `word_manuscript_claim_audit_v6_docx.md`: Word claim-boundary audit for the generated V6 DOCX.
- `ARA-Net_V6_full_manuscript_docx_qa.md`: structural DOCX QA report and render-status note.
- `final_submission_closure_packet.md`: final manuscript-integration packet for Figure 1, OASIS handling, citations, and terminology.
- `word_manuscript_claim_audit.md`: generated claim-boundary audit for the uploaded Word manuscript.
- `word_manuscript_rewrite_map.md`: section-by-section rewrite map for converting the old Word manuscript into the V6 submission.
- `claim_boundary_audit.md`: generated public-file audit for unsupported Braak/CAS/OASIS/zero-shot/clinical-deployment overclaims.
- `final_model_error_analysis.md`: aggregate error-analysis summaries for AIBL heldout and internal test.
- `manuscript_v6_rewrite_package.md`: manuscript-ready Methods/Results/Discussion material.
- `reviewer_response_v6_matrix.md`: response-oriented matrix for reviewer/editor concerns.
- `public_release_manifest.md`: generated manifest of public tracked files and restricted-artifact checks.
- `tables/lightweight_runtime_metrics.md`: reproducible RC-SPE parameter-size and CPU inference benchmark.
- `tables/clinical_presentation_evidence.md`: reviewer-facing presentation evidence pack covering atlas-region validation, OASIS domain-shift interpretation, and annotation claim boundaries.
- `rcspe_lightweight_inference_insert.md`: manuscript-ready lightweight inference paragraph and short version.
- `ui_research_workbench_insert.md`: manuscript-ready research-workbench figure caption and methods/discussion language.
- `manual_paper_figures/`: manually edited manuscript figures extracted, renamed, and documented for the English GitHub release.

## Figures

The authoritative final main-figure sequence is specified in `final_figure_blueprint.md`:

1. ARA-Net study framework and evidence chain.
2. Atlas-guided multimodal feature system.
3. RC-SPE risk-constrained probability ensemble.
4. Locked external classification performance and model comparison.
5. Subject-level error structure and disease-boundary behavior.
6. Algorithmic ablation, calibration, risk tradeoff, and stability.
7. Atlas structural neurodegeneration consistency and claim boundary.

Existing generated result panels that can be reused under the new numbering:

- `manual_paper_figures/figure1_atlas_guided_staging_overview.png` -> manuscript-aligned Figure 1 overview.
- `manual_paper_figures/figure2_locked_external_performance.png` -> manuscript-aligned locked external performance figure.
- `manual_paper_figures/figure3_subject_level_error_structure.png` -> manuscript-aligned subject-level error and atlas evidence figure.
- `manual_paper_figures/figure4_research_workbench_ui.png` -> manuscript-aligned research UI figure.
- `manual_paper_figures/figure5_atlas_feature_evidence_panel.png` -> manuscript-aligned atlas evidence panel.
- `manual_paper_figures/figure6_end_to_end_workflow.png` -> manuscript-aligned workflow figure.
- `manual_paper_figures/figure7_rcspe_probability_ensemble_ui.png` -> manuscript-aligned RC-SPE probability ensemble figure.
- `figures/figure2_final_external_rescue.png/pdf` -> evidence for new Figure 4.
- `figures/figure3_final_subject_confusion.png/pdf` -> evidence for new Figure 5.
- `figures/figure4_final_bootstrap_stability.png/pdf` -> evidence for new Figure 6.
- `figures/figure5_final_error_profiles.png/pdf` -> evidence for new Figure 5.
- `figures/figure3_rcspe_architecture_nbe_style.png/pdf` -> evidence for new Figure 3.

## Reproducibility

Regenerate the core reviewer matrix:

```bash
python scripts/generate_core_reviewer_evidence_matrix.py
```

Regenerate the requirement-level goal audit:

```bash
python scripts/generate_goal_completion_audit.py
```

Regenerate the RC-SPE algorithmic evidence package on a machine with access to the private prediction CSV files:

```bash
python scripts/generate_algorithm_innovation_evidence.py \
  --pred-root outputs/v4 \
  --out-dir reports/v6_algorithm_innovation
```

Run the claim-boundary audit:

```bash
python scripts/audit_claim_boundaries.py
```

Regenerate the public release manifest:

```bash
python scripts/generate_public_release_manifest.py
```

Build and audit the generated V6 DOCX:

```bash
/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/build_v6_manuscript_docx.py

/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/audit_word_manuscript_claims.py \
  --docx reports/v6_final_model/ARA-Net_V6_full_manuscript_draft.docx \
  --output reports/v6_final_model/word_manuscript_claim_audit_v6_docx.md
```

Run the public deployment smoke test:

```bash
python deployment/research_inference.py \
  --input-csv examples/probability_input_example.csv \
  --output examples/predictions_subject.csv \
  --unit subject
```

Regenerate the public lightweight benchmark:

```bash
python scripts/measure_lightweight_runtime.py
```

Regenerate the reviewer-facing presentation evidence additions on a machine with the private enriched prediction table and source aggregate reports:

```bash
python scripts/build_clinical_presentation_evidence.py
```

## Data Boundary

Raw ADNI, AIBL, OASIS, and IXI data are not redistributed. Row-level subject/scan predictions, private clinical spreadsheets, MRI volumes, and model checkpoints are excluded from the public package. Public files are limited to code, aggregate reports, final figures, documentation, and toy probability examples.
