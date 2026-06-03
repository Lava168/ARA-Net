# V6 Final Model Public Evidence Package

This folder contains the public, aggregate evidence package for the revised ARA-Net work.

## Main Evidence Files

- `final_rescue_model_lock_report.md`: locked final model, subject-level endpoint, bootstrap intervals, and claim boundary.
- `core_reviewer_evidence_matrix.md`: generated matrix for the three core revision issues: external classification, CAS replacement, and Braak-alternative biological validation.
- `final_figure_blueprint.md`: planned main and supplementary figure set with panel-by-panel content.
- `final_submission_closure_packet.md`: final manuscript-integration packet for Figure 1, OASIS handling, citations, and terminology.
- `claim_boundary_audit.md`: generated public-file audit for unsupported Braak/CAS/OASIS/zero-shot/clinical-deployment overclaims.
- `final_model_error_analysis.md`: aggregate error-analysis summaries for AIBL heldout and internal test.
- `manuscript_v6_rewrite_package.md`: manuscript-ready Methods/Results/Discussion material.
- `reviewer_response_v6_matrix.md`: response-oriented matrix for reviewer/editor concerns.
- `public_release_manifest.md`: generated manifest of public tracked files and restricted-artifact checks.

## Figures

- `figures/figure2_final_external_rescue.png/pdf`
- `figures/figure3_final_subject_confusion.png/pdf`
- `figures/figure4_final_bootstrap_stability.png/pdf`
- `figures/figure5_final_error_profiles.png/pdf`

## Reproducibility

Regenerate the core reviewer matrix:

```bash
python scripts/generate_core_reviewer_evidence_matrix.py
```

Run the claim-boundary audit:

```bash
python scripts/audit_claim_boundaries.py
```

Regenerate the public release manifest:

```bash
python scripts/generate_public_release_manifest.py
```

Run the public deployment smoke test:

```bash
python deployment/research_inference.py \
  --input-csv examples/probability_input_example.csv \
  --output examples/predictions_subject.csv \
  --unit subject
```

## Data Boundary

Raw ADNI, AIBL, OASIS, and IXI data are not redistributed. Row-level subject/scan predictions, private clinical spreadsheets, MRI volumes, and model checkpoints are excluded from the public package. Public files are limited to code, aggregate reports, final figures, documentation, and toy probability examples.
