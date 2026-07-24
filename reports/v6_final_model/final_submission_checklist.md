# Final Submission Checklist

## Locked Main Model

- [x] Main model selected: subject-level balanced rescue probability ensemble.
- [x] OASIS excluded from tuning.
- [x] Primary endpoint selected: AIBL locked heldout subject-level staging.
- [x] Specificity endpoint selected: IXI healthy CN retention.
- [x] Scan-level reference retained as secondary analysis.

## Stability And Error Analysis

- [x] Bootstrap stability computed with 2,000 resamples.
- [x] AIBL heldout BAcc 95% CI: 0.759-0.899.
- [x] AIBL heldout MCI recall 95% CI: 0.531-0.839.
- [x] AIBL heldout AD recall 95% CI: 0.710-0.966.
- [x] AIBL heldout subject-level confusion matrix generated.
- [x] Internal subject-level confusion matrix generated.
- [x] AIBL MCI/AD error-profile table generated.
- [x] Core reviewer-evidence matrix generated for external classification, CAS replacement, and Braak-alternative validation.
- [x] Requirement-level goal completion audit generated for the full V6 rebuild objective.
- [x] Claim-boundary audit generated for unsupported Braak/CAS/OASIS/zero-shot/clinical-deployment overclaims.

## Manuscript Rewrite

- [x] Abstract rewritten around final subject-level model.
- [x] Introduction rewritten around external validation, invalid CAS, and non-significant Braak issue.
- [x] Methods updated with subject-level splits, ensemble formula, subject averaging, bootstrap, error analysis.
- [x] Results updated with final main model, comparator models, error analysis, biological consistency, and OASIS limitation.
- [x] Discussion rewritten with honest claim boundary.
- [x] Full V6 manuscript draft generated for replacing the old Word manuscript body.
- [x] Generated V6 DOCX manuscript replacement draft.
- [x] Generated V6 DOCX passed Word claim-boundary audit with 0 blockers and 0 warnings.
- [x] Generated V6 DOCX passed structural QA and visual render QA with rendered page PNG/PDF inspection.
- [x] Cover letter core paragraph drafted.

## Figures

- [x] Figure 2 final external rescue generated: `figures/figure2_final_external_rescue.png/pdf`.
- [x] Figure 1 final workflow blueprint/caption specified in `final_figure_blueprint.md`.
- [x] Figure 3 final subject confusion generated: `figures/figure3_final_subject_confusion.png/pdf`.
- [x] Figure 4 bootstrap stability generated: `figures/figure4_final_bootstrap_stability.png/pdf`.
- [x] Figure 5 error profiles generated: `figures/figure5_final_error_profiles.png/pdf`.
- [x] Update Figure 1 workflow caption/text so it names the final subject-level rescue ensemble, not only v4 HGB.
- [x] Decide whether old OASIS figure stays in supplement or is replaced by a brief limitation table.

## Formula And Methods

- [x] Log-probability ensemble formula written.
- [x] Temperature and class-offset calibration written.
- [x] Subject-level probability averaging formula written.
- [x] Bootstrap evaluation described.
- [x] CAS replacement described as atlas-region structural neurodegeneration consistency.
- [x] Apply final closure-packet replacement rules in the generated V6 Word manuscript: remove old CAS-as-attention-biomarker equations and claims.
- [x] Apply final closure-packet replacement rules in the generated V6 Word manuscript: remove direct Braak-stage validation wording.
- [x] Word manuscript claim audit generated for old CAS/Braak/OASIS/clinical-deployment language.
- [x] Word manuscript claim audit generated for the V6 DOCX replacement draft.
- [x] Word manuscript section-level rewrite map generated.

## Code / Data Availability

- [x] Final scripts present:
  - `scripts/rescue_probability_optimizer.py`
  - `scripts/final_rescue_model_package.py`
  - `scripts/generate_v6_final_figures.py`
- [x] Final reports present in `reports/v6_final_model/`.
- [x] Core reviewer matrix script present: `scripts/generate_core_reviewer_evidence_matrix.py`.
- [x] Goal completion audit script present: `scripts/generate_goal_completion_audit.py`.
- [x] Claim-boundary audit script present: `scripts/audit_claim_boundaries.py`.
- [x] Public release manifest script present: `scripts/generate_public_release_manifest.py`.
- [x] GitHub release evidence folder prepared with README, public manifest, aggregate reports, and figures.
- [x] Add statement that raw ADNI/AIBL/OASIS/IXI data are governed by their source-access agreements.
- [x] Add statement that derived manifests, split definitions, model code, and analysis scripts will be released.

## Citation / Reference Cleanup

- [x] Verify and add Pfeifer 2025 Tree Smoothing citation requested by reviewer.
- [x] Verify and add Retzlaff 2024 post-hoc vs ante-hoc explanation citation requested by reviewer.
- [x] Fix broken formula and figure references in the public V6 rewrite package.
- [x] Shorten figure titles in the final V6 figure set.
- [x] Standardize terminology: atlas-guided, multimodal, subject-level, heldout, stress test.

## Remaining Scientific Risk

- The old uploaded Word manuscript remains a failed historical draft; use the generated V6 DOCX replacement draft for resubmission work.
- Visual DOCX render QA was completed using rendered page PNG/PDF inspection; rerun only if the Word manuscript text or layout is changed again.
- OASIS remains weak and must not be sold as solved.
- Internal subject-level BAcc remains modest; write it as calibration limitation, not primary failure.
- Clinical-only comparator is strong; use it as a transparent upper bound.
- AIBL validation is domain-adapted external heldout, not pure zero-shot.
