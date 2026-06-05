# V6 LaTeX Seven-Figure Plan

This plan is for the LaTeX replacement manuscript. The author block, affiliations, funding statement, and competing-interest statement should remain unchanged from the original LaTeX file. The seven main figures should emphasize the final model objective: domain-adapted external subject-level CN/MCI/AD staging with bounded biological validation.

## Figure 1. Revised Study Framework And Central Model Objective

Purpose: Make the paper visibly different from the old v3 attention-only manuscript.

Panels:
- A: Old v3 failure points: weak AIBL external classification, poor IXI specificity, invalid attention-only CAS, non-significant Braak analysis.
- B: V6 objective: subject-level atlas-guided multimodal AD staging.
- C: Final model path: atlas MRI features + core clinical variables -> base models -> RC-SPE -> subject-level probability averaging.
- D: Endpoint hierarchy: AIBL heldout primary, IXI specificity, OASIS stress-test limitation.
- E: Claim boundary: structural MRI proxy only; not attention biomarker, direct Braak validation, or clinical device.

## Figure 2. Data Line And Endpoint Design

Purpose: Show that the data work is a real rebuild.

Panels:
- A: ADNI train/validation/internal test inventory.
- B: AIBL adaptation train/validation and locked heldout inventory.
- C: IXI healthy negative-control cohort.
- D: OASIS stress-test cohort.
- E: Scan-to-subject endpoint aggregation and leakage control.

## Figure 3. RC-SPE Locked Subject-Level Probability Ensemble

Purpose: Make the core model clear and three-class, not binary.

Panels:
- A: Six base-model probability streams.
- B: RC-SPE risk-constrained objective balancing AIBL BAcc, MCI/AD recall, AD-to-CN avoidance, calibration, and IXI healthy specificity.
- C: Log-probability pooling weights, class offsets, and temperature scaling.
- D: Subject-level probability averaging across repeated scans.
- E: CN/MCI/AD output with confidence and margin.

## Figure 4. External Classification Performance

Purpose: Directly answer the external classification criticism.

Panels:
- A: AIBL BAcc comparison: old v3, v4 atlas+clinical, final scan-level, final subject-level, clinical-only comparator.
- B: AIBL macro AUC comparison.
- C: Final AIBL CN/MCI/AD recall.
- D: IXI healthy CN retention and false impairment rate.
- E: OASIS stress-test mini-panel showing weak transfer as a limitation.

## Figure 5. Subject-Level Confusion And Error Profile

Purpose: Show that remaining errors are boundary-like.

Panels:
- A: AIBL heldout subject-level confusion matrix.
- B: Internal subject-level confusion matrix.
- C: AIBL true-to-predicted transition flow.
- D: Error subtype bars: MCI-to-CN, MCI-to-AD, AD-to-MCI, AD-to-CN.
- E: Error feature profiles: MMSE, hippocampus volume, ventricle volume, AD-like z-score, max probability, margin.

## Figure 6. Algorithmic Ablation, Calibration, Risk Tradeoff, And Stability

Purpose: Show that RC-SPE is not a generic average ensemble and that the main result is stable enough to lock.

Panels:
- A: Algorithmic ablation bars comparing best single model, arithmetic mean, equal log-pooling, partial RC-SPE, scan-level RC-SPE, and subject-level RC-SPE.
- B: Calibration reliability panel for best single model, equal log-pooling, and final RC-SPE.
- C: Risk-constraint scatterplot showing AIBL BAcc or MCI recall versus IXI false impairment rate, with the locked RC-SPE profile highlighted.
- D: Leave-one-model-out BAcc stability and AD-to-CN error preservation.
- E: Bootstrap BAcc/MCI/AD recall forest plot with 95% CIs.

## Figure 7. CAS/Braak Replacement And Biological Consistency

Purpose: Solve the CAS/Braak reviewer criticism without overclaiming.

Panels:
- A: Old attention-only CAS failure callout.
- B: A priori AD-key atlas regions: bilateral hippocampus, amygdala, lateral ventricles.
- C: AIBL heldout score 0.510 vs uniform null 0.286, CI 0.479-0.526, p=0.026.
- D: All-labeled AD score 0.426 vs null 0.286; ADNI-only non-significant check.
- E: Claim-boundary badges: MRI proxy only; not attention-map biomarker; not direct Braak-stage proof; not clinical deployment.

## What To Avoid

- Do not present OASIS as solved.
- Do not make clinical-only RF look like the central ARA-Net model.
- Do not describe RC-SPE as a new end-to-end neural architecture.
- Do not draw attention maps as validated biomarkers.
- Do not claim direct Braak-stage validation.
- Do not imply clinical deployment readiness.
