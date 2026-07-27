# ARA-Net Clinical Presentation Evidence Pack

## What This Adds

- Lightweight deployment numbers for the RC-SPE probability head.
- Atlas-level AD-prior evidence validation for hippocampus, amygdala, and lateral ventricles.
- Explicit OASIS domain-shift explanation instead of hiding the weak external result.
- A UI-ready checklist for reviewer-facing clinical presentation without overclaiming clinical deployment.

## Atlas Evidence Validation

| Evidence | Value |
|---|---:|
| AIBL AD-key concentration score | 0.510 |
| Uniform regional null | 0.286 |
| Score minus null | 0.225 |
| Bootstrap 95% CI | 0.479-0.526 |
| Permutation p | 0.026 |

### AIBL Heldout Structural Direction Checks

| Feature | CN mean | MCI mean | AD mean | AD-CN | Cohen d | Matches AD prior |
|---|---:|---:|---:|---:|---:|---|
| Hippocampus volume | 0.008 | 0.008 | 0.007 | -0.001 | -1.233 | yes |
| Amygdala volume | 0.003 | 0.003 | 0.003 | -0.000 | -1.150 | yes |
| Lateral ventricle volume | 0.032 | 0.043 | 0.057 | 0.025 | 1.640 | yes |
| Cortex volume | 0.448 | 0.435 | 0.440 | -0.008 | -0.453 | yes |
| Atlas AD-like z | -0.058 | 0.605 | 1.250 | 1.309 | 1.800 | yes |

## OASIS Domain-Shift Finding

| OASIS metric | Value |
|---|---:|
| n | 99 |
| Accuracy | 0.586 |
| Balanced accuracy | 0.334 |
| Macro AUC | 0.554 |
| CN recall | 0.966 |
| MCI recall | 0.034 |
| AD recall | 0.000 |
| Balanced-accuracy drop vs AIBL heldout | -0.499 |
| MCI-recall drop vs AIBL heldout | -0.651 |
| AD-recall drop vs AIBL heldout | -0.852 |

### OASIS Prediction Collapse

| Predicted class | n | rate |
|---|---:|---:|
| CN | 94 | 0.949 |
| MCI | 4 | 0.040 |
| AD | 1 | 0.010 |

### Largest OASIS-vs-AIBL Structural Shifts

| Feature | AIBL mean | OASIS mean | OASIS-AIBL | SMD |
|---|---:|---:|---:|---:|
| Cortex volume | 0.445 | 0.132 | -0.313 | -3.565 |
| Amygdala volume | 0.003 | 0.001 | -0.003 | -3.491 |
| Hippocampus volume | 0.008 | 0.002 | -0.006 | -3.415 |
| Lateral ventricle volume | 0.037 | 0.021 | -0.016 | -0.864 |
| Atlas AD-like z | 0.213 | 0.059 | -0.154 | -0.191 |

## Manuscript-Ready Interpretation

ARA-Net's locked AIBL heldout result supports domain-adapted external subject-level staging, but OASIS remains a stress-test failure. The OASIS pattern is not a random accuracy drop: predictions collapse toward CN, preserving high CN recall while nearly eliminating MCI and AD recall. This is consistent with a domain-shift problem involving cohort/scanner/protocol differences, label-distribution differences, and incomplete clinical-feature harmonization. The correct claim is therefore not zero-shot generalization, but strong AIBL heldout performance with an explicitly documented OASIS limitation.

## Annotation Boundary

Current evidence supports atlas-level structural neurodegeneration consistency. It does not yet provide clinician-drawn lesion masks, CAGM-style clinical annotation overlap, entorhinal-cortex validation in the public enriched table, or temporal-lobe region masks. Those targets should be described as planned validation rather than completed evidence.

Unavailable annotation targets in the current public evidence table:
- clin_entorhinal
- temporal_lobe_annotation
- clinician_roi_annotation
