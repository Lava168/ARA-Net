# Manuscript Overview

**ARA-Net: Atlas-Guided Multimodal Alzheimer's Disease Staging with Locked External Subject-Level Validation and Structural Neurodegeneration Consistency**

This overview is aligned with `manuscript.docx` from the final ARAnet manuscript folder. It summarizes the public, claim-bounded GitHub package and mirrors the README-facing evidence.

## Core Contribution

ARA-Net stages structural MRI cases into CN, MCI, and AD by combining atlas-guided MRI features with clinical variables and a lightweight probability ensemble head. The model emphasizes subject-level external validation and structural neurodegeneration consistency rather than unrestricted clinical deployment.

## Method Summary

1. Structural MRI scans are mapped to an atlas of 21 anatomical brain regions.
2. Regional features are extracted, including volume, mean intensity, standard deviation, percentile features, bilateral asymmetry, and AD-key aggregated features.
3. Core clinical variables are integrated: age, sex, education, APOE4, MMSE, and CDR-SB.
4. Six heterogeneous probability streams produce CN/MCI/AD probabilities.
5. RC-SPE combines those streams by non-negative weighted log-probability pooling, class offsets, temperature scaling, and subject-level averaging.
6. Final outputs include calibrated CN/MCI/AD probabilities, predicted stage, confidence, and decision margin.

## Locked External Results

| Setting | Unit | Accuracy | Balanced accuracy | Macro AUC | AD-vs-CN AUC / CN retention | CN / MCI / AD recall |
|---|---:|---:|---:|---:|---:|---:|
| AIBL locked external test | Subject | 90.3% | 83.3% | 93.7% | AD-vs-CN AUC 100.0% | 96.1% / 68.6% / 85.2% |
| AIBL locked external test | Scan | 90.9% | 82.0% | 93.9% | AD-vs-CN AUC 99.8% | 96.4% / 64.2% / 85.4% |
| IXI healthy control | Subject | 100.0% | 100.0% | NA | CN retention 100.0% | 100.0% / 0.0% / 0.0% |

## RC-SPE Evidence

Full RC-SPE achieved 0.903 accuracy, 0.833 balanced accuracy, 0.078 ECE, 0.320 NLL, and 0.160 Brier score on the locked AIBL subject-level endpoint. It improved balanced accuracy over the best single base model and equal log-pooling while retaining IXI healthy controls as CN.

## Structural Consistency

AD-key structural evidence was concentrated in the hippocampus, amygdala, and lateral ventricles. The AD-key enrichment score on locked AIBL was 0.510 versus a uniform-null expectation of 0.286 (`p = 0.026`; bootstrap 95% CI 0.479-0.526). Multi-cohort directional validation was strongest in ADNI and AIBL, while OASIS showed a domain-shift stress-test limitation.

## Manuscript Figures

The README-ready figure set is stored at `assets/manuscript_figures/` and includes Figure 1 through Figure 10 from the final manuscript figure directory.

## Claim Boundary

The repository is a research prototype. It does not provide a clinical diagnostic device, does not redistribute restricted MRI data, and does not include private training checkpoints. The public deployment wrapper operates on already-produced base-model probability streams.
