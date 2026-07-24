# ARA-Net Manuscript Overview

This page summarizes the English GitHub-facing content aligned to the manually edited ARA-Net manuscript package. It is intended to make the repository understandable without exposing private row-level subject data, raw MRI volumes, or restricted cohort files.

## Title

ARA-Net: Atlas-Guided Multimodal Alzheimer's Disease Staging with Locked External Subject-Level Validation and Structural Neurodegeneration Consistency

## Study Objective

ARA-Net addresses three-class Alzheimer's disease staging across cognitively normal controls, mild cognitive impairment, and Alzheimer's disease. The central goal is not only high retrospective classification accuracy, but also a clinically interpretable evidence chain: locked external validation, subject-level aggregation for repeated scans, lightweight inference, and atlas-level structural plausibility.

## Model Summary

The locked model is RC-SPE, a risk-constrained subject-level probability ensemble. It combines six heterogeneous probability streams using log-probability pooling, non-negative model weights, class-specific offsets, temperature scaling, and subject-level probability averaging.

The model is deliberately deployed as a probability-level research wrapper. The public package does not redistribute private MRI preprocessing outputs, trained checkpoints, or row-level cohort predictions.

## Data And Evaluation

The manuscript framework covers 4,388 MRI scans from 2,023 subjects across ADNI, AIBL, OASIS, and IXI. The final locked external emphasis is AIBL heldout subject-level validation, with IXI used as a healthy-control specificity stress test and OASIS reported as a domain-shift limitation.

| Dataset / setting | Unit | Accuracy | Balanced accuracy | Macro AUC | AD-vs-CN AUC / CN retention | CN / MCI / AD recall |
|---|---:|---:|---:|---:|---:|---:|
| AIBL locked external test | Subject | 90.3% | 83.3% | 93.7% | AD-vs-CN AUC 100.0% | 96.1% / 68.6% / 85.2% |
| AIBL locked external test | Scan | 90.9% | 82.0% | 93.9% | AD-vs-CN AUC 99.8% | 96.4% / 64.2% / 85.4% |
| IXI healthy controls | Subject | 100.0% | 100.0% | NA | CN retention 100.0% | 100.0% / 0.0% / 0.0% |

## Lightweight Inference

The RC-SPE head contains only 10 scalar parameters and is stored as a 1,663 byte JSON configuration. The manuscript reports sub-millisecond CPU inference, which supports the claim that the public wrapper is lightweight once base-model probabilities have already been produced.

## Atlas Evidence

The model's structural interpretation is evaluated through atlas-level evidence rather than unsupported lesion-level clinical claims. AD-key regions include hippocampal, amygdalar, ventricular, medial temporal, and related neurodegeneration-sensitive structures. The public evidence reports an AD-key enrichment of 0.510 compared with a uniform null expectation of 0.286, with bootstrap 95% CI 0.479 to 0.526 and p = 0.026.

## Subject-Level Aggregation And Error Structure

AIBL locked external testing contains 397 scans that aggregate into 216 subject-level endpoints. Subject-level probability averaging reduces repeated-scan volatility and evaluates a more clinically natural endpoint.

The main residual error pattern is concentrated around the MCI/AD boundary:

| Transition | Rate |
|---|---:|
| CN retained as CN | 0.961 |
| MCI retained as MCI | 0.686 |
| AD retained as AD | 0.852 |
| MCI predicted as AD | 0.257 |
| AD predicted as MCI | 0.148 |
| AD predicted as CN | 0.000 |

This supports a bounded interpretation: ARA-Net's remaining errors are mainly adjacent-stage errors rather than systematic collapse of AD cases into healthy controls.

## UI And Deployment Boundary

The repository includes a browser research interface for GitHub presentation. It shows an upload-style workflow, probability output, aggregate evidence, representative brain-region evidence, and high-resolution PyVista/VTK and Nilearn visual assets. The public UI is a research demonstration and does not process restricted raw MRI data into a diagnosis.

## Public Figures

The manuscript-aligned figure set is available in [manual_paper_figures](../reports/v6_final_model/manual_paper_figures/README.md). The result UI is available locally at `frontend/v6-final-analysis.html` or through the static web server used during development.
