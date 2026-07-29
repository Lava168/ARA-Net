# Data Card

## Data Sources

The final ARA-Net manuscript uses structural MRI and clinical data from ADNI, AIBL, OASIS, and IXI. The manuscript reports 4,388 structural MRI scans corresponding to 2,023 subjects according to the data-partition manifest. Raw MRI volumes and restricted clinical data are not redistributed in this repository.

## Public Repository Data

This repository includes code, aggregate metrics, manuscript-level summaries, generated figures, model/data cards, and de-identified examples. It intentionally excludes:

- raw MRI files
- source clinical spreadsheets
- subject-level prediction files from restricted cohorts
- scan-level prediction files from restricted cohorts
- restricted subject IDs or scan IDs
- private training checkpoints

## Cohort Roles

| Cohort | Role in manuscript-facing package |
|---|---|
| ADNI | Development, validation, internal testing, and directional structural checks |
| AIBL | External-domain adaptation/validation and locked external heldout testing |
| IXI | Healthy-control negative-control specificity testing |
| OASIS | External stress test for domain-shift and applicability-boundary analysis |

## Split Summary From Public Documentation

| split | scans | subjects | role |
|---|---:|---:|---|
| ADNI train | 1,686 | 450 | training |
| ADNI validation | 355 | 97 | model selection |
| ADNI internal test | 360 | 96 | internal test |
| AIBL adaptation train | 719 | 385 | external-domain adaptation |
| AIBL adaptation validation | 191 | 105 | external-domain calibration |
| AIBL locked heldout | 397 | 210 | primary external test |
| OASIS external | 99 | 99 | stress test |
| IXI external | 581 | 581 | healthy negative-control test |

Subject counts above are split-inventory counts. Manuscript metric tables may report evaluable subject-level endpoint units after repeated-scan probability aggregation.

## Access Requirements

Users must obtain raw data access through the original data providers and comply with all applicable data-use agreements.

## Reproducibility Boundary

The public code can reproduce the final RC-SPE probability-fusion logic, deployment wrapper behavior, and aggregate reporting when supplied with authorized probability and feature files. It cannot reproduce raw-data preprocessing without access to the underlying datasets.
