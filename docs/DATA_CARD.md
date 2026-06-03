# Data Card

## Data Sources

The revised ARA-Net analyses use ADNI, AIBL, OASIS, and IXI data. Raw data are not redistributed in this repository.

## Public Repository Data

This repository includes aggregate metrics, figures, and manuscript-level summaries. It intentionally excludes:

- raw MRI files
- clinical spreadsheets from source datasets
- subject-level prediction files
- scan-level prediction files
- subject IDs and scan IDs from restricted datasets

## Split Summary

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

These subject counts are unique-participant split-inventory counts. Final metric tables may report evaluable subject-level endpoint units after repeated-scan probability aggregation; in longitudinal cohorts, a participant can contribute distinct diagnostic-state endpoint units over time.

## Access Requirements

Users must obtain raw data access through the original data providers and comply with all applicable data-use agreements.

## Reproducibility Boundary

The public code can reproduce the final ensemble logic, figures, and aggregate reports when supplied with authorized prediction and feature files. It cannot reproduce raw-data preprocessing without access to the underlying datasets.
