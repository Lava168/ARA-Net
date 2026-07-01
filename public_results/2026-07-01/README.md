# ARA-Net Public Results Package

Date: 2026-07-01

This folder contains public, aggregate-level result artifacts for the ARA-Net / RC-SPE study. It is intended to support manuscript review, benchmarking, and reproducibility without redistributing restricted ADNI, AIBL, OASIS, or other subject-level clinical/imaging data.

## Included

- `configs/`: final ensemble configuration used by the public inference package.
- `metrics/`: public JSON summaries for the locked final model and algorithm-innovation analysis.
- `tables/`: aggregate classification, calibration, ablation, risk-constraint, and confusion-transition tables.
- `figures/`: performance, calibration, ablation, bootstrap-stability, and RC-SPE architecture figures.
- `docs/`: data/model cards, public release manifest, clinical validation protocol, and deployment notes.

## Excluded

The following were intentionally not included:

- Raw MRI, FreeSurfer/FastSurfer derivatives, voxel-level arrays, or atlas feature matrices from restricted cohorts.
- Subject-level prediction tables and top-confident-error lists.
- Any file containing ADNI/AIBL/OASIS subject IDs, scan IDs, image IDs, or individual-level clinical records.

Use the scripts and documentation in the repository to regenerate restricted artifacts inside an approved data-use environment.
