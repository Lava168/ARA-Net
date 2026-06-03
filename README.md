# ARA-Net

Atlas-guided multimodal Alzheimer's disease staging research code.

This repository contains the reproducible analysis scripts, research deployment wrapper, and manuscript-supporting reports for the revised ARA-Net project. The current public package is framed as an **open-source deployable research prototype**, not as a clinical diagnostic device.

## Current V6 Result

The locked main model is a subject-level balanced rescue probability ensemble. It was tuned using ADNI validation, AIBL adaptation validation, and IXI healthy specificity. OASIS was not used for final tuning and is reported only as an external stress-test limitation.

Main locked AIBL heldout subject-level result:

| Metric | Value |
|---|---:|
| Accuracy | 0.903 |
| Balanced accuracy | 0.833 |
| Macro AUC | 0.937 |
| AD-vs-CN AUC | 1.000 |
| Recall CN/MCI/AD | 0.961 / 0.686 / 0.852 |
| IXI healthy CN retention | 1.000 |

Bootstrap 95% confidence intervals for the locked AIBL heldout subject-level result:

| Metric | 95% CI |
|---|---:|
| Balanced accuracy | 0.759-0.899 |
| MCI recall | 0.531-0.839 |
| AD recall | 0.710-0.966 |

## Repository Contents

- `scripts/rescue_probability_optimizer.py`: probability ensemble, calibration, and subject-level averaging.
- `scripts/final_rescue_model_package.py`: final metrics, bootstrap, and error-analysis package generation.
- `scripts/generate_v6_final_figures.py`: final v6 manuscript figures.
- `scripts/generate_core_reviewer_evidence_matrix.py`: reproducible evidence matrix for external validation, CAS replacement, and Braak-alternative biological validation.
- `deployment/research_inference.py`: CLI for research inference from base-model class probabilities.
- `deployment/research_api.py`: HTTP API and static web-console server for research deployment.
- `deployment/final_ensemble_config.json`: locked final ensemble weights, offsets, and temperature.
- `frontend/`: browser-based research console for CSV upload, prediction review, and CSV export.
- `docs/MODEL_CARD.md`: model-card summary, intended use, metrics, and limitations.
- `docs/DATA_CARD.md`: data provenance and public-release boundary.
- `docs/CLINICAL_VALIDATION_PROTOCOL.md`: prospective validation protocol draft.
- `reports/v6_final_model/`: public manuscript-supporting reports, aggregate tables, and figures.
- `reports/v6_final_model/core_reviewer_evidence_matrix.md`: generated reviewer-evidence matrix for the three core revision issues.
- `reports/v6_final_model/public_release_manifest.md`: generated manifest of public tracked files and restricted-artifact checks.

The public reports intentionally exclude row-level subject/scan prediction files and dataset-derived identifiers.

## Data Availability

Raw ADNI, AIBL, OASIS, and IXI data are governed by their original data-use agreements and are not redistributed in this repository. Users must obtain access from the respective data providers. Public files in this repository are limited to code, aggregate reports, figures, and de-identified/manuscript-level summaries.

## Clinical-Use Boundary

This software is not intended for direct clinical deployment or standalone diagnosis. It is a research-grade, retrospective validation pipeline intended to support further prospective evaluation. Clinical use would require prospective multi-center validation, scanner/protocol robustness testing, local calibration, workflow integration, uncertainty reporting, cybersecurity review, and regulatory assessment.

## Research Deployment

The public deployment wrapper combines already-produced base-model CN/MCI/AD probabilities using the locked final ensemble configuration. It does not process raw MRI files and does not redistribute restricted model artifacts or datasets.

CLI example:

```bash
python deployment/research_inference.py \
  --input-csv examples/probability_input_example.csv \
  --output examples/predictions_subject.csv \
  --unit subject
```

API example:

```bash
python deployment/research_api.py --port 8080
curl http://localhost:8080/health
```

Web console:

```bash
python deployment/research_api.py --host 127.0.0.1 --port 8080
```

Then open [http://localhost:8080](http://localhost:8080). The console accepts the same base-model probability CSV format as the CLI, calls `POST /predict`, summarizes class distribution and confidence, and exports prediction CSV files.

Docker example:

```bash
docker build -t aranet-research .
docker run --rm -p 8080:8080 aranet-research
```

## Reproducing The V6 Reports

After generating prediction CSV files with the training/evaluation scripts, run:

```bash
python scripts/final_rescue_model_package.py \
  --subject-pred-dir outputs/v4/rescue_probability_subject_quick_no_oasis_tune \
  --scan-pred-dir outputs/v4/rescue_probability_no_oasis_tune \
  --subject-summary outputs/v4/rescue_probability_subject_quick_no_oasis_tune/summary.json \
  --scan-summary outputs/v4/rescue_probability_no_oasis_tune/summary.json \
  --feature-csv outputs/v4/atlas_feature_cache_v4.csv \
  --adni-clinical /path/to/adni/master_subjects_v2.csv \
  --aibl-clinical /path/to/aibl/aibl_adnimergelike.csv \
  --out-dir reports/v6_final_model \
  --n-bootstrap 2000
```

Then regenerate figures:

```bash
python scripts/generate_v6_final_figures.py \
  --summary reports/v6_final_model/final_rescue_model_summary_public.json \
  --table2 reports/v4/tables/table2_classification.csv \
  --table-dir reports/v6_final_model/tables \
  --out-dir reports/v6_final_model/figures
```

Generate the core reviewer-evidence matrix:

```bash
python scripts/generate_core_reviewer_evidence_matrix.py
```

Generate the public release manifest:

```bash
python scripts/generate_public_release_manifest.py
```

## Python Dependencies

Core analysis dependencies are listed in `requirements.txt`. The lightweight deployment wrapper uses `requirements-deploy.txt`. Some legacy training scripts additionally require PyTorch and scikit-learn. Figure generation requires Matplotlib.

## Documentation

- [Open-source and deployment plan](docs/OPEN_SOURCE_AND_DEPLOYMENT.md)
- [Model card](docs/MODEL_CARD.md)
- [Data card](docs/DATA_CARD.md)
- [Clinical validation protocol draft](docs/CLINICAL_VALIDATION_PROTOCOL.md)
- [Regulatory notes](docs/REGULATORY_NOTES.md)
