# Atlas-guided AD staging — submission layout checklist

This repository implements the MedIA / open-source submission layout commonly
referred to as **`atlas-guided-ad-staging/`**. The GitHub project name remains
[`Lava168/ARA-Net`](https://github.com/Lava168/ARA-Net); the on-disk package
layout matches the checklist below.

Aligned manuscript: *ARA-Net: Atlas-Guided Multimodal Alzheimer's Disease Staging
with Locked External Subject-Level Validation and Structural Neurodegeneration
Consistency* (`manuscript(2).docx` / V6 draft).

## Required tree

```text
atlas-guided-ad-staging/   (= this repository root)
├── README.md
├── LICENSE
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── adni_development.yaml
│   ├── aibl_adaptation.yaml
│   ├── aibl_heldout.yaml
│   ├── ixi_specificity.yaml
│   └── oasis_stress_test.yaml
├── src/
│   ├── data/
│   ├── atlas/
│   ├── models/
│   ├── fusion/
│   ├── calibration/
│   ├── constraints/
│   ├── aggregation/
│   ├── evaluation/
│   └── interpretation/
├── scripts/
│   ├── prepare_features.py
│   ├── train_base_models.py
│   ├── fit_rc_spe.py
│   ├── evaluate_aibl.py
│   ├── evaluate_ixi.py
│   ├── evaluate_oasis.py
│   ├── reproduce_ablation.py
│   └── reproduce_figures.py
├── data/
│   ├── README.md
│   ├── example_metadata.csv
│   └── synthetic/
├── tests/
│   ├── test_fusion.py
│   ├── test_subject_aggregation.py
│   └── test_metrics.py
└── outputs/
    └── expected_results/
```

## Smoke reproduction

```bash
python scripts/prepare_features.py
python scripts/train_base_models.py
python scripts/fit_rc_spe.py
python scripts/evaluate_aibl.py
python scripts/evaluate_ixi.py
python scripts/evaluate_oasis.py
python scripts/reproduce_ablation.py
python scripts/reproduce_figures.py
pytest -q
```

## Boundary

Public scripts operate on **probability streams** and **synthetic / aggregate**
atlas tables. Restricted ADNI/AIBL/OASIS/IXI MRI volumes and row-level subject
tables are not redistributed.
