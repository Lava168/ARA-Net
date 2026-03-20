| Method | Year | Modality | N | Eval | Acc (%) | BAcc (%) | AUC | Note |
|--------|------|----------|---|------|---------|----------|-----|------|
| 3D-CNN | 2017 | sMRI (3D) | ADNI | 5-fold CV | 59.7 | — | — | Baseline 3D-CNN; 3-class |
| THAN | 2022 | sMRI (3D) | ADNI | 5-fold CV | 62.9 | — | 0.7 | Transformer-based hierarchical attention |
| STNet | 2023 | sMRI (3D) | ADNI | 5-fold CV | 71.8 | — | — | Spatial-temporal network; 3-class |
| LSTM-Robust | 2023 | sMRI (3D) | ADNI | 5-fold CV | 76.0 | — | — | Longitudinal LSTM; 3-class |
| ECAResNet269 + SMOTE/FL | 2025 | sMRI (2D coronal) | 1346 | Patient-split | — | 74.0 | — | SMOTE + focal loss; reports BAcc |
| Ensemble 138 ViT | 2024 | sMRI (ROI 3D) | ADNI | 5-fold CV | — | — | — | ROI-based 3D ViT ensemble; interpretable |
| 3D HCCT | 2024 | sMRI (3D) | ADNI | Single split | 96.1 | — | — | CNN+Transformer hybrid; no CV, likely data leakage |
| DEMNET | 2021 | sMRI (2D) | ADNI+Kaggle | Hold-out | 95.2 | — | — | Single split on Kaggle; no subject-level split |
|--------|------|----------|---|------|---------|----------|-----|------|
| **ARA-Net Ensemble** | 2025 | sMRI (3D) | 2,401 | 6s×5f CV | 69.1±2.2 | 69.9±2.0 | 0.862±0.012 | Ours |
| ARA-Net (Full) | 2025 | sMRI (3D) | 2,401 | 6s×5f CV | 66.7±2.4 | 67.1±2.4 | 0.830±0.016 | Ours |
| ARA-Net (−Atlas) | 2025 | sMRI (3D) | 2,401 | 6s×5f CV | 67.8±1.8 | 70.7±1.7 | 0.861±0.013 | Ours |