# Rescue Probability Optimizer

## balanced

- score: 0.7106
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.167, 0.424, 0.195, 0.057, 0.154, 0.002
- offsets CN/MCI/AD: -0.515, -0.131, 0.646
- temperature: 1.227

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.555 | 0.576 | 0.753 | 0.928 | 0.000 | 0.400/0.608/0.719 | {'CN': 63, 'MCI': 174, 'AD': 118} |
| internal_test | 0.519 | 0.496 | 0.745 | 0.935 | 0.000 | 0.351/0.637/0.500 | {'CN': 43, 'MCI': 211, 'AD': 106} |
| aibl_adapt_val | 0.927 | 0.913 | 0.942 | 1.000 | 0.000 | 0.939/0.800/1.000 | {'CN': 141, 'MCI': 29, 'AD': 21} |
| aibl_heldout | 0.904 | 0.810 | 0.941 | 0.996 | 0.000 | 0.960/0.642/0.829 | {'CN': 293, 'MCI': 51, 'AD': 53} |
| oasis_external | 0.182 | 0.299 | 0.553 | 0.384 | 0.000 | 0.169/0.000/0.727 | {'CN': 18, 'MCI': 3, 'AD': 78} |
| ixi_external | 0.997 | 0.997 | 0.000 | 0.000 | 0.997 | 0.997/0.000/0.000 | {'CN': 579, 'MCI': 2, 'AD': 0} |

## internal_ad_recall

- score: 0.7948
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.352, 0.193, 0.275, 0.050, 0.029, 0.102
- offsets CN/MCI/AD: -0.758, -1.071, 1.829
- temperature: 0.580

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.521 | 0.628 | 0.753 | 0.958 | 0.000 | 0.672/0.229/0.984 | {'CN': 128, 'MCI': 51, 'AD': 176} |
| internal_test | 0.531 | 0.619 | 0.750 | 0.938 | 0.000 | 0.775/0.222/0.859 | {'CN': 125, 'MCI': 50, 'AD': 185} |
| aibl_adapt_val | 0.859 | 0.673 | 0.937 | 1.000 | 0.000 | 0.980/0.040/1.000 | {'CN': 163, 'MCI': 3, 'AD': 25} |
| aibl_heldout | 0.846 | 0.659 | 0.930 | 0.992 | 0.000 | 0.967/0.132/0.878 | {'CN': 320, 'MCI': 12, 'AD': 65} |
| oasis_external | 0.586 | 0.352 | 0.510 | 0.418 | 0.000 | 0.966/0.000/0.091 | {'CN': 89, 'MCI': 0, 'AD': 10} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## aibl_mci_recall

- score: 0.9208
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.035, 0.067, 0.223, 0.537, 0.007, 0.131
- offsets CN/MCI/AD: -0.987, 1.029, -0.042
- temperature: 0.556

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.504 | 0.387 | 0.720 | 0.867 | 0.000 | 0.136/0.946/0.078 | {'CN': 25, 'MCI': 321, 'AD': 9} |
| internal_test | 0.508 | 0.378 | 0.703 | 0.871 | 0.000 | 0.099/0.982/0.051 | {'CN': 11, 'MCI': 341, 'AD': 8} |
| aibl_adapt_val | 0.927 | 0.920 | 0.960 | 1.000 | 0.000 | 0.932/0.880/0.947 | {'CN': 140, 'MCI': 33, 'AD': 18} |
| aibl_heldout | 0.907 | 0.803 | 0.950 | 0.997 | 0.000 | 0.957/0.868/0.585 | {'CN': 292, 'MCI': 75, 'AD': 30} |
| oasis_external | 0.606 | 0.368 | 0.582 | 0.667 | 0.000 | 0.932/0.172/0.000 | {'CN': 88, 'MCI': 11, 'AD': 0} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## minority_rescue

- score: 0.7377
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.167, 0.424, 0.195, 0.057, 0.154, 0.002
- offsets CN/MCI/AD: -0.515, -0.131, 0.646
- temperature: 1.227

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.555 | 0.576 | 0.753 | 0.928 | 0.000 | 0.400/0.608/0.719 | {'CN': 63, 'MCI': 174, 'AD': 118} |
| internal_test | 0.519 | 0.496 | 0.745 | 0.935 | 0.000 | 0.351/0.637/0.500 | {'CN': 43, 'MCI': 211, 'AD': 106} |
| aibl_adapt_val | 0.927 | 0.913 | 0.942 | 1.000 | 0.000 | 0.939/0.800/1.000 | {'CN': 141, 'MCI': 29, 'AD': 21} |
| aibl_heldout | 0.904 | 0.810 | 0.941 | 0.996 | 0.000 | 0.960/0.642/0.829 | {'CN': 293, 'MCI': 51, 'AD': 53} |
| oasis_external | 0.182 | 0.299 | 0.553 | 0.384 | 0.000 | 0.169/0.000/0.727 | {'CN': 18, 'MCI': 3, 'AD': 78} |
| ixi_external | 0.997 | 0.997 | 0.000 | 0.000 | 0.997 | 0.997/0.000/0.000 | {'CN': 579, 'MCI': 2, 'AD': 0} |
