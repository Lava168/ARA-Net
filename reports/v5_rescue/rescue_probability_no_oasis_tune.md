# Rescue Probability Optimizer

## balanced

- score: 0.7127
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_biomarker_enhanced__logreg_balanced, aibl_adapted_atlas_cognitive__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__logreg_balanced, aibl_adapted_clinical_core_only__rf_balanced, aibl_adapted_clinical_core_only__svm_rbf_balanced
- weights: 0.243, 0.061, 0.112, 0.070, 0.062, 0.035, 0.156, 0.191, 0.015, 0.056
- offsets CN/MCI/AD: -0.766, -0.209, 0.975
- temperature: 0.684

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.549 | 0.579 | 0.759 | 0.922 | 0.000 | 0.408/0.578/0.750 | {'CN': 70, 'MCI': 164, 'AD': 121} |
| internal_test | 0.486 | 0.471 | 0.742 | 0.916 | 0.000 | 0.333/0.579/0.500 | {'CN': 43, 'MCI': 201, 'AD': 116} |
| aibl_adapt_val | 0.927 | 0.913 | 0.945 | 1.000 | 0.000 | 0.939/0.800/1.000 | {'CN': 141, 'MCI': 29, 'AD': 21} |
| aibl_heldout | 0.909 | 0.820 | 0.939 | 0.998 | 0.000 | 0.964/0.642/0.854 | {'CN': 294, 'MCI': 50, 'AD': 53} |
| oasis_external | 0.596 | 0.333 | 0.566 | 0.471 | 0.000 | 1.000/0.000/0.000 | {'CN': 99, 'MCI': 0, 'AD': 0} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## internal_ad_recall

- score: 0.7981
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_biomarker_enhanced__logreg_balanced, aibl_adapted_atlas_cognitive__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__logreg_balanced, aibl_adapted_clinical_core_only__rf_balanced, aibl_adapted_clinical_core_only__svm_rbf_balanced
- weights: 0.305, 0.184, 0.134, 0.088, 0.089, 0.090, 0.019, 0.015, 0.065, 0.012
- offsets CN/MCI/AD: -0.435, -0.504, 0.939
- temperature: 1.293

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.501 | 0.610 | 0.745 | 0.952 | 0.000 | 0.584/0.247/1.000 | {'CN': 104, 'MCI': 56, 'AD': 195} |
| internal_test | 0.494 | 0.593 | 0.734 | 0.942 | 0.000 | 0.631/0.199/0.949 | {'CN': 99, 'MCI': 48, 'AD': 213} |
| aibl_adapt_val | 0.864 | 0.731 | 0.927 | 1.000 | 0.000 | 0.952/0.240/1.000 | {'CN': 152, 'MCI': 12, 'AD': 27} |
| aibl_heldout | 0.864 | 0.700 | 0.915 | 0.996 | 0.000 | 0.970/0.226/0.902 | {'CN': 314, 'MCI': 22, 'AD': 61} |
| oasis_external | 0.596 | 0.358 | 0.498 | 0.438 | 0.000 | 0.983/0.000/0.091 | {'CN': 93, 'MCI': 0, 'AD': 6} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## aibl_mci_recall

- score: 0.9222
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_biomarker_enhanced__logreg_balanced, aibl_adapted_atlas_cognitive__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__logreg_balanced, aibl_adapted_clinical_core_only__rf_balanced, aibl_adapted_clinical_core_only__svm_rbf_balanced
- weights: 0.063, 0.018, 0.001, 0.072, 0.244, 0.026, 0.084, 0.408, 0.031, 0.054
- offsets CN/MCI/AD: -1.424, 0.671, 0.753
- temperature: 0.711

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.465 | 0.344 | 0.707 | 0.855 | 0.000 | 0.000/0.970/0.062 | {'CN': 0, 'MCI': 343, 'AD': 12} |
| internal_test | 0.447 | 0.332 | 0.661 | 0.868 | 0.000 | 0.000/0.895/0.103 | {'CN': 0, 'MCI': 331, 'AD': 29} |
| aibl_adapt_val | 0.571 | 0.792 | 0.945 | 1.000 | 0.000 | 0.456/0.920/1.000 | {'CN': 68, 'MCI': 103, 'AD': 20} |
| aibl_heldout | 0.572 | 0.703 | 0.917 | 0.999 | 0.000 | 0.498/0.830/0.780 | {'CN': 152, 'MCI': 203, 'AD': 42} |
| oasis_external | 0.586 | 0.339 | 0.583 | 0.624 | 0.000 | 0.949/0.069/0.000 | {'CN': 93, 'MCI': 6, 'AD': 0} |
| ixi_external | 0.998 | 0.998 | 0.000 | 0.000 | 0.998 | 0.998/0.000/0.000 | {'CN': 580, 'MCI': 1, 'AD': 0} |

## minority_rescue

- score: 0.7483
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_biomarker_enhanced__logreg_balanced, aibl_adapted_atlas_cognitive__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__logreg_balanced, aibl_adapted_clinical_core_only__rf_balanced, aibl_adapted_clinical_core_only__svm_rbf_balanced
- weights: 0.168, 0.078, 0.210, 0.184, 0.002, 0.099, 0.039, 0.067, 0.004, 0.149
- offsets CN/MCI/AD: -0.340, -0.029, 0.369
- temperature: 1.725

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.555 | 0.561 | 0.765 | 0.945 | 0.000 | 0.336/0.675/0.672 | {'CN': 53, 'MCI': 198, 'AD': 104} |
| internal_test | 0.514 | 0.467 | 0.767 | 0.937 | 0.000 | 0.297/0.708/0.397 | {'CN': 36, 'MCI': 240, 'AD': 84} |
| aibl_adapt_val | 0.921 | 0.911 | 0.937 | 1.000 | 0.000 | 0.932/0.800/1.000 | {'CN': 140, 'MCI': 30, 'AD': 21} |
| aibl_heldout | 0.904 | 0.810 | 0.942 | 0.996 | 0.000 | 0.960/0.642/0.829 | {'CN': 293, 'MCI': 52, 'AD': 52} |
| oasis_external | 0.566 | 0.316 | 0.548 | 0.404 | 0.000 | 0.949/0.000/0.000 | {'CN': 93, 'MCI': 4, 'AD': 2} |
| ixi_external | 0.998 | 0.998 | 0.000 | 0.000 | 0.998 | 0.998/0.000/0.000 | {'CN': 580, 'MCI': 1, 'AD': 0} |
