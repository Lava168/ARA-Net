# Rescue Probability Optimizer

## balanced

- score: 0.6908
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.356, 0.044, 0.323, 0.183, 0.088, 0.005
- offsets CN/MCI/AD: -0.796, -0.190, 0.986
- temperature: 0.672

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.526 | 0.536 | 0.742 | 0.897 | 0.000 | 0.333/0.574/0.700 | {'CN': 16, 'MCI': 48, 'AD': 33} |
| internal_test | 0.458 | 0.448 | 0.719 | 0.921 | 0.000 | 0.241/0.553/0.550 | {'CN': 10, 'MCI': 55, 'AD': 31} |
| aibl_adapt_val | 0.906 | 0.895 | 0.952 | 1.000 | 0.000 | 0.920/0.765/1.000 | {'CN': 71, 'MCI': 19, 'AD': 16} |
| aibl_heldout | 0.903 | 0.833 | 0.937 | 1.000 | 0.000 | 0.961/0.686/0.852 | {'CN': 150, 'MCI': 33, 'AD': 33} |
| oasis_external | 0.586 | 0.334 | 0.554 | 0.371 | 0.000 | 0.966/0.034/0.000 | {'CN': 94, 'MCI': 4, 'AD': 1} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## internal_ad_recall

- score: 0.7927
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.077, 0.147, 0.488, 0.065, 0.128, 0.095
- offsets CN/MCI/AD: -0.275, -0.334, 0.609
- temperature: 1.811

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.495 | 0.600 | 0.733 | 0.912 | 0.000 | 0.567/0.234/1.000 | {'CN': 27, 'MCI': 14, 'AD': 56} |
| internal_test | 0.417 | 0.532 | 0.722 | 0.948 | 0.000 | 0.517/0.128/0.950 | {'CN': 27, 'MCI': 10, 'AD': 59} |
| aibl_adapt_val | 0.858 | 0.706 | 0.951 | 1.000 | 0.000 | 1.000/0.118/1.000 | {'CN': 85, 'MCI': 2, 'AD': 19} |
| aibl_heldout | 0.847 | 0.698 | 0.930 | 0.999 | 0.000 | 0.987/0.143/0.963 | {'CN': 167, 'MCI': 6, 'AD': 43} |
| oasis_external | 0.586 | 0.328 | 0.517 | 0.422 | 0.000 | 0.983/0.000/0.000 | {'CN': 91, 'MCI': 0, 'AD': 8} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |

## aibl_mci_recall

- score: 0.9122
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.164, 0.292, 0.125, 0.254, 0.081, 0.085
- offsets CN/MCI/AD: -0.741, 0.750, -0.009
- temperature: 0.766

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.505 | 0.361 | 0.751 | 0.945 | 0.000 | 0.033/1.000/0.050 | {'CN': 1, 'MCI': 95, 'AD': 1} |
| internal_test | 0.521 | 0.377 | 0.776 | 0.929 | 0.000 | 0.103/0.979/0.050 | {'CN': 3, 'MCI': 91, 'AD': 2} |
| aibl_adapt_val | 0.915 | 0.910 | 0.956 | 1.000 | 0.000 | 0.920/0.882/0.929 | {'CN': 71, 'MCI': 22, 'AD': 13} |
| aibl_heldout | 0.898 | 0.811 | 0.952 | 0.998 | 0.000 | 0.955/0.886/0.593 | {'CN': 149, 'MCI': 49, 'AD': 18} |
| oasis_external | 0.364 | 0.367 | 0.566 | 0.655 | 0.000 | 0.136/0.966/0.000 | {'CN': 10, 'MCI': 89, 'AD': 0} |
| ixi_external | 0.959 | 0.959 | 0.000 | 0.000 | 0.959 | 0.959/0.000/0.000 | {'CN': 557, 'MCI': 24, 'AD': 0} |

## minority_rescue

- score: 0.7111
- runs: aibl_adapted_atlas_biomarker_enhanced__hgb, aibl_adapted_atlas_core_clinical__hgb, aibl_adapted_clinical_biomarker_only__rf_balanced, aibl_adapted_clinical_core_only__hgb, aibl_adapted_clinical_core_only__rf_balanced, rf__logreg
- weights: 0.356, 0.044, 0.323, 0.183, 0.088, 0.005
- offsets CN/MCI/AD: -0.796, -0.190, 0.986
- temperature: 0.672

| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |
|---|---:|---:|---:|---:|---:|---|---|
| val | 0.526 | 0.536 | 0.742 | 0.897 | 0.000 | 0.333/0.574/0.700 | {'CN': 16, 'MCI': 48, 'AD': 33} |
| internal_test | 0.458 | 0.448 | 0.719 | 0.921 | 0.000 | 0.241/0.553/0.550 | {'CN': 10, 'MCI': 55, 'AD': 31} |
| aibl_adapt_val | 0.906 | 0.895 | 0.952 | 1.000 | 0.000 | 0.920/0.765/1.000 | {'CN': 71, 'MCI': 19, 'AD': 16} |
| aibl_heldout | 0.903 | 0.833 | 0.937 | 1.000 | 0.000 | 0.961/0.686/0.852 | {'CN': 150, 'MCI': 33, 'AD': 33} |
| oasis_external | 0.586 | 0.334 | 0.554 | 0.371 | 0.000 | 0.966/0.034/0.000 | {'CN': 94, 'MCI': 4, 'AD': 1} |
| ixi_external | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000/0.000/0.000 | {'CN': 581, 'MCI': 0, 'AD': 0} |
