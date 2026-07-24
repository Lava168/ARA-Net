# Hybrid Candidate Ranking

## adni_only

| rank | run | AIBL BAcc | AIBL AUC | CN/MCI/AD recall | IXI CN retention | Internal BAcc | OASIS BAcc | pred |
|---:|---|---:|---:|---|---:|---:|---:|---|
| 1 | `atlas_only__rf_balanced` | 0.451 | 0.777 | 0.729/0.623/0.000 | 0.079 | 0.495 | 0.356 | {'CN': 244, 'MCI': 153, 'AD': 0} |
| 2 | `atlas_demographic__rf_balanced` | 0.436 | 0.758 | 0.703/0.604/0.000 | 0.960 | 0.536 | 0.367 | {'CN': 237, 'MCI': 160, 'AD': 0} |
| 3 | `atlas_biomarker_enhanced__svm_rbf_balanced` | 0.435 | 0.701 | 0.812/0.396/0.098 | 0.380 | 0.405 | 0.391 | {'CN': 281, 'MCI': 102, 'AD': 14} |
| 4 | `atlas_core_clinical__rf_balanced` | 0.431 | 0.748 | 0.634/0.660/0.000 | 0.998 | 0.547 | 0.367 | {'CN': 214, 'MCI': 183, 'AD': 0} |
| 5 | `atlas_only__extratrees_balanced` | 0.431 | 0.766 | 0.782/0.509/0.000 | 0.028 | 0.472 | 0.374 | {'CN': 267, 'MCI': 130, 'AD': 0} |
| 6 | `atlas_cognitive__rf_balanced` | 0.426 | 0.744 | 0.657/0.623/0.000 | 0.978 | 0.538 | 0.367 | {'CN': 222, 'MCI': 175, 'AD': 0} |
| 7 | `atlas_biomarker_enhanced__extratrees_balanced` | 0.421 | 0.753 | 0.545/0.717/0.000 | 0.836 | 0.471 | 0.357 | {'CN': 184, 'MCI': 213, 'AD': 0} |
| 8 | `atlas_biomarker_enhanced__hgb` | 0.420 | 0.706 | 0.525/0.736/0.000 | 0.995 | 0.560 | 0.362 | {'CN': 179, 'MCI': 217, 'AD': 1} |
| 9 | `atlas_demographic__hgb` | 0.420 | 0.703 | 0.617/0.642/0.000 | 1.000 | 0.527 | 0.403 | {'CN': 216, 'MCI': 181, 'AD': 0} |
| 10 | `atlas_core_clinical__hgb` | 0.420 | 0.703 | 0.617/0.642/0.000 | 1.000 | 0.527 | 0.403 | {'CN': 216, 'MCI': 181, 'AD': 0} |
| 11 | `atlas_cognitive__hgb` | 0.420 | 0.703 | 0.617/0.642/0.000 | 1.000 | 0.527 | 0.403 | {'CN': 216, 'MCI': 181, 'AD': 0} |
| 12 | `atlas_demographic__extratrees_balanced` | 0.410 | 0.734 | 0.703/0.528/0.000 | 0.962 | 0.496 | 0.374 | {'CN': 246, 'MCI': 151, 'AD': 0} |
| 13 | `atlas_biomarker_enhanced__rf_balanced` | 0.406 | 0.753 | 0.614/0.604/0.000 | 0.179 | 0.580 | 0.357 | {'CN': 210, 'MCI': 187, 'AD': 0} |
| 14 | `atlas_cognitive__extratrees_balanced` | 0.405 | 0.737 | 0.686/0.528/0.000 | 0.967 | 0.497 | 0.374 | {'CN': 241, 'MCI': 156, 'AD': 0} |
| 15 | `atlas_core_clinical__extratrees_balanced` | 0.404 | 0.733 | 0.723/0.491/0.000 | 0.971 | 0.493 | 0.380 | {'CN': 257, 'MCI': 140, 'AD': 0} |
| 16 | `clinical_core_only__rf_balanced` | 0.389 | 0.597 | 0.337/0.830/0.000 | 0.000 | 0.432 | 0.333 | {'CN': 112, 'MCI': 285, 'AD': 0} |
| 17 | `atlas_only__logreg_balanced` | 0.383 | 0.599 | 0.007/0.264/0.878 | 0.157 | 0.463 | 0.290 | {'CN': 2, 'MCI': 111, 'AD': 284} |
| 18 | `atlas_only__hgb` | 0.378 | 0.679 | 0.587/0.547/0.000 | 0.864 | 0.487 | 0.386 | {'CN': 206, 'MCI': 191, 'AD': 0} |

## aibl_adapted

| rank | run | AIBL BAcc | AIBL AUC | CN/MCI/AD recall | IXI CN retention | Internal BAcc | OASIS BAcc | pred |
|---:|---|---:|---:|---|---:|---:|---:|---|
| 1 | `clinical_core_only__rf_balanced` | 0.835 | 0.957 | 0.970/0.755/0.780 | 1.000 | 0.411 | 0.333 | {'CN': 297, 'MCI': 58, 'AD': 42} |
| 2 | `clinical_biomarker_only__rf_balanced` | 0.792 | 0.958 | 0.974/0.623/0.780 | 1.000 | 0.357 | 0.333 | {'CN': 304, 'MCI': 50, 'AD': 43} |
| 3 | `clinical_core_only__hgb` | 0.782 | 0.951 | 0.970/0.717/0.659 | 1.000 | 0.434 | 0.333 | {'CN': 304, 'MCI': 60, 'AD': 33} |
| 4 | `clinical_biomarker_only__hgb` | 0.773 | 0.945 | 0.980/0.679/0.659 | 1.000 | 0.396 | 0.333 | {'CN': 312, 'MCI': 55, 'AD': 30} |
| 5 | `clinical_core_only__logreg_balanced` | 0.747 | 0.955 | 0.970/0.491/0.780 | 1.000 | 0.430 | 0.333 | {'CN': 306, 'MCI': 44, 'AD': 47} |
| 6 | `atlas_core_clinical__hgb` | 0.741 | 0.942 | 0.964/0.528/0.732 | 0.998 | 0.548 | 0.322 | {'CN': 311, 'MCI': 47, 'AD': 39} |
| 7 | `atlas_cognitive__hgb` | 0.741 | 0.942 | 0.964/0.528/0.732 | 0.998 | 0.548 | 0.322 | {'CN': 311, 'MCI': 47, 'AD': 39} |
| 8 | `clinical_core_only__svm_rbf_balanced` | 0.722 | 0.951 | 0.970/0.415/0.780 | 1.000 | 0.473 | 0.333 | {'CN': 308, 'MCI': 40, 'AD': 49} |
| 9 | `atlas_biomarker_enhanced__hgb` | 0.703 | 0.942 | 0.957/0.396/0.756 | 0.997 | 0.570 | 0.310 | {'CN': 314, 'MCI': 40, 'AD': 43} |
| 10 | `clinical_biomarker_only__logreg_balanced` | 0.697 | 0.951 | 0.980/0.283/0.829 | 1.000 | 0.476 | 0.333 | {'CN': 318, 'MCI': 28, 'AD': 51} |
| 11 | `atlas_biomarker_enhanced__logreg_balanced` | 0.653 | 0.947 | 0.983/0.170/0.805 | 1.000 | 0.536 | 0.345 | {'CN': 325, 'MCI': 21, 'AD': 51} |
| 12 | `atlas_cognitive__logreg_balanced` | 0.650 | 0.952 | 0.980/0.189/0.780 | 1.000 | 0.540 | 0.333 | {'CN': 325, 'MCI': 24, 'AD': 48} |
| 13 | `atlas_core_clinical__logreg_balanced` | 0.650 | 0.952 | 0.980/0.189/0.780 | 1.000 | 0.540 | 0.333 | {'CN': 325, 'MCI': 24, 'AD': 48} |
| 14 | `atlas_core_clinical__svm_rbf_balanced` | 0.609 | 0.809 | 0.950/0.170/0.707 | 0.031 | 0.455 | 0.327 | {'CN': 322, 'MCI': 32, 'AD': 43} |
| 15 | `atlas_cognitive__svm_rbf_balanced` | 0.609 | 0.809 | 0.950/0.170/0.707 | 0.031 | 0.455 | 0.327 | {'CN': 322, 'MCI': 32, 'AD': 43} |
| 16 | `atlas_biomarker_enhanced__rf_balanced` | 0.533 | 0.932 | 0.977/0.208/0.415 | 0.800 | 0.524 | 0.322 | {'CN': 353, 'MCI': 19, 'AD': 25} |
| 17 | `atlas_core_clinical__rf_balanced` | 0.527 | 0.934 | 0.977/0.189/0.415 | 0.991 | 0.543 | 0.321 | {'CN': 353, 'MCI': 19, 'AD': 25} |
| 18 | `atlas_cognitive__rf_balanced` | 0.522 | 0.931 | 0.977/0.151/0.439 | 0.983 | 0.531 | 0.321 | {'CN': 352, 'MCI': 16, 'AD': 29} |
