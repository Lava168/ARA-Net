| model | evaluation | acc | balanced_acc | macro_auc | ad_vs_cn_auc | cn_retention | recall_cn | recall_mci | recall_ad | note |
|---|---|---|---|---|---|---|---|---|---|---|
| Old v3 ensemble | AIBL external | 0.606 | 0.399 | 0.597 |  |  |  |  |  | Failed external baseline |
| Old v3 ensemble | IXI healthy | 0.439 | 0.439 | NA |  | 0.439 |  |  |  | High false impairment rate |
| Atlas-only HGB | AIBL heldout | 0.776 | 0.479 | 0.732 | 0.884 |  | 0.944 | 0.151 | 0.341 | MRI-only baseline |
| Cascade RF-logreg | AIBL heldout | 0.751 | 0.391 | 0.756 | 0.886 |  | 0.954 | 0.000 | 0.220 | High specificity but weak MCI |
| ADNI-only hybrid RF | AIBL heldout | 0.549 | 0.406 | 0.753 | 0.870 |  | 0.614 | 0.604 | 0.000 | Zero-shot remains insufficient |
| Recommended atlas+clinical HGB | AIBL heldout | 0.882 | 0.741 | 0.942 | 0.990 |  | 0.964 | 0.528 | 0.732 | Main atlas-guided multimodal model |
| Recommended atlas+clinical HGB | IXI healthy | 0.998 | 0.998 | NA |  | 0.998 | 0.998 | 0.000 | 0.000 | Healthy negative control |
| Clinical-only RF | AIBL heldout | 0.922 | 0.835 | 0.957 | 0.997 |  | 0.970 | 0.755 | 0.780 | Comparator / upper bound |
| Biomarker-enhanced HGB | AIBL heldout | 0.861 | 0.703 | 0.942 | 0.990 |  | 0.957 | 0.396 | 0.756 | Sensitivity analysis |
