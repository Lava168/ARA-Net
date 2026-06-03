# Final Classification Table For The Revised Manuscript

| model / protocol | evaluation unit | test cohort | n | Acc | BAcc | macro AUC | AD-vs-CN AUC / CN retention | recall CN/MCI/AD | role |
|---|---|---|---:|---:|---:|---:|---|---|---|
| Old v3 ensemble | scan | AIBL external | 1307 | 0.606 | 0.399 | 0.597 | NA | NA | Failed external baseline |
| Old v3 ensemble | scan | IXI healthy | 581 | 0.439 | 0.439 | NA | CN retention 0.439 | 0.439/0.000/0.000 | Failed healthy specificity baseline |
| v4 atlas+clinical HGB | scan | AIBL heldout | 397 | 0.882 | 0.741 | 0.942 | AD-vs-CN AUC 0.990 | 0.964/0.528/0.732 | Earlier rebuilt atlas-guided model |
| v4 atlas+clinical HGB | scan | IXI healthy | 581 | 0.998 | 0.998 | NA | CN retention 0.998 | 0.998/0.000/0.000 | Healthy negative-control specificity |
| Final rescue ensemble | scan | AIBL heldout | 397 | 0.909 | 0.820 | 0.939 | AD-vs-CN AUC 0.998 | 0.964/0.642/0.854 | Scan-level reference |
| Final rescue ensemble | subject | AIBL heldout | 216 | 0.903 | 0.833 | 0.937 | AD-vs-CN AUC 1.000 | 0.961/0.686/0.852 | Locked primary result |
| Final rescue ensemble | subject | IXI healthy | 581 | 1.000 | 1.000 | NA | CN retention 1.000 | 1.000/0.000/0.000 | Locked specificity check |
| Final rescue ensemble | subject | OASIS stress test | 99 | 0.586 | 0.334 | 0.554 | AD-vs-CN AUC 0.371 | 0.966/0.034/0.000 | Limitation, not a claimed validation |
| Clinical-only RF comparator | scan | AIBL heldout | 397 | 0.922 | 0.835 | 0.957 | AD-vs-CN AUC 0.997 | 0.970/0.755/0.780 | Clinical comparator / upper bound |

Primary manuscript claim: the locked final subject-level ensemble is the main model, while the scan-level result is a reference analysis and the clinical-only model is a comparator rather than the central atlas-guided model.
