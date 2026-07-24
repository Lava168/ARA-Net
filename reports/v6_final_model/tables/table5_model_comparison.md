# Table 5 | ARA-Net and Baseline Model Comparison Across Multi-Cohort AD Staging

| Model / strategy | Evaluation unit | AIBL BAcc | AIBL macro AUC | AD-vs-CN AUC | CN/MCI/AD recall | IXI CN retention | OASIS BAcc | Model size / parameters |
|---|---|---:|---:|---:|---|---:|---:|---|
| Old v3 ensemble | scan | 39.9% | 59.7% | N/A | N/A | 43.9% | N/A | N/A |
| Atlas-only HGB | scan | 47.9% | 73.2% | 88.4% | 94.4%/15.1%/34.1% | 98.3% | N/A | N/A |
| Cascade RF-logreg | scan | 39.1% | 75.6% | 88.6% | 95.4%/0.0%/22.0% | 100.0% | N/A | N/A |
| ADNI-only hybrid RF | scan | 40.6% | 75.3% | 87.0% | 61.4%/60.4%/0.0% | N/A | N/A | N/A |
| v4 atlas+clinical HGB | scan | 74.1% | 94.2% | 99.0% | 96.4%/52.8%/73.2% | 99.8% | 32.2% | N/A |
| Biomarker-enhanced HGB | scan | 70.3% | 94.2% | N/A | 95.7%/39.6%/75.6% | 99.7% | 31.0% | N/A |
| Clinical-only RF comparator | scan | 83.5% | 95.7% | 99.7% | 97.0%/75.5%/78.0% | 100.0% | 33.3% | N/A |
| Best single base model | subject | 75.6% | 94.5% | 99.4% | 95.5%/57.1%/74.1% | 99.7% | 31.0% | N/A |
| Full RC-SPE | scan | 82.0% | 93.9% | 99.8% | 96.4%/64.2%/85.4% | 100.0% | 33.4% | 10 scalar params / 1,663 bytes JSON |
| ARA-Net / RC-SPE | subject | 83.3% | 93.7% | 100.0% | 96.1%/68.6%/85.2% | 100.0% | 33.4% | 10 scalar params / 1,663 bytes JSON |

Note: AIBL is the primary locked external test cohort; IXI is used as a healthy-control specificity check; OASIS is reported as a stress-test limitation, not as a successful external validation. Model checkpoint sizes were not computed for baselines other than ARA-Net / RC-SPE and are therefore reported as N/A.
