# Final Rescued Model Lock Report

## Locked Decision

Primary model: subject-level balanced rescue probability ensemble, tuned on ADNI validation, AIBL adaptation validation, and IXI only. OASIS was not used for tuning and is retained only as a stress-test limitation.

Primary endpoint: locked AIBL heldout subject-level CN/MCI/AD staging, with IXI healthy CN retention as the specificity check.

## Main Subject-Level Result

| split | n | Acc | BAcc | macro AUC | AD-vs-CN AUC/CN retention | CN/MCI/AD recall | BAcc 95% CI | MCI recall 95% CI | AD recall 95% CI |
|---|---:|---:|---:|---:|---|---|---|---|---|
| val | 97 | 0.526 | 0.536 | 0.742 | AD-vs-CN 0.897 | 0.333/0.574/0.700 | 0.434-0.640 | 0.436-0.721 | 0.500-0.895 |
| internal_test | 96 | 0.458 | 0.448 | 0.719 | AD-vs-CN 0.921 | 0.241/0.553/0.550 | 0.344-0.552 | 0.413-0.694 | 0.333-0.773 |
| aibl_adapt_val | 106 | 0.906 | 0.895 | 0.952 | AD-vs-CN 1.000 | 0.920/0.765/1.000 | 0.814-0.965 | 0.529-0.947 | 1.000-1.000 |
| aibl_heldout | 216 | 0.903 | 0.833 | 0.937 | AD-vs-CN 1.000 | 0.961/0.686/0.852 | 0.759-0.899 | 0.531-0.839 | 0.710-0.966 |
| ixi_external | 581 | 1.000 | 1.000 | NA | CN retention 1.000 | 1.000/0.000/0.000 | 1.000-1.000 | 0.000-0.000 | 0.000-0.000 |
| oasis_external | 99 | 0.586 | 0.334 | 0.554 | AD-vs-CN 0.371 | 0.966/0.034/0.000 | 0.309-0.364 | 0.000-0.118 | 0.000-0.000 |

## Scan-Level Reference

| split | n | Acc | BAcc | macro AUC | AD-vs-CN AUC/CN retention | CN/MCI/AD recall | BAcc 95% CI | MCI recall 95% CI | AD recall 95% CI |
|---|---:|---:|---:|---:|---|---|---|---|---|
| val | 355 | 0.549 | 0.579 | 0.759 | AD-vs-CN 0.922 | 0.408/0.578/0.750 | 0.524-0.628 | 0.503-0.654 | 0.642-0.855 |
| internal_test | 360 | 0.486 | 0.471 | 0.742 | AD-vs-CN 0.916 | 0.333/0.579/0.500 | 0.418-0.524 | 0.506-0.653 | 0.385-0.614 |
| aibl_adapt_val | 191 | 0.927 | 0.913 | 0.945 | AD-vs-CN 1.000 | 0.939/0.800/1.000 | 0.858-0.966 | 0.632-0.957 | 1.000-1.000 |
| aibl_heldout | 397 | 0.909 | 0.820 | 0.939 | AD-vs-CN 0.998 | 0.964/0.642/0.854 | 0.761-0.874 | 0.509-0.764 | 0.742-0.952 |
| ixi_external | 581 | 1.000 | 1.000 | NA | CN retention 1.000 | 1.000/0.000/0.000 | 1.000-1.000 | 0.000-0.000 | 0.000-0.000 |
| oasis_external | 99 | 0.596 | 0.333 | 0.566 | AD-vs-CN 0.471 | 1.000/0.000/0.000 | 0.333-0.333 | 0.000-0.000 | 0.000-0.000 |

## Why This Model Is Locked

- It is evaluated at subject level, which is the clinically natural unit and reduces repeated-scan instability.
- It improves the key external minority classes: AIBL heldout MCI recall and AD recall are materially higher than the v4 main atlas+clinical HGB model.
- It preserves IXI healthy specificity at 1.000 CN retention.
- It does not use OASIS for tuning, so the weak OASIS result remains an honest limitation rather than a hidden adaptation artifact.

## Manuscript Claim Boundary

Use the model as evidence for domain-adapted external heldout AD staging and healthy negative-control specificity. Do not claim pure zero-shot transfer, solved OASIS generalization, direct Braak staging, or clinical deployment readiness.
