# Final Model MCI/AD Error Analysis

The analysis below uses subject-level probabilities for the locked final rescue ensemble. Errors are summarized at the subject level, with repeated scans averaged before classification.

## Confusion Patterns

### aibl_heldout Confusion Pattern

| true | predicted | n | rate within true |
|---|---|---:|---:|
| CN | CN | 148 | 0.961 |
| CN | MCI | 5 | 0.032 |
| CN | AD | 1 | 0.006 |
| MCI | CN | 2 | 0.057 |
| MCI | MCI | 24 | 0.686 |
| MCI | AD | 9 | 0.257 |
| AD | CN | 0 | 0.000 |
| AD | MCI | 4 | 0.148 |
| AD | AD | 23 | 0.852 |

### internal_test Confusion Pattern

| true | predicted | n | rate within true |
|---|---|---:|---:|
| CN | CN | 7 | 0.241 |
| CN | MCI | 20 | 0.690 |
| CN | AD | 2 | 0.069 |
| MCI | CN | 3 | 0.064 |
| MCI | MCI | 26 | 0.553 |
| MCI | AD | 18 | 0.383 |
| AD | CN | 0 | 0.000 |
| AD | MCI | 9 | 0.450 |
| AD | AD | 11 | 0.550 |

## Error-Group Feature Profiles

### aibl_heldout Error Groups

| group | n | age | MMSE | CDR-SB | hippocampus vol | ventricle vol | AD-like z | max prob | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CN_correct | 148 | 72.709 | 28.701 | 0.002 | 0.008 | 0.032 | -0.092 | 0.883 | 0.791 |
| CN_to_MCI_AD | 6 | 74.800 | 28.417 | 0.417 | 0.008 | 0.038 | 0.780 | 0.655 | 0.394 |
| MCI_correct | 24 | 75.340 | 27.753 | 0.500 | 0.008 | 0.047 | 0.584 | 0.668 | 0.403 |
| MCI_to_CN | 2 | 68.000 | 30.000 | 0.000 | 0.008 | 0.024 | 0.139 | 0.912 | 0.845 |
| MCI_to_AD | 9 | 70.944 | 23.639 | 0.556 | 0.007 | 0.039 | 0.765 | 0.730 | 0.510 |
| AD_correct | 23 | 74.193 | 21.203 | 0.830 | 0.007 | 0.060 | 1.350 | 0.862 | 0.761 |
| AD_to_CN_MCI | 4 | 74.250 | 25.750 | 0.500 | 0.008 | 0.040 | 0.679 | 0.634 | 0.422 |

### internal_test Error Groups

| group | n | age | MMSE | CDR-SB | hippocampus vol | ventricle vol | AD-like z | max prob | margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CN_correct | 7 | 74.714 | NA | NA | 0.008 | 0.038 | 0.021 | 0.659 | 0.394 |
| CN_to_MCI_AD | 22 | 77.909 | NA | NA | 0.008 | 0.037 | 0.109 | 0.591 | 0.284 |
| MCI_correct | 26 | 73.231 | NA | NA | 0.007 | 0.047 | 1.127 | 0.636 | 0.338 |
| MCI_to_CN | 3 | 78.000 | NA | NA | 0.007 | 0.033 | 1.077 | 0.544 | 0.121 |
| MCI_to_AD | 18 | 74.278 | NA | NA | 0.007 | 0.046 | 1.313 | 0.684 | 0.418 |
| AD_correct | 11 | 72.727 | NA | NA | 0.007 | 0.063 | 2.008 | 0.656 | 0.344 |
| AD_to_CN_MCI | 9 | 75.889 | NA | NA | 0.007 | 0.048 | 0.960 | 0.549 | 0.179 |

## Interpretation For The Paper

- AIBL heldout errors are mainly boundary errors between MCI and AD, not wholesale collapse into CN.
- AIBL heldout AD is rarely mistaken for CN; remaining AD errors mostly fall into MCI, which is clinically less severe than missing impairment entirely.
- Internal AD recall is improved compared with the v4 main model, but the internal confusion pattern still shows calibration tension between preserving CN specificity and recovering AD.
- OASIS remains excluded from model selection and should be discussed as a separate external stress-test failure.
