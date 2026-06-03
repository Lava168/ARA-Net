# Model Card: ARA-Net V6 Subject-Level Rescue Ensemble

## Model Summary

ARA-Net V6 is an atlas-guided multimodal Alzheimer's disease staging research model. The public deployment wrapper combines probabilities from six base models using log-probability pooling, temperature scaling, and class-specific offsets. Repeated scans are averaged at subject level for the primary prediction.

## Intended Use

The model is intended for retrospective research evaluation and prospective clinical-decision-support studies. It is not intended for direct diagnosis, treatment selection, emergency triage, or unsupervised patient care.

## Input

The public deployment wrapper expects class probabilities from the configured base models. It does not process raw MRI files. This design avoids redistributing restricted datasets or model artifacts while allowing the final ensemble and subject-level calibration logic to be reproduced.

## Output

The model outputs:

- predicted class: CN, MCI, or AD
- calibrated class probabilities
- confidence
- decision margin
- clinical-use notice

## Primary Evaluation

Locked AIBL heldout subject-level evaluation:

| Metric | Value |
|---|---:|
| Accuracy | 0.903 |
| Balanced accuracy | 0.833 |
| Macro AUC | 0.937 |
| AD-vs-CN AUC | 1.000 |
| Recall CN/MCI/AD | 0.961 / 0.686 / 0.852 |

Bootstrap 95% confidence intervals:

| Metric | 95% CI |
|---|---:|
| Balanced accuracy | 0.759-0.899 |
| MCI recall | 0.531-0.839 |
| AD recall | 0.710-0.966 |

IXI healthy-control CN retention was 1.000.

## Known Limitations

- OASIS transfer remains weak and is reported as an external stress-test limitation.
- AIBL validation is domain-adapted external heldout evaluation, not pure ADNI-to-AIBL zero-shot transfer.
- Internal subject-level balanced accuracy remains modest.
- Clinical variables contain strong diagnostic signal; the clinical-only comparator is an important upper bound.
- The biological validation is an MRI neurodegeneration proxy, not direct Braak staging.
- The model has not undergone prospective clinical validation or regulatory review.

## Ethical And Safety Notes

This model should not be used to make clinical decisions. Outputs should be interpreted only in a research setting by qualified investigators.
