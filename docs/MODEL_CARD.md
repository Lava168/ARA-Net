# Model Card: ARA-Net V6 RC-SPE

## Model Summary

ARA-Net V6 is an atlas-guided multimodal Alzheimer's disease staging research model. The locked algorithm is **RC-SPE**: a risk-constrained subject-level probability ensemble. The public deployment wrapper combines probabilities from six base models using log-probability pooling, non-negative model weights, temperature scaling, and class-specific offsets. Repeated scans are averaged at the subject-level endpoint unit for the primary prediction.

## Intended Use

The model is intended for retrospective research evaluation and future prospective validation studies. It is not intended for direct diagnosis, treatment selection, emergency triage, or unsupervised patient care.

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

The reported `n` is the number of evaluable subject-level endpoint units after probability aggregation, not necessarily the raw unique-participant count in the split inventory.

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

## Algorithmic Evidence

The RC-SPE evidence package compares the locked algorithm against single-model, simple ensemble, partial-parameter, calibration, risk-profile, and leave-one-model-out variants using aggregate metrics only.

| Variant | AIBL BAcc | MCI recall | AD recall | AD-to-CN errors | IXI CN retention |
|---|---:|---:|---:|---:|---:|
| Best single base model | 0.756 | 0.571 | 0.741 | 2 | 0.997 |
| Arithmetic mean ensemble | 0.711 | 0.400 | 0.741 | 3 | 1.000 |
| Equal log-pooling | 0.648 | 0.171 | 0.778 | 5 | 1.000 |
| Full RC-SPE, subject-level | 0.833 | 0.686 | 0.852 | 0 | 1.000 |

The final algorithm reduced AIBL expected calibration error from 0.122 for the best single base model to 0.078 while improving balanced accuracy. A high-MCI-recall risk profile reached MCI recall 0.886 but reduced AD recall to 0.593 and IXI CN retention to 0.959, so it was not selected as the locked profile.

## Known Limitations

- OASIS transfer remains weak and is reported as an external stress-test limitation.
- AIBL validation is domain-adapted external heldout evaluation, not pure ADNI-to-AIBL zero-shot transfer.
- Internal subject-level balanced accuracy remains modest.
- Clinical variables contain strong diagnostic signal; the clinical-only comparator is an important upper bound.
- RC-SPE is a calibrated risk-constrained probability framework, not a new end-to-end neural architecture.
- The biological validation is an MRI neurodegeneration proxy, not direct Braak staging.
- The model has not undergone prospective clinical validation or regulatory review.

## Ethical And Safety Notes

This model should not be used to make clinical decisions. Outputs should be interpreted only in a research setting by qualified investigators.
