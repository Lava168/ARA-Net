# Model Card: ARA-Net RC-SPE

## Model Summary

ARA-Net is an atlas-guided multimodal Alzheimer's disease staging research framework for CN/MCI/AD classification. The final manuscript-facing algorithm is **RC-SPE**: a Risk-Constrained Subject-level Probability Ensemble. It combines six heterogeneous CN/MCI/AD probability streams using non-negative weighted log-probability pooling, class-specific offsets, temperature scaling, and subject-level probability aggregation.

The public deployment wrapper operates on already-produced base-model probabilities. It does not process raw MRI files and does not redistribute restricted MRI data, private checkpoints, or subject-level cohort tables.

## Intended Use

ARA-Net is intended for retrospective research, reproducibility review, and future prospective validation studies. It is not intended for direct diagnosis, treatment selection, emergency triage, or unsupervised patient care.

## Inputs

The full research workflow uses structural MRI atlas features and core clinical variables, including age, sex, education, APOE4, MMSE, and CDR-SB. The public deployment wrapper expects six base-model CN/MCI/AD probability streams plus subject/scan grouping fields for aggregation.

## Outputs

The model outputs:

- predicted class: CN, MCI, or AD
- calibrated class probabilities
- confidence
- decision margin
- clinical-use notice

## Primary Evaluation

Locked AIBL external subject-level endpoint:

| Metric | Value |
|---|---:|
| Accuracy | 0.903 |
| Balanced accuracy | 0.833 |
| Macro AUC | 0.937 |
| AD-vs-CN AUC | 1.000 |
| Recall CN/MCI/AD | 0.961 / 0.686 / 0.852 |

Locked AIBL scan-level endpoint:

| Metric | Value |
|---|---:|
| Accuracy | 0.909 |
| Balanced accuracy | 0.820 |
| Macro AUC | 0.939 |
| AD-vs-CN AUC | 0.998 |
| Recall CN/MCI/AD | 0.964 / 0.642 / 0.854 |

IXI healthy-control CN retention was 1.000.

## Calibration And Ablation Evidence

| Variant | AIBL BAcc | MCI recall | AD recall | AD-to-CN errors | IXI CN retention | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Best single base model | 0.756 | 0.571 | 0.741 | 2 | 0.997 | 0.122 |
| Arithmetic mean ensemble | 0.711 | 0.400 | 0.741 | 3 | 1.000 | 0.112 |
| Equal log-pooling | 0.648 | 0.171 | 0.778 | 5 | 1.000 | 0.094 |
| Full RC-SPE | 0.833 | 0.686 | 0.852 | 0 | 1.000 | 0.078 |

The final RC-SPE head has 10 scalar parameters: six stream weights, three class offsets, and one temperature parameter.

## Structural Consistency

ARA-Net evaluates model-associated structural evidence against prespecified AD-key regions: bilateral hippocampi, bilateral amygdalae, and bilateral lateral ventricles. In the locked AIBL external test set, the AD-key enrichment score was 0.510 versus a uniform-null expectation of 0.286 (`p = 0.026`; bootstrap 95% CI 0.479-0.526).

## Known Limitations

- MCI remains the main uncertainty source.
- OASIS is reported as an external stress-test boundary, not a successful zero-shot generalization result.
- The public deployment wrapper assumes upstream MRI processing and base probability generation have already been completed.
- Structural consistency is an MRI neurodegeneration proxy, not direct Braak staging.
- The model has not undergone prospective clinical validation or regulatory review.

## Ethical And Safety Notes

This model should not be used to make clinical decisions. Outputs should be interpreted only in a research setting by qualified investigators.
