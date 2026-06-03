# Core Reviewer Evidence Matrix

## Purpose

This generated matrix consolidates the evidence that the revised ARA-Net work is a substantive rebuild rather than a small v3 patch. It focuses on the three critical issues: external classification, invalid CAS, and non-significant Braak validation.

## Generated From

- `reports/v6_final_model/final_rescue_model_summary_public.json` sha256:1248b2f9da44
- `reports/v4/tables/table4_neurodegeneration.csv` sha256:d1ecf85ce88d
- `reports/v4/tables/table2_classification.csv` sha256:9eac4ffe8c26
- `deployment/final_ensemble_config.json` sha256:388ccf80aff5

## Locked Model

- Model: ARA-Net V6 subject-level rescue ensemble (v6.0-research)
- Classes: CN, MCI, AD
- Primary endpoint: AIBL locked heldout subject-level CN/MCI/AD staging.
- Clinical-use boundary: Not a medical device. Not cleared or approved for clinical use.

## Requirement-Level Evidence

| reviewer issue | current status | quantitative evidence | manuscript use | claim boundary |
| --- | --- | --- | --- | --- |
| External CN/MCI/AD classification | Substantially addressed | Old v3 AIBL BAcc 0.399 -> v4 atlas+clinical BAcc 0.741 -> final subject AIBL BAcc 0.833 (95% CI 0.759-0.899); Acc 0.903, macro AUC 0.937, AD-vs-CN AUC 1.000, recall CN/MCI/AD 0.961/0.686/0.852. | Lead with locked AIBL subject-level result. | Domain-adapted external heldout, not pure zero-shot transfer. |
| Healthy external specificity | Addressed for negative-control use | Old v3 IXI CN retention 0.439; final IXI CN retention 1.000 with 581 healthy subjects and false-impairment rate 0.000. | Report IXI as healthy negative-control specificity. | IXI does not provide AD/MCI staging labels. |
| OASIS external transfer | Not solved; preserved as limitation | OASIS subject BAcc 0.334, macro AUC 0.554, AD-vs-CN AUC 0.371, recall CN/MCI/AD 0.966/0.034/0.000. | Keep OASIS as stress-test limitation. | Do not claim OASIS validation success. |
| CAS validity | Resolved by removing/replacing invalid attention-only CAS | AIBL AD-key atlas-volume consistency score 0.510 vs uniform null 0.286, delta 0.225, 95% CI 0.479-0.526, permutation p=0.0260. | Replace CAS biomarker wording with atlas structural neurodegeneration consistency. | This validates a structural MRI proxy, not attention maps as biomarkers. |
| Braak or substitute biology | Addressed as Braak-alternative proxy, not direct Braak proof | All labeled AD-key consistency score 0.426 vs null 0.286, p=0.0207; ADNI-only internal check remains non-significant (score 0.342, p=0.1843). | Use 'MRI neurodegeneration proxy' and remove direct Braak-stage claims. | No neuropathological Braak-stage validation is available. |
| MCI/AD error risk | Improved and explicitly quantified | AIBL confusion rows CN/MCI/AD: [148, 5, 1], [2, 24, 9], [0, 4, 23]; AD-to-CN errors 0; MCI recall 95% CI 0.531-0.839; AD recall 95% CI 0.710-0.966. | Report confusion matrix and error-profile figure. | MCI remains the main residual weakness. |
| Internal calibration risk | Open limitation | Internal subject BAcc 0.448, recall CN/MCI/AD 0.241/0.553/0.550, confusion rows CN/MCI/AD: [7, 20, 2], [3, 26, 18], [0, 9, 11]. | Discuss internal CN-to-MCI shift as calibration limitation. | Do not overstate universal readiness. |

## Primary Metrics Snapshot

| cohort | n | Acc | BAcc | macro AUC | AD-vs-CN / specificity | recall CN/MCI/AD |
| --- | --- | --- | --- | --- | --- | --- |
| AIBL heldout subject | 216 | 0.903 | 0.833 | 0.937 | 1.000 | 0.961/0.686/0.852 |
| IXI healthy subject | 581 | 1.000 | 1.000 | NA | CN retention 1.000 | 1.000/0.000/0.000 |
| OASIS stress subject | 99 | 0.586 | 0.334 | 0.554 | 0.371 | 0.966/0.034/0.000 |
| Internal subject | 96 | 0.458 | 0.448 | 0.719 | 0.921 | 0.241/0.553/0.550 |

## Reviewer-Safe Claim

The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance and IXI healthy specificity. The old attention-only CAS and direct Braak claims should be removed; the biological claim should be limited to an atlas-region MRI neurodegeneration proxy. OASIS transfer and internal calibration remain explicit limitations.

## Manuscript Actions

- Lead the Results with the final AIBL subject-level endpoint, bootstrap intervals, and confusion matrix.
- State that IXI is a healthy negative-control specificity analysis.
- Keep OASIS as an honest stress-test limitation.
- Replace CAS/Braak language with atlas structural neurodegeneration consistency language.
- Include MCI/AD error analysis and avoid clinical-deployment claims.
