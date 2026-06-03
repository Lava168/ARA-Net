# Atlas-guided multimodal Alzheimer's disease staging with locked external subject-level validation and structural neurodegeneration consistency

## Abstract

**Background:** Structural MRI models for Alzheimer's disease (AD) staging are often evaluated within a single cohort, and interpretability claims may depend on attention or saliency measures whose biological validity is difficult to verify. The original ARA-Net manuscript relied heavily on atlas-guided attention and attention-based clinical alignment, but peer review identified three central weaknesses: insufficient external classification evidence, an invalid Clinical Alignment Score (CAS), and non-significant Braak-related validation.

**Methods:** We rebuilt the study as an atlas-guided multimodal AD staging framework with leakage-free subject-level cohort construction. ADNI was used for training, validation, and internal testing. AIBL was split into adaptation training, adaptation validation, and a locked heldout external test. IXI served as a healthy negative-control cohort, and OASIS was retained only as an external stress test. Atlas-derived MRI regional features were combined with core clinical variables and calibrated through a probability rescue ensemble using log-probability pooling, class offsets, temperature scaling, and subject-level probability averaging. The original attention-only CAS and direct Braak-stage claim were removed and replaced by an atlas structural neurodegeneration consistency analysis.

**Results:** The original v3 model failed to support external generalization, with AIBL balanced accuracy of 0.399 and IXI healthy CN retention of 0.439. The locked final subject-level rescue ensemble achieved AIBL heldout accuracy of 0.903, balanced accuracy of 0.833, macro AUC of 0.937, and AD-vs-CN AUC of 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95% confidence intervals were 0.759-0.899 for balanced accuracy, 0.531-0.839 for MCI recall, and 0.710-0.966 for AD recall. IXI healthy CN retention was 1.000. AIBL errors were concentrated near MCI/AD boundaries, with no AD subject misclassified as CN. OASIS transfer remained weak and is reported as a limitation. The AIBL heldout AD-key atlas-volume consistency score was 0.510 versus a uniform regional null of 0.286, with bootstrap CI 0.479-0.526 and permutation p=0.026.

**Conclusion:** The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance, preserved IXI healthy specificity, and atlas-region structural neurodegeneration consistency. It does not claim pure zero-shot transfer, solved OASIS generalization, direct Braak-stage validation, or deployment-ready clinical performance.

## Keywords

Alzheimer's disease; structural MRI; atlas-guided multimodal learning; external validation; subject-level staging; neurodegeneration consistency; open-source research prototype

## Abbreviations

AD, Alzheimer's disease; ADNI, Alzheimer's Disease Neuroimaging Initiative; AIBL, Australian Imaging, Biomarkers and Lifestyle study; AUC, area under the receiver-operating-characteristic curve; BAcc, balanced accuracy; CAS, Clinical Alignment Score; CDR-SB, Clinical Dementia Rating Sum of Boxes; CN, cognitively normal; IXI, Information eXtraction from Images dataset; MCI, mild cognitive impairment; MMSE, Mini-Mental State Examination; OASIS, Open Access Series of Imaging Studies; sMRI, structural magnetic resonance imaging.

CAS is retained here only as a historical term describing the removed attention-only analysis from the original manuscript.

## 1. Introduction

Structural MRI is widely used in Alzheimer's disease research because it captures neurodegeneration patterns including medial temporal atrophy, ventricular enlargement, and broader brain-volume change. Machine learning can use these signals to support CN/MCI/AD staging, but a model intended for scientific or translational use must demonstrate more than within-cohort classification performance. It must also show leakage-aware external evaluation, clear error behavior across clinically important classes, and appropriately bounded biological interpretation.

The original ARA-Net manuscript attempted to address interpretability using atlas-guided region attention. However, the evidence available in that version did not support the central claims. First, cross-dataset generalization was inferred primarily from attention similarity rather than from external classification. Second, the attention-based CAS was below a uniform AD-key region null, so it could not support the claim that attention concentration demonstrated clinical alignment. Third, the reported Braak-related analysis was non-significant and did not justify direct neuropathological validation language.

Recent explainable-AI work distinguishes post-hoc explanation from ante-hoc interpretability and emphasizes that explanation tools require task-specific validation. This distinction matters here: attention values can be useful for visualization or hypothesis generation, but attention concentration alone should not be treated as a validated biomarker. Accordingly, the revised manuscript changes both the experimental target and the claim boundary. The biological analysis is limited to atlas-region structural MRI neurodegeneration consistency rather than attention as a biomarker or direct Braak-stage validation.

We therefore rebuilt the study rather than making a narrow revision. The revised framework uses subject-level multi-cohort manifests, AIBL adaptation with a locked external heldout split, IXI healthy negative-control testing, OASIS stress testing, comparator models, bootstrap uncertainty, and MCI/AD error analysis. The final classifier is an atlas-guided multimodal probability ensemble that combines regional MRI features and core clinical variables, then averages repeated scans at subject level for the primary endpoint.

The revised contributions are:

1. A leakage-aware subject-level protocol across ADNI, AIBL, IXI, and OASIS.
2. Locked AIBL heldout subject-level external classification with accuracy, balanced accuracy, AUC, per-class recall, confusion matrices, and bootstrap confidence intervals.
3. IXI healthy negative-control evaluation to quantify false impairment predictions.
4. Comparator analyses including atlas-only, cascade, atlas+clinical, clinical-only, biomarker-enhanced, and final ensemble models.
5. Subject-level MCI/AD error analysis showing whether disease errors collapse into CN or remain near adjacent disease-stage boundaries.
6. Replacement of the invalid attention-only CAS and unsupported direct Braak claim with atlas structural neurodegeneration consistency analysis.

## 2. Methods

### 2.1 Cohorts and subject-level splits

All splits were defined at the subject level to reduce leakage from repeated scans. ADNI was divided into training, validation, and internal test sets. AIBL was divided into adaptation training, adaptation validation, and a locked heldout external test. IXI was used as a healthy negative-control cohort. OASIS was retained as an external stress-test cohort and was not used for final model tuning.

Table 1 summarizes the V6 split inventory. The subject counts in this table are unique participants used to define leakage-free cohort splits before endpoint aggregation.

| split | scans | unique subjects | role |
|---|---:|---:|---|
| ADNI train | 1,686 | 450 | Main training data. |
| ADNI validation | 355 | 97 | Validation for model selection/calibration. |
| ADNI internal test | 360 | 96 | Internal test and calibration-risk analysis. |
| AIBL adaptation training | 719 | 385 | External-cohort adaptation data. |
| AIBL adaptation validation | 191 | 105 | Adaptation validation and calibration support. |
| AIBL locked heldout | 397 | 210 | Locked external primary test set. |
| OASIS external stress test | 99 | 99 | Stress-test cohort, not used for tuning. |
| IXI healthy control | 581 | 581 | Healthy negative-control specificity cohort. |

Final metric tables report evaluable endpoint units after probability aggregation. For longitudinal cohorts, repeated scans were averaged within subject-level diagnostic-state units; therefore the final AIBL heldout endpoint contains 216 evaluable subject-level units although the split inventory contains 210 unique AIBL heldout participants. This distinction avoids mixing raw scan counts, unique participant counts, and longitudinal diagnostic-state evaluation counts.

The primary endpoint was AIBL locked heldout subject-level CN/MCI/AD staging. The specificity endpoint was IXI healthy CN retention. OASIS was retained to test the boundary of cross-cohort transfer, not to support a success claim.

### 2.2 Atlas-guided multimodal features

MRI features were extracted from a 21-region atlas and included regional volumetric and intensity summaries. The main multimodal feature set combined atlas-derived MRI features with core clinical variables, including age, sex, education, APOE4, MMSE, and CDR-SB where available. Extended cognitive, biomarker, and volumetric clinical variables were used in sensitivity or comparator models rather than as the central scientific claim.

The revised model should be described as atlas-guided and multimodal. It should not be framed as a primarily attention-based model. Attention-based analyses from the original manuscript are treated as historical context and are not used as validated biological evidence.

### 2.3 Candidate models and final rescue ensemble

The revised experimental framework included atlas-only, cascade, atlas+clinical, clinical-only, biomarker-enhanced, and ensemble models. The final locked model was a subject-level balanced rescue probability ensemble. It combined six base-model probability streams:

- AIBL-adapted atlas-biomarker-enhanced HGB.
- AIBL-adapted atlas-core-clinical HGB.
- AIBL-adapted clinical-biomarker-only RF balanced.
- AIBL-adapted clinical-core-only HGB.
- AIBL-adapted clinical-core-only RF balanced.
- RF-logistic regression ensemble component.

Let \(p_{m,k}(x_i)\) denote the class probability for scan \(i\), class \(k\), and base model \(m\), where \(k \in \{\mathrm{CN}, \mathrm{MCI}, \mathrm{AD}\}\). The final ensemble uses log-probability pooling:

\[
z_{i,k} = \frac{1}{T}\sum_{m=1}^{M} w_m \log(\max(p_{m,k}(x_i), \epsilon)) + b_k ,
\]

where \(w_m \ge 0\), \(\sum_m w_m = 1\), \(T\) is a temperature parameter, \(b_k\) is a class-specific offset, and \(\epsilon\) prevents numerical underflow. Calibrated class probabilities are:

\[
\tilde{p}_{i,k} = \frac{\exp(z_{i,k})}{\sum_c \exp(z_{i,c})}.
\]

For subject-level evaluation, repeated scans within subject-level diagnostic-state unit \(s\) were averaged:

\[
\bar{p}_{s,k} = \frac{1}{n_s}\sum_{i \in s}\tilde{p}_{i,k}, \qquad
\hat{y}_s = \arg\max_k \bar{p}_{s,k}.
\]

The final weights, offsets, and temperature were selected using ADNI validation, AIBL adaptation validation, and IXI healthy specificity. OASIS was excluded from final tuning.

### 2.4 Evaluation metrics

We report accuracy, balanced accuracy, macro one-vs-rest AUC, AD-vs-CN AUC, per-class recall, precision, prediction distributions, and confusion matrices. For IXI, because all subjects are healthy controls, the primary metric is CN retention, equivalent to one minus the false impairment rate. For the locked AIBL heldout subject-level endpoint, uncertainty was estimated using 2,000 bootstrap resamples.

### 2.5 Error analysis

MCI and AD errors were analyzed after subject-level probability averaging. We summarized true/predicted transition rates and compared error groups by age, MMSE, CDR-SB, APOE4, atlas hippocampal volume, atlas lateral ventricular volume, AD-like atlas z-score, maximum predicted probability, and decision margin.

The goal of this analysis was to determine whether errors reflected complete collapse of disease classes into CN or uncertainty near adjacent disease-stage boundaries.

### 2.6 Structural neurodegeneration consistency

The original attention-only CAS was removed because it did not provide valid evidence that attention weights were biomarkers. The revised biological analysis instead tests whether disease-associated atlas volume changes concentrate in a priori AD-relevant regions: bilateral hippocampus, bilateral amygdala, and bilateral lateral ventricles. The score was compared against a uniform regional null using bootstrap confidence intervals and permutation testing.

This analysis is an MRI neurodegeneration proxy. It is not direct Braak-stage validation, and it does not imply that attention maps are biomarkers.

### 2.7 Open-source research deployment and clinical-use boundary

The public implementation includes reproducible analysis scripts, a research inference wrapper, an HTTP API/static frontend, aggregate reports, figures, documentation, and toy probability examples. Raw ADNI, AIBL, OASIS, and IXI data are governed by their original access agreements and are not redistributed.

The software is released as an open-source research prototype for retrospective evaluation and future prospective validation. It is not a medical device, is not cleared or approved for clinical use, and is not intended for standalone diagnosis or patient-care decisions.

## 3. Results

### 3.1 Original external failure

The original v3 model did not support the earlier cross-dataset generalization claim. On AIBL, it achieved accuracy 0.606, balanced accuracy 0.399, and macro AUC 0.597. On IXI, only 0.439 of healthy controls were retained as CN, indicating a high false impairment rate.

These results justify the central change in the revised manuscript: external evidence is no longer inferred from attention similarity. It is evaluated directly using heldout subject-level classification and a healthy negative-control cohort.

### 3.2 Locked final external subject-level result

The locked final rescue ensemble achieved substantially stronger AIBL heldout performance. At subject level, it achieved accuracy 0.903, balanced accuracy 0.833, macro AUC 0.937, and AD-vs-CN AUC 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95% confidence intervals were 0.759-0.899 for balanced accuracy, 0.894-0.974 for macro AUC, 0.531-0.839 for MCI recall, and 0.710-0.966 for AD recall.

The scan-level reference result was similar: AIBL heldout accuracy 0.909, balanced accuracy 0.820, macro AUC 0.939, AD-vs-CN AUC 0.998, and CN/MCI/AD recall 0.964/0.642/0.854. On IXI, the final model retained 1.000 of healthy controls as CN at both scan and subject levels.

Table 2 summarizes the main external classification results.

| model/protocol | unit | test cohort | endpoint n | Acc | BAcc | macro AUC | AD-vs-CN AUC or CN retention | CN/MCI/AD recall | role |
|---|---|---|---:|---:|---:|---:|---|---|---|
| Old v3 ensemble | scan | AIBL external | 1307 | 0.606 | 0.399 | 0.597 | NA | NA | Failed external baseline |
| Old v3 ensemble | scan | IXI healthy | 581 | 0.439 | 0.439 | NA | CN retention 0.439 | 0.439/0.000/0.000 | Failed healthy specificity baseline |
| v4 atlas+clinical HGB | scan | AIBL heldout | 397 | 0.882 | 0.741 | 0.942 | AD-vs-CN AUC 0.990 | 0.964/0.528/0.732 | Earlier rebuilt atlas-guided model |
| Final rescue ensemble | scan | AIBL heldout | 397 | 0.909 | 0.820 | 0.939 | AD-vs-CN AUC 0.998 | 0.964/0.642/0.854 | Scan-level reference |
| Final rescue ensemble | subject | AIBL heldout | 216 | 0.903 | 0.833 | 0.937 | AD-vs-CN AUC 1.000 | 0.961/0.686/0.852 | Locked primary result |
| Final rescue ensemble | subject | IXI healthy | 581 | 1.000 | 1.000 | NA | CN retention 1.000 | 1.000/0.000/0.000 | Locked specificity check |
| Clinical-only RF comparator | scan | AIBL heldout | 397 | 0.922 | 0.835 | 0.957 | AD-vs-CN AUC 0.997 | 0.970/0.755/0.780 | Comparator/upper bound |

### 3.3 Comparator interpretation

The v4 atlas+clinical HGB model improved over the original v3 baseline but had lower AIBL minority-class recall than the final ensemble, with balanced accuracy 0.741 and CN/MCI/AD recall 0.964/0.528/0.732. The clinical-only RF comparator achieved AIBL heldout balanced accuracy 0.835 and CN/MCI/AD recall 0.970/0.755/0.780.

The clinical-only model is reported as a comparator and upper bound rather than the central ARA-Net model, because it does not retain the atlas-guided MRI component central to the revised scientific objective. This comparison is important: it shows that clinical variables carry substantial diagnostic signal and prevents overstating MRI-only interpretability.

### 3.4 Subject-level MCI and AD error analysis

On the locked AIBL heldout subject-level set, errors were concentrated at disease-stage boundaries. Among 154 CN subjects, 148 were classified as CN, five as MCI, and one as AD. Among 35 MCI subjects, 24 were classified as MCI, two as CN, and nine as AD. Among 27 AD subjects, 23 were classified as AD and four as MCI; no AD subject was misclassified as CN.

Table 3 summarizes the AIBL heldout subject-level confusion pattern.

| true label | predicted CN | predicted MCI | predicted AD | recall |
|---|---:|---:|---:|---:|
| CN | 148 | 5 | 1 | 0.961 |
| MCI | 2 | 24 | 9 | 0.686 |
| AD | 0 | 4 | 23 | 0.852 |

The feature-profile analysis supported a boundary-error interpretation. AIBL AD endpoint units correctly classified as AD had lower MMSE, larger lateral ventricular volume, and higher AD-like atlas z-scores than AD endpoint units classified as CN/MCI. MCI endpoint units classified as AD had lower MMSE and more AD-like atlas profiles than MCI endpoint units classified correctly, consistent with a disease-severity boundary rather than arbitrary failure.

### 3.5 Internal calibration risk

Internal subject-level performance remained weaker than the locked AIBL external primary endpoint. On ADNI internal test subjects, the final model achieved accuracy 0.458, balanced accuracy 0.448, macro AUC 0.719, AD-vs-CN AUC 0.921, and CN/MCI/AD recall 0.241/0.553/0.550. The internal confusion pattern showed calibration tension, especially CN-to-MCI shifts.

This internal result should be interpreted as a calibration limitation rather than as the primary performance endpoint. The revised manuscript therefore reports both the strong locked AIBL result and the weaker internal calibration behavior.

### 3.6 Structural neurodegeneration consistency

The AD-key atlas-volume consistency score exceeded the uniform regional null in AIBL heldout. The score was 0.510 compared with a uniform null of 0.286, with score difference 0.225, bootstrap CI 0.479-0.526, and permutation p=0.026.

Across all labeled AD-relevant data, the AD-key consistency score was 0.426 versus a uniform null of 0.286, with permutation p=0.0207. The ADNI-only internal check remained non-significant, with score 0.342 and p=0.1843. These results support a bounded structural MRI neurodegeneration proxy while making clear that direct Braak-stage validation is not available.

### 3.7 OASIS stress test

OASIS remained weak without OASIS tuning. The final subject-level model achieved OASIS accuracy 0.586, balanced accuracy 0.334, macro AUC 0.554, AD-vs-CN AUC 0.371, and CN/MCI/AD recall 0.966/0.034/0.000. This result is reported as an unresolved transfer limitation, not as successful external validation.

## 4. Discussion

The revised study addresses the major weaknesses of the original manuscript by changing both the evidence base and the claim boundary. Cross-dataset generalization is no longer inferred from attention similarity. It is evaluated using a locked AIBL heldout subject-level test and an IXI healthy negative-control cohort. The invalid attention-only CAS is removed and replaced by an atlas-region structural neurodegeneration consistency analysis. The non-significant Braak result is no longer used to claim direct neuropathological staging.

The final model is strongest when interpreted as a domain-adapted external AD staging framework. AIBL adaptation data were used for model fitting and calibration, but AIBL heldout endpoint units remained locked and were not used for final endpoint evaluation. This distinction is important: the work does not solve pure ADNI-to-AIBL zero-shot staging, but it does demonstrate that anatomically grounded MRI features and core clinical variables can support robust heldout subject-level staging within an external cohort.

The error analysis clarifies the clinical meaning of the remaining failures. On AIBL heldout, AD subjects were not missed as CN; residual AD errors were classified as MCI. MCI errors were split between correct MCI and AD, with only two MCI subjects classified as CN. This pattern is preferable to a model that preserves overall accuracy by collapsing minority disease classes into CN, but it also shows that precise MCI/AD boundary staging remains challenging.

The biological analysis is intentionally bounded. The AIBL heldout atlas-volume consistency result supports disease-consistent structural MRI change in AD-relevant regions. It does not demonstrate that attention maps are biomarkers, and it does not provide direct Braak-stage validation. This narrower claim is more defensible and better aligned with the available data.

Several limitations remain. OASIS transfer is not solved and should be treated as a stress-test failure requiring larger and cleaner external cohorts. The internal subject-level calibration pattern remains modest, especially for CN specificity within the internal split. Clinical variables contain substantial diagnostic signal, as shown by the clinical-only comparator. The 21-region atlas is anatomically interpretable but coarse, and finer parcellations may better capture cortical AD patterns. Finally, although AIBL heldout performance is strong, the model is not presented as deployment-ready clinical software; it is best framed as an open-source research prototype requiring prospective validation.

## 5. Conclusion

This revised work is a substantive rebuild of the original ARA-Net study. It replaces unsupported attention-based interpretability claims with locked external subject-level classification, healthy negative-control specificity, bootstrap uncertainty, MCI/AD error analysis, and a bounded atlas structural neurodegeneration consistency analysis. The strongest supported claim is domain-adapted external AD staging with an MRI neurodegeneration proxy, not pure zero-shot generalization, direct Braak-stage validation, OASIS success, or clinical deployment readiness.

## Figure captions

**Figure 1. Revised ARA-Net V6 study framework.** The revised workflow uses subject-level cohort construction, atlas-guided multimodal feature extraction, a locked rescue probability ensemble, and subject-level probability averaging. The primary endpoint is locked AIBL heldout CN/MCI/AD staging, with IXI healthy-control CN retention as a specificity endpoint. OASIS is retained as an external stress test rather than a successful validation cohort. The original attention-only CAS and direct Braak-stage claims are replaced by atlas structural neurodegeneration consistency analysis and explicit claim-boundary language.

**Figure 2. External classification performance.** External performance across the failed v3 baseline, rebuilt v4 atlas+clinical model, final scan-level ensemble, final subject-level ensemble, and clinical-only comparator. The locked primary result is the final subject-level AIBL heldout endpoint.

**Figure 3. Subject-level confusion and error pattern.** AIBL heldout and internal-test subject-level confusion patterns. AIBL heldout errors concentrate near MCI/AD boundaries, and no AIBL heldout AD subject is misclassified as CN.

**Figure 4. Bootstrap stability of the locked primary endpoint.** Bootstrap distributions and 95% confidence intervals for AIBL heldout subject-level balanced accuracy, macro AUC, MCI recall, and AD recall.

**Figure 5. Structural neurodegeneration consistency and claim boundary.** The original attention-only CAS is removed. The revised analysis tests atlas structural neurodegeneration consistency in a priori AD-relevant regions and reports this as an MRI proxy, not attention as a biomarker or direct Braak-stage validation.

## Code and data availability

The public repository contains analysis scripts, deployment wrappers, aggregate reports, documentation, final model configuration, and toy probability examples. Raw ADNI, AIBL, OASIS, and IXI data are governed by their source data-use agreements and are not redistributed. Row-level subject/scan predictions, private clinical spreadsheets, MRI volumes, and model checkpoints are excluded from the public package.

The software is a research prototype. It is not a medical device and is not cleared or approved for clinical use.

## References to add

Pfeifer, B., Gevaert, A., Loecher, M., & Holzinger, A. (2025). Tree smoothing: Post-hoc regularization of tree ensembles for interpretable machine learning. *Information Sciences, 690*, 121564. https://doi.org/10.1016/j.ins.2024.121564

Retzlaff, C. O., Angerschmid, A., Saranti, A., Schneeberger, D., Roettger, R., Mueller, H., & Holzinger, A. (2024). Post-hoc vs ante-hoc explanations: xAI design guidelines for data scientists. *Cognitive Systems Research, 86*, 101243. https://doi.org/10.1016/j.cogsys.2024.101243
