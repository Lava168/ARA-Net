# V6 Manuscript Rewrite Package

中文结论：这一版应该把论文主线从“ARA-Net attention 模型”改成“atlas-guided multimodal AD staging with locked external subject-level validation”。最终主模型锁定为 subject-level balanced rescue probability ensemble；旧 v4 atlas+clinical HGB 作为 rebuilt baseline；clinical-only RF 作为 comparator / upper bound；OASIS 只作为 limitation。

## Recommended Title

Atlas-guided multimodal Alzheimer's disease staging with locked external subject-level validation and structural neurodegeneration consistency

## Abstract

**Background:** Structural MRI models for Alzheimer's disease (AD) staging are often evaluated within a single cohort, and their interpretability claims may rely on attention or saliency measures whose biological validity is difficult to verify. We substantially rebuilt the original atlas-guided ARA-Net study to test whether an anatomically grounded multimodal framework can provide stronger external subject-level AD staging while preserving healthy-control specificity.

**Methods:** We constructed leakage-free subject-level splits across ADNI, AIBL, OASIS, and IXI. ADNI was used for training, validation, and internal testing. AIBL was divided into adaptation training, adaptation validation, and a locked heldout external test. IXI served as a healthy negative-control cohort. OASIS was retained only as an external stress test. Atlas-derived MRI regional features were combined with core clinical variables and calibrated by a log-probability ensemble with class offsets and temperature scaling. Repeated scans were averaged at subject level before the primary evaluation. Comparator models included atlas-only, cascade, atlas+clinical HGB, clinical-only RF, and biomarker-enhanced variants. The original attention-only CAS and direct Braak claim were replaced by an atlas-region neurodegeneration consistency analysis.

**Results:** The original v3 model failed external validation, with AIBL balanced accuracy of 0.399 and IXI CN retention of 0.439. The final locked subject-level rescue ensemble achieved AIBL heldout accuracy of 0.903, balanced accuracy of 0.833, macro AUC of 0.937, and AD-vs-CN AUC of 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95% confidence intervals were 0.759-0.899 for balanced accuracy, 0.531-0.839 for MCI recall, and 0.710-0.966 for AD recall. The same model retained all IXI healthy controls as CN. AIBL heldout errors were concentrated near MCI/AD boundaries: no AD endpoint unit was misclassified as CN. OASIS transfer remained poor and is reported as a limitation. The AD-key volume consistency score in AIBL heldout was 0.510 versus a uniform regional null of 0.286, with bootstrap CI [0.479, 0.526] and permutation p=0.026.

**Conclusion:** The revised work provides substantially stronger evidence for domain-adapted external subject-level AD staging, healthy negative-control specificity, and atlas-region neurodegeneration consistency. It does not claim pure zero-shot transfer, solved OASIS generalization, direct Braak staging, or deployment-ready clinical performance.

## Introduction

Structural MRI is central to AD research because it captures neurodegeneration patterns including medial temporal atrophy and ventricular enlargement. Machine learning can use these patterns to support CN/MCI/AD staging, but three issues limit clinical and scientific credibility: inadequate cross-cohort validation, weak interpretability validation, and overstatement of neuropathological alignment when direct Braak-stage labels are unavailable.

The original ARA-Net manuscript attempted to address interpretability using atlas-guided attention. However, the original evidence was insufficient in three fundamental ways. First, cross-dataset generalization was inferred largely from attention similarity, not external classification performance. Second, the Clinical Alignment Score was below the uniform AD-key region null, so it could not support the claimed clinical alignment. Third, the reported Braak correlation was non-significant, making direct neuropathological interpretation unsupported.

We therefore rebuilt the study rather than making a narrow revision. The revised framework uses explicit subject-level multi-cohort manifests, AIBL adaptation with a locked external heldout split, IXI healthy negative-control testing, OASIS stress testing, and multiple comparator models. The final classifier is an atlas-guided multimodal probability ensemble that combines regional MRI features and core clinical variables, then averages repeated scans at subject level for the primary endpoint.

The revised contributions are:

1. A leakage-free subject-level protocol across ADNI, AIBL, IXI, and OASIS.
2. Locked AIBL heldout subject-level external classification with per-class recall, AUC, confusion matrices, and bootstrap confidence intervals.
3. IXI healthy negative-control evaluation to measure false impairment predictions.
4. Comparator analyses including atlas-only, clinical-only, and biomarker-enhanced models.
5. A replacement of the invalid attention-only CAS with an atlas-region neurodegeneration consistency score.
6. A revised biological interpretation that avoids direct Braak-stage claims and instead reports an MRI neurodegeneration proxy.

## Methods

### Cohorts And Splits

All data splits were defined at the subject level. ADNI was divided into training, validation, and internal test sets. AIBL was divided into adaptation training, adaptation validation, and a locked heldout external test. IXI was used as a healthy negative-control cohort. OASIS was retained as a stress-test cohort and was not used for final model tuning.

The resulting split sizes were: ADNI train 1,686 scans from 450 subjects; ADNI validation 355 scans from 97 subjects; ADNI internal test 360 scans from 96 subjects; AIBL adaptation training 719 scans from 385 subjects; AIBL adaptation validation 191 scans from 105 subjects; AIBL locked heldout 397 scans from 210 subjects; OASIS external stress test 99 scans from 99 subjects; and IXI healthy control 581 scans from 581 subjects.

### Atlas-Guided Multimodal Features

MRI features were extracted from a 21-region atlas and included regional volumes and intensity summaries. The main multimodal feature set combined atlas-derived MRI features with core clinical variables including age, sex, education, APOE4, MMSE, and CDR-SB where available. Extended cognitive, biomarker, and volumetric clinical variables were used in sensitivity or comparator models rather than as the central claim.

### Final Rescue Ensemble

Let \(p_{m,k}(x_i)\) be the predicted probability for class \(k\) from base model \(m\) for scan \(i\), where \(k \in \{\mathrm{CN}, \mathrm{MCI}, \mathrm{AD}\}\). The final ensemble combines base-model probabilities by log-probability pooling:

\[
z_{i,k} = \frac{1}{T}\sum_{m=1}^{M} w_m \log(\max(p_{m,k}(x_i), \epsilon)) + b_k ,
\]

where \(w_m \ge 0\), \(\sum_m w_m = 1\), \(T\) is a temperature parameter, \(b_k\) is a class-specific offset, and \(\epsilon\) avoids numerical underflow. Calibrated class probabilities are then:

\[
\tilde{p}_{i,k} = \frac{\exp(z_{i,k})}{\sum_{c}\exp(z_{i,c})}.
\]

For the primary subject-level analysis, repeated scans for subject \(s\) were averaged:

\[
\bar{p}_{s,k} = \frac{1}{n_s}\sum_{i \in s}\tilde{p}_{i,k}, \qquad
\hat{y}_s = \arg\max_k \bar{p}_{s,k}.
\]

Ensemble weights, class offsets, and temperature were selected using ADNI validation, AIBL adaptation validation, and IXI healthy specificity. OASIS was not used for final tuning. Final metric tables report evaluable subject-level endpoint units after repeated-scan probability aggregation; in longitudinal cohorts, these endpoint units are separate from the unique-participant split inventory used to prevent leakage.

### Evaluation

We report accuracy, balanced accuracy, macro one-vs-rest AUC, AD-vs-CN AUC, per-class recall and precision, prediction distribution, and confusion matrices. For IXI, because all subjects are healthy controls, the primary metric is CN retention, equivalent to one minus the false impairment rate. For the locked final AIBL heldout subject-level result, uncertainty was estimated using 2,000 bootstrap resamples.

### Error Analysis

MCI and AD errors were analyzed at the subject level after averaging repeated scans. We summarized true/predicted transition rates and compared error groups by age, MMSE, CDR-SB, APOE4, atlas hippocampal volume, atlas lateral ventricular volume, AD-like atlas z-score, maximum predicted probability, and decision margin.

### Biological Consistency Analysis

The original attention-only CAS was removed. We instead evaluated whether disease-associated atlas volume changes concentrated in a priori AD-relevant regions: bilateral hippocampus, bilateral amygdala, and bilateral lateral ventricles. The score was compared against a uniform regional null using bootstrap confidence intervals and permutation testing. This analysis is described as MRI neurodegeneration-proxy validation, not direct Braak staging.

## Results

### Original External Failure

The original v3 ensemble did not support the original cross-dataset generalization claim. On AIBL, it achieved accuracy 0.606, balanced accuracy 0.399, and macro AUC 0.597. On IXI, only 0.439 of healthy controls were retained as CN, indicating a high false-impairment rate.

### Final Locked External Subject-Level Result

The locked final rescue ensemble achieved substantially stronger AIBL heldout performance. At subject level, it achieved accuracy 0.903, balanced accuracy 0.833, macro AUC 0.937, and AD-vs-CN AUC 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95% confidence intervals were 0.759-0.899 for balanced accuracy, 0.894-0.974 for macro AUC, 0.531-0.839 for MCI recall, and 0.710-0.966 for AD recall.

The scan-level reference result was similar: AIBL heldout accuracy 0.909, balanced accuracy 0.820, macro AUC 0.939, AD-vs-CN AUC 0.998, and CN/MCI/AD recall 0.964/0.642/0.854. On IXI, the final ensemble retained 1.000 of healthy controls as CN at both scan and subject levels.

### Comparator Models

The v4 atlas+clinical HGB model improved over the original v3 baseline but had lower AIBL minority-class recall than the final ensemble: balanced accuracy 0.741 and CN/MCI/AD recall 0.964/0.528/0.732. The clinical-only RF comparator achieved AIBL heldout balanced accuracy 0.835 and CN/MCI/AD recall 0.970/0.755/0.780. We report the clinical-only model as a comparator and upper bound because it does not retain the atlas-guided MRI component central to the revised scientific objective.

### MCI And AD Error Analysis

On the locked AIBL heldout subject-level set, errors were concentrated at disease-stage boundaries. Among 154 CN endpoint units, 148 were classified as CN, five as MCI, and one as AD. Among 35 MCI endpoint units, 24 were classified as MCI, two as CN, and nine as AD. Among 27 AD endpoint units, 23 were classified as AD and four as MCI; no AD endpoint unit was misclassified as CN.

The feature-profile analysis supported this boundary interpretation. AIBL AD endpoint units correctly classified as AD had lower MMSE, larger lateral ventricular volume, and higher AD-like atlas z-scores than AD endpoint units classified as CN/MCI. MCI endpoint units classified as AD had lower MMSE and more AD-like atlas profiles than MCI endpoint units classified correctly, consistent with a disease-severity boundary rather than arbitrary failure.

### Biological Consistency

The AD-key volume consistency score exceeded the uniform regional null in AIBL heldout. The score was 0.510 compared with a uniform null of 0.286, with score difference 0.225, bootstrap CI [0.479, 0.526], and permutation p=0.026. This supports disease-consistent structural MRI change in AD-relevant regions, while avoiding the unsupported claim that attention mass alone is a validated biomarker.

### OASIS Stress Test

OASIS remained weak without OASIS tuning. The final subject-level model achieved OASIS accuracy 0.586, balanced accuracy 0.334, macro AUC 0.554, and CN/MCI/AD recall 0.966/0.034/0.000. This result is reported as an unresolved transfer limitation, not as successful external validation.

## Discussion

The revised study addresses the major weaknesses of the original manuscript by changing both the experimental evidence and the claim boundary. First, cross-dataset generalization is no longer inferred from attention similarity. It is evaluated using a locked AIBL heldout subject-level test and an IXI healthy negative-control cohort. Second, the invalid attention-only CAS is removed and replaced by an atlas-region structural neurodegeneration consistency analysis. Third, the non-significant Braak result is no longer used to claim direct neuropathological staging.

The final model is strongest when interpreted as a domain-adapted external AD staging framework. AIBL adaptation data were used for model fitting and calibration, but AIBL heldout endpoint units remained locked and were not used for final evaluation. This distinction is important: the work does not solve pure ADNI-to-AIBL zero-shot staging, but it does demonstrate that anatomically grounded MRI features and core clinical variables can support robust heldout subject-level staging within an external cohort.

The error analysis also clarifies the clinical meaning of the remaining failures. On AIBL heldout, AD subjects were not missed as CN; residual AD errors were classified as MCI. MCI errors were mostly split between correct MCI and AD, with only two MCI subjects classified as CN. This pattern is preferable to a model that preserves overall accuracy by collapsing minority disease classes into CN, but it also shows that precise MCI/AD boundary staging remains challenging.

Several limitations remain. OASIS transfer is not solved and should be treated as a stress-test failure requiring larger and cleaner external cohorts. The biological analysis is an MRI neurodegeneration proxy rather than direct Braak staging. Clinical variables contain substantial diagnostic signal, as shown by the clinical-only comparator. The 21-region atlas is anatomically interpretable but coarse, and finer parcellations may better capture cortical AD patterns. Finally, although AIBL heldout performance is strong, the model is not presented as deployment-ready clinical software; it is best framed as decision-support research requiring prospective validation.

## Cover Letter Core Paragraph

In response to the previous decision, we did not attempt a narrow revision of the original manuscript. Instead, we rebuilt the experimental framework and substantially revised the central claims. The revised manuscript now includes leakage-free subject-level manifests, AIBL adaptation with a locked external heldout test, IXI healthy negative-control specificity testing, OASIS stress-test reporting, multiple comparator models, subject-level probability averaging, bootstrap uncertainty, MCI/AD error analysis, and a replacement of the original attention-only CAS/Braak claims with an empirically tested atlas-region neurodegeneration consistency analysis. The locked final subject-level ensemble achieved AIBL heldout balanced accuracy of 0.833, macro AUC of 0.937, AD-vs-CN AUC of 1.000, CN/MCI/AD recall of 0.961/0.686/0.852, and IXI CN retention of 1.000. We explicitly report that OASIS transfer remains unresolved and avoid claims of pure zero-shot transfer, direct Braak staging, or deployment-ready clinical performance.
