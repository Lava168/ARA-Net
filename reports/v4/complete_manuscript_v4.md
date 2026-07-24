# Complete V4 Manuscript Draft

This draft is assembled from the v4 rebuild package. It is intended as the working source for the Word manuscript rewrite, not as the final formatted journal file.

## Claim Boundaries For The Rewrite

Allowed claims:

- The revised atlas-guided multimodal model achieved strong domain-adapted performance on a locked AIBL heldout split.
- IXI was used as a healthy external negative-control cohort to estimate false impairment.
- The original attention-only CAS was replaced by a structural atlas neurodegeneration consistency score.
- The biological validation supports an MRI neurodegeneration proxy, not direct Braak staging.

Claims to avoid:

- Do not claim pure ADNI-to-AIBL zero-shot staging is solved.
- Do not claim direct Braak-stage validation.
- Do not claim attention mass alone is a validated biomarker.
- Do not hide or soften the weak OASIS transfer result.
- Do not frame the model as ready for clinical deployment.

## Manuscript Text

# Atlas-guided multimodal Alzheimer disease staging with external heldout validation and neurodegeneration-consistent regional biomarkers

## Abstract

### Background

Structural MRI models for Alzheimer's disease (AD) are often evaluated within a single dataset and explained using post-hoc or attention-based measures whose biological validity is difficult to verify. The original ARA-Net study used atlas-guided attention for interpretable CN/MCI/AD classification, but external classification, attention-based clinical alignment, and Braak-related validation were insufficiently supported.

### Methods

We rebuilt the framework as a cross-cohort atlas-guided multimodal AD staging pipeline. Subject-level splits were constructed across ADNI, AIBL, OASIS, and IXI. ADNI was divided into training, validation, and internal test sets; AIBL was divided into adaptation training, adaptation validation, and a locked heldout split; OASIS was retained as an external stress-test cohort; and IXI was used as a healthy external negative-control cohort. Atlas-derived MRI regional features were combined with core clinical variables in a clinically adapted classifier. Comparator models included MRI/atlas-only, cascade, clinical-only, and biomarker-enhanced models. The original attention-based CAS was replaced by an atlas neurodegeneration consistency score based on AD-relevant regional volume changes.

### Results

The original v3 model failed external validation, with AIBL balanced accuracy of 0.399 and IXI healthy CN retention of 0.439. The revised atlas-guided multimodal model achieved AIBL heldout accuracy of 0.882, balanced accuracy of 0.741, macro AUC of 0.942, and AD-vs-CN AUC of 0.990. CN/MCI/AD recall was 0.964/0.528/0.732. On IXI, the same model retained 0.998 of healthy scans as CN. These results were reproduced across four seeds. A clinical-only comparator achieved AIBL heldout balanced accuracy of 0.830 +/- 0.003, indicating the strength of clinical variables but serving as a comparator rather than the main atlas-guided model. The AD-key volume consistency score on AIBL heldout was 0.510 versus a uniform null of 0.286, with bootstrap CI [0.479, 0.526] and permutation p=0.026.

### Conclusion

The revised study provides evidence for domain-adapted external heldout performance, healthy-control specificity, and atlas-region neurodegeneration consistency. It does not claim pure zero-shot ADNI-to-AIBL staging or direct Braak validation. OASIS transfer remains unresolved and is reported as a limitation.

## 1. Introduction

Structural MRI is central to AD research because it captures neurodegeneration patterns such as medial temporal atrophy and ventricular expansion. Machine learning models can exploit these patterns for CN/MCI/AD classification, but three limitations restrict their clinical and scientific utility. First, many models are evaluated only within a single cohort and therefore do not demonstrate cross-cohort robustness. Second, attention or saliency maps are often interpreted as biological evidence without validating whether the highlighted regions behave consistently with disease mechanisms. Third, claims about Braak staging or neuropathological alignment are often stronger than the available MRI labels can support.

The previous ARA-Net manuscript attempted to address interpretability by using atlas-guided region attention. However, peer review identified three fundamental issues. External classification was not reported despite cross-dataset generalization being a core claim. The proposed Clinical Alignment Score (CAS) was below the uniform AD-key region null, making it unsuitable as evidence of clinical alignment. The reported Braak correlation was non-significant, undermining the biological validation claim.

We therefore rebuilt the study rather than making a narrow revision. The revised work asks a different and more testable question: can an atlas-guided multimodal framework combine anatomically grounded MRI information with core clinical variables to achieve robust external heldout performance while preserving healthy-control specificity and producing disease-consistent regional MRI patterns?

The contributions are:

1. A leakage-free subject-level multi-cohort experimental protocol across ADNI, AIBL, OASIS, and IXI.
2. External heldout classification on AIBL with explicit per-class recall, AUC, and confusion behavior.
3. IXI healthy negative-control testing to quantify false impairment in cognitively normal healthy subjects.
4. MRI/atlas-only, atlas+clinical, clinical-only, cascade, and biomarker-enhanced comparator models.
5. A replacement of the original attention-only CAS with an atlas neurodegeneration consistency score.
6. A revised biological interpretation that avoids unsupported direct Braak staging and instead evaluates an MRI neurodegeneration proxy.

## 2. Materials and Methods

### 2.1 Cohorts

The revised analysis uses ADNI, AIBL, OASIS, and IXI. ADNI provides the main training and internal testing data. AIBL provides an external cohort with adaptation and locked heldout splits. OASIS is retained as an external stress-test cohort. IXI is used as a healthy external negative-control cohort because its scans should be retained as CN by a clinically usable model.

### 2.2 Subject-level splits

All splits were assigned at the subject level. This prevents the same subject from appearing in both training and evaluation sets. The final split sizes are:

| Split | Scans | Subjects | CN | MCI | AD | Role |
|---|---:|---:|---:|---:|---:|---|
| train | 1686 | 450 | 514 | 819 | 353 | ADNI training |
| val | 355 | 97 | 125 | 166 | 64 | ADNI model selection |
| internal_test | 360 | 96 | 111 | 171 | 78 | ADNI internal test |
| aibl_adapt_train | 719 | 385 | 536 | 105 | 78 | AIBL adaptation training |
| aibl_adapt_val | 191 | 105 | 147 | 25 | 19 | AIBL adaptation validation |
| aibl_heldout | 397 | 210 | 303 | 53 | 41 | Locked AIBL heldout external test |
| oasis_external | 99 | 99 | 59 | 29 | 11 | External stress test |
| ixi_external | 581 | 581 | 581 | 0 | 0 | Healthy negative-control test |

### 2.3 Atlas-derived MRI features

Each MRI scan was represented by atlas-derived regional features from the 21-region FreeSurfer-like parcellation used in the original ARA-Net work. For each region, we extracted relative volume and intensity summaries. We additionally computed bilateral asymmetry measures and AD-relevant aggregate features, including hippocampal volume sum, ventricle volume sum, and hippocampus-to-ventricle ratio.

### 2.4 Clinical and biomarker variables

Core clinical variables included age, sex, education, APOE4, MMSE, and CDR-SB where available. A biomarker-enhanced sensitivity feature set additionally included available CSF, amyloid, WMH, and volumetric clinical table variables. Models using biomarker-enhanced variables are interpreted as sensitivity analyses, not as the primary atlas-guided model.

### 2.5 Model protocols

We evaluated two major protocols:

1. **ADNI-only protocol:** training uses only ADNI train and model selection uses ADNI validation. This evaluates zero-shot external behavior.
2. **AIBL-adapted protocol:** training uses ADNI train plus AIBL adaptation training, model selection uses ADNI validation plus AIBL adaptation validation, and final external performance is reported on locked AIBL heldout. This evaluates domain-adapted external heldout performance.

### 2.6 Candidate models

We evaluated atlas-only models, atlas+clinical models, clinical-only models, biomarker-enhanced models, and a two-stage cascade model. The recommended atlas-guided multimodal model is `atlas_core_clinical__hgb`, a histogram gradient boosting classifier using atlas MRI features plus core clinical variables. The strongest classifier, `clinical_core_only__rf_balanced`, is reported as a clinical-only comparator and upper-bound.

### 2.7 Evaluation metrics

We report accuracy, balanced accuracy, macro one-vs-rest AUC, AD-vs-CN AUC, per-class recall, and prediction distribution. For IXI, because all scans are healthy controls, the primary metric is CN retention rate, equivalent to one minus false impairment rate.

### 2.8 Atlas neurodegeneration consistency score

The original attention-based CAS was removed because it was below the uniform AD-key region null. We instead evaluate an atlas neurodegeneration consistency score. The AD-key region set includes bilateral hippocampus, amygdala, and lateral ventricles. The score measures whether disease-associated volume changes concentrate in these regions relative to a uniform regional null. Statistical uncertainty is estimated using bootstrap confidence intervals and permutation testing.

### 2.9 Biological interpretation

We do not claim direct Braak staging validation. Instead, we interpret the regional analysis as a Braak-alternative MRI neurodegeneration proxy. It tests whether structural changes concentrate in regions consistent with known AD-related atrophy and ventricular expansion.

## 3. Results

### 3.1 Original v3 external failure

The original v3 ensemble failed to support the claimed external generalization. On AIBL, it achieved accuracy 0.606, balanced accuracy 0.399, and macro AUC 0.597. On IXI, only 0.439 of healthy scans were retained as CN, indicating a high false impairment rate. These findings motivate the rebuilt v4 experimental framework.

### 3.2 Atlas-only and cascade baselines

The atlas-only HGB model improved healthy specificity but did not solve external staging. On AIBL heldout, it achieved accuracy 0.776, balanced accuracy 0.479, macro AUC 0.732, and recall CN/MCI/AD of 0.944/0.151/0.341. IXI CN retention improved to 0.983. The cascade model achieved IXI CN retention of 1.000 but failed AIBL heldout MCI detection, with balanced accuracy 0.391 and MCI recall 0.000.

### 3.3 ADNI-only zero-shot behavior

ADNI-only hybrid models remained weak on AIBL AD detection. The best ADNI-only hybrid candidate achieved AIBL heldout balanced accuracy of 0.406 and predicted no AD cases. Therefore, the revised paper does not claim pure ADNI-to-AIBL zero-shot staging is solved.

### 3.4 Main atlas-guided multimodal model

The recommended atlas-guided multimodal model, `atlas_core_clinical__hgb`, achieved strong domain-adapted external heldout performance. On AIBL heldout, it achieved accuracy 0.882, balanced accuracy 0.741, macro AUC 0.942, and AD-vs-CN AUC 0.990. CN/MCI/AD recall was 0.964/0.528/0.732. On IXI, CN retention was 0.998, with only one healthy scan predicted as impaired.

Across four seeds, the recommended model reproduced the same AIBL heldout balanced accuracy of 0.741 +/- 0.000 and IXI CN retention of 0.998 +/- 0.000.

### 3.5 Clinical-only and biomarker-enhanced sensitivity analyses

The clinical-only random forest was the strongest classifier, achieving AIBL heldout accuracy 0.920 +/- 0.001, balanced accuracy 0.830 +/- 0.003, macro AUC 0.957 +/- 0.001, and IXI CN retention 1.000 +/- 0.000. We report this as a comparator and upper-bound because it does not retain atlas-derived MRI information. The biomarker-enhanced HGB model achieved AIBL heldout balanced accuracy 0.703 and IXI CN retention 0.997.

### 3.6 OASIS stress-test result

OASIS transfer remained weak. The recommended atlas-guided multimodal model achieved OASIS balanced accuracy 0.322. This result is reported as an unresolved external stress test rather than as successful generalization.

### 3.7 Atlas neurodegeneration consistency

On AIBL heldout, the AD-key volume score was 0.510 compared with a uniform null of 0.286. The score-minus-null difference was 0.225, with bootstrap CI [0.479, 0.526] and permutation p=0.026. In the combined AIBL adaptation and heldout evaluation, the AD-key score was 0.512, with p=0.035. Across all labeled AD data, the score was 0.426, with p=0.0207. The ADNI validation plus internal test subset alone was not significant, with p=0.1843.

## 4. Discussion

The revised study addresses the three central failures of the original manuscript. First, external classification is now explicitly evaluated. The original v3 model had weak AIBL balanced accuracy and poor IXI specificity, while the revised atlas-guided multimodal model achieves substantially stronger AIBL heldout performance and preserves healthy specificity on IXI. Second, the original attention-only CAS is replaced with an empirically tested atlas neurodegeneration consistency score that exceeds a uniform regional null on AIBL heldout. Third, the direct Braak claim is removed and replaced by a more defensible MRI neurodegeneration proxy interpretation.

The results also show an important nuance: clinical variables are highly predictive. The clinical-only comparator outperforms the atlas-guided multimodal model on AIBL heldout. This means the revised paper should not claim that atlas MRI features alone dominate clinical information. Instead, the value of the atlas-guided model is that it retains anatomically grounded MRI information while achieving strong external heldout performance and enabling a regional biological consistency analysis.

The IXI negative-control result is important because a model that performs well on disease cohorts but misclassifies healthy subjects as impaired would have limited clinical utility. The revised model retains 0.998 of IXI scans as CN, directly addressing the false impairment problem observed in v3.

### Limitations

Several limitations remain. The strongest external result is obtained under an AIBL-adapted heldout protocol, not pure ADNI-to-AIBL zero-shot transfer. OASIS transfer remains weak. The biological validation is an MRI neurodegeneration proxy rather than direct Braak staging. The 21-region atlas is coarse and may miss finer cortical patterns. Finally, clinical variables carry substantial diagnostic information, and the clinical-only model should be interpreted as a strong comparator rather than ignored.

## 5. Conclusion

The revised work provides a substantially rebuilt experimental framework for atlas-guided multimodal AD staging. It demonstrates domain-adapted AIBL heldout performance, IXI healthy negative-control specificity, and AD-consistent atlas-region neurodegeneration patterns. It does not claim pure zero-shot generalization or direct Braak staging. The resulting work is substantially different from the original v3 manuscript and provides a stronger, more honest basis for resubmission.

## Required Tables And Figures

### Table 1. Cohort and split summary

Include the subject-level split table from Section 2.2.

### Table 2. Main classification results

Rows: old v3, atlas-only, cascade, ADNI-only hybrid, recommended atlas+clinical HGB, clinical-only RF, biomarker-enhanced HGB. Columns: evaluation cohort, Acc, BAcc, macro AUC, AD-vs-CN AUC/CN retention, CN/MCI/AD recall.

### Table 3. Candidate model ranking

Use top candidates from `hybrid_candidate_ranking.md`.

### Table 4. Neurodegeneration consistency score

Rows: AIBL heldout, AIBL adaptation+heldout, all labeled AD, ADNI validation+internal test. Columns: AD-key score, uniform null, delta, CI, permutation p.

### Figure 1. Revised framework

Show cohort construction, atlas MRI feature extraction, clinical variable integration, model training, heldout evaluation, IXI negative control, and neurodegeneration consistency analysis.

### Figure 2. External classification comparison

Bar plot comparing AIBL BAcc/AUC and IXI CN retention across old v3, atlas-only, cascade, atlas+clinical, and clinical-only models.

### Figure 3. AIBL heldout confusion matrices

Show recommended atlas+clinical model and clinical-only comparator.

### Figure 4. Neurodegeneration consistency

Show AD-key score versus uniform null and regional volume gradients.

### Figure 5. Limitations/stress-test panel

Show OASIS poor transfer explicitly, with wording that it is an unresolved stress-test result.

## Manuscript Tables

### Table 1. Cohort and split summary

| split | scans | subjects | CN | MCI | AD | role |
|---|---|---|---|---|---|---|
| train | 1686 | 450 | 514 | 819 | 353 | ADNI training |
| val | 355 | 97 | 125 | 166 | 64 | ADNI validation |
| internal_test | 360 | 96 | 111 | 171 | 78 | ADNI internal test |
| aibl_adapt_train | 719 | 385 | 536 | 105 | 78 | AIBL adaptation training |
| aibl_adapt_val | 191 | 105 | 147 | 25 | 19 | AIBL adaptation validation |
| aibl_heldout | 397 | 210 | 303 | 53 | 41 | Locked AIBL heldout external test |
| oasis_external | 99 | 99 | 59 | 29 | 11 | OASIS external stress test |
| ixi_external | 581 | 581 | 581 | 0 | 0 | IXI healthy negative-control cohort |

### Table 2. Main classification results

| model | evaluation | acc | balanced_acc | macro_auc | ad_vs_cn_auc | cn_retention | recall_cn | recall_mci | recall_ad | note |
|---|---|---|---|---|---|---|---|---|---|---|
| Old v3 ensemble | AIBL external | 0.606 | 0.399 | 0.597 |  |  |  |  |  | Failed external baseline |
| Old v3 ensemble | IXI healthy | 0.439 | 0.439 | NA |  | 0.439 |  |  |  | High false impairment rate |
| Atlas-only HGB | AIBL heldout | 0.776 | 0.479 | 0.732 | 0.884 |  | 0.944 | 0.151 | 0.341 | MRI-only baseline |
| Cascade RF-logreg | AIBL heldout | 0.751 | 0.391 | 0.756 | 0.886 |  | 0.954 | 0.000 | 0.220 | High specificity but weak MCI |
| ADNI-only hybrid RF | AIBL heldout | 0.549 | 0.406 | 0.753 | 0.870 |  | 0.614 | 0.604 | 0.000 | Zero-shot remains insufficient |
| Recommended atlas+clinical HGB | AIBL heldout | 0.882 | 0.741 | 0.942 | 0.990 |  | 0.964 | 0.528 | 0.732 | Main atlas-guided multimodal model |
| Recommended atlas+clinical HGB | IXI healthy | 0.998 | 0.998 | NA |  | 0.998 | 0.998 | 0.000 | 0.000 | Healthy negative control |
| Clinical-only RF | AIBL heldout | 0.922 | 0.835 | 0.957 | 0.997 |  | 0.970 | 0.755 | 0.780 | Comparator / upper bound |
| Biomarker-enhanced HGB | AIBL heldout | 0.861 | 0.703 | 0.942 | 0.990 |  | 0.957 | 0.396 | 0.756 | Sensitivity analysis |

### Table 3. Replicate stability for key hybrid candidates

| run | aibl_heldout_bacc_mean | aibl_heldout_bacc_std | aibl_heldout_auc_mean | ixi_cn_retention_mean | ixi_cn_retention_std | n |
|---|---|---|---|---|---|---|
| atlas_biomarker_enhanced__hgb | 0.703 | 0.000 | 0.942 | 0.997 | 0.000 | 4 |
| atlas_core_clinical__hgb | 0.741 | 0.000 | 0.942 | 0.998 | 0.000 | 4 |
| clinical_core_only__rf_balanced | 0.830 | 0.003 | 0.957 | 1.000 | 0.000 | 4 |

### Table 4. Atlas neurodegeneration consistency validation

| group | ad_key_score | uniform_null | delta | ci_low | ci_high | permutation_p |
|---|---|---|---|---|---|---|
| aibl_heldout | 0.510 | 0.286 | 0.225 | 0.479 | 0.526 | 0.026 |
| aibl_adapt_heldout | 0.512 | 0.286 | 0.226 | 0.493 | 0.525 | 0.035 |
| all_labeled_ad | 0.426 | 0.286 | 0.141 | 0.348 | 0.478 | 0.021 |
| adni_val_internal_test | 0.342 | 0.286 | 0.056 | 0.258 | 0.503 | 0.184 |

## Manuscript Figures

### Figure 1. Revised study design

![Figure 1. Revised study design](/Users/mac/Documents/章节1/reports/v4/figures/figure1_revised_study_design.png)

**Caption:** Revised v4 workflow with subject-level splitting, AIBL adaptation and locked heldout testing, IXI healthy negative-control evaluation, OASIS stress testing, and atlas-region biological validation.

PDF version: `/Users/mac/Documents/章节1/reports/v4/figures/figure1_revised_study_design.pdf`

### Figure 2. External classification improvement

![Figure 2. External classification improvement](/Users/mac/Documents/章节1/reports/v4/figures/figure2_external_classification_improvement.png)

**Caption:** External performance improvement across the failed v3 baseline, atlas-only and cascade baselines, the main atlas+clinical model, and a clinical-only comparator.

PDF version: `/Users/mac/Documents/章节1/reports/v4/figures/figure2_external_classification_improvement.pdf`

### Figure 3. AIBL heldout confusion matrices

![Figure 3. AIBL heldout confusion matrices](/Users/mac/Documents/章节1/reports/v4/figures/figure3_aibl_confusion_matrices.png)

**Caption:** AIBL locked heldout confusion matrices for the main atlas+clinical HGB model and the clinical-only RF comparator.

PDF version: `/Users/mac/Documents/章节1/reports/v4/figures/figure3_aibl_confusion_matrices.pdf`

### Figure 4. Neurodegeneration consistency score

![Figure 4. Neurodegeneration consistency score](/Users/mac/Documents/章节1/reports/v4/figures/figure4_neurodegeneration_consistency.png)

**Caption:** MRI neurodegeneration consistency analysis showing AD-key volume signal concentration and AIBL heldout regional gradients in ventricles, hippocampus, and amygdala.

PDF version: `/Users/mac/Documents/章节1/reports/v4/figures/figure4_neurodegeneration_consistency.pdf`

### Figure 5. Honest OASIS stress-test limitation

![Figure 5. Honest OASIS stress-test limitation](/Users/mac/Documents/章节1/reports/v4/figures/figure5_oasis_stress_test.png)

**Caption:** OASIS stress-test results, reported explicitly as an unresolved transfer limitation rather than hidden.

PDF version: `/Users/mac/Documents/章节1/reports/v4/figures/figure5_oasis_stress_test.pdf`

## Figure Caption Text For Manuscript

**Figure 1. Revised study design.** Revised v4 workflow with subject-level splitting, AIBL adaptation and locked heldout testing, IXI healthy negative-control evaluation, OASIS stress testing, and atlas-region biological validation.

**Figure 2. External classification improvement.** External performance improvement across the failed v3 baseline, atlas-only and cascade baselines, the main atlas+clinical model, and a clinical-only comparator.

**Figure 3. AIBL heldout confusion matrices.** AIBL locked heldout confusion matrices for the main atlas+clinical HGB model and the clinical-only RF comparator.

**Figure 4. Neurodegeneration consistency score.** MRI neurodegeneration consistency analysis showing AD-key volume signal concentration and AIBL heldout regional gradients in ventricles, hippocampus, and amygdala.

**Figure 5. OASIS stress-test limitation.** OASIS stress-test results, reported explicitly as an unresolved transfer limitation rather than hidden.

## Submission Integration Checklist

1. Replace the old ARA-Net title/abstract with the v4 atlas-guided multimodal framing.
2. Insert Table 1 in Methods after cohort/split description.
3. Insert Table 2 in Results after the external classification paragraph.
4. Insert Table 3 near the stability/sensitivity analysis.
5. Insert Table 4 near the neurodegeneration consistency section.
6. Insert Figures 1-4 in the main manuscript if space allows.
7. Keep Figure 5 in the main manuscript or supplement, but do not omit the OASIS limitation from the text.
8. Remove direct Braak-staging wording and all claims that CAS validates attention as a biomarker.
9. Fix all old equation boxes, broken figure references, and `Error! Reference source not found` artifacts before submission.
10. Add reviewer-accessible code/data manifest instructions before resubmission.
