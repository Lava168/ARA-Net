# Cover Letter Draft For Substantial Resubmission

Dear Editor,

We are submitting a substantially rebuilt version of our manuscript, formerly titled "ARA-Net: Atlas-Guided Region Attention for Interpretable Alzheimer's Disease Diagnosis from Structural MRI." In response to the previous decision, we did not attempt a narrow revision of the original work. Instead, we rebuilt the experimental framework, revised the central claims, and added new external validation, comparator, and biological-consistency analyses.

The revised manuscript is now positioned as a cross-cohort atlas-guided and clinically adapted AD staging framework rather than a purely attention-based MRI classifier. The new framework combines anatomically grounded atlas-derived MRI features with core clinical variables, evaluates classification on a locked AIBL heldout split, tests healthy-control specificity on IXI, and replaces the original attention-only CAS/Braak claims with an empirically tested atlas-region neurodegeneration consistency analysis.

The main new evidence is as follows. First, we added explicit external classification results. The previous v3 model failed on external validation, with AIBL balanced accuracy of 0.399 and IXI healthy CN retention of 0.439. The revised atlas-guided multimodal model achieved AIBL heldout accuracy of 0.882, balanced accuracy of 0.741, macro AUC of 0.942, and AD-vs-CN AUC of 0.990, with CN/MCI/AD recall of 0.964/0.528/0.732. On the IXI healthy negative-control cohort, it retained CN specificity at 0.998. These results were reproduced across four seeds for the key hybrid models.

Second, we added comparator models requested by the reviewers. The revised experiments include MRI/atlas-only, cascade, atlas+clinical, clinical-only, and biomarker-enhanced sensitivity analyses. The clinical-only model achieved the strongest AIBL heldout classification performance, but we present it as a comparator/upper-bound rather than the main atlas-guided model. The main model is selected because it retains the atlas-derived MRI component and supports the mechanistic aims of the study.

Third, we replaced the previous CAS interpretation. The original attention-based CAS was below the uniform 6/21 null and therefore could not support the claimed clinical alignment. The revised manuscript instead evaluates an atlas-region neurodegeneration consistency score based on disease-consistent volume changes in a priori AD-relevant regions. In the locked AIBL heldout split, the AD-key volume score was 0.510 versus a uniform null of 0.286, with bootstrap CI [0.479, 0.526] and permutation p=0.026.

Fourth, we removed the unsupported direct Braak claim. Because the original Braak correlation was non-significant, the revised manuscript now frames the analysis as a Braak-alternative MRI neurodegeneration proxy rather than direct neuropathological staging. We explicitly state this limitation and avoid claiming direct Braak validation.

We believe the revised manuscript is substantively different from the previous submission. It directly addresses the previously identified concerns regarding unsupported cross-dataset generalization, invalid CAS interpretation, and non-significant Braak validation, while adding leakage-free subject-level cohort manifests, external heldout classification, healthy negative-control testing, stronger comparator baselines, and reproducible analysis scripts.

We appreciate your consideration of this substantially revised work.

Sincerely,

[Author names]
