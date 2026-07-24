# Insertable Manuscript Sections Draft

## Revised Contribution Paragraph

This study was substantially redesigned from an attention-only structural MRI classifier into a cross-cohort atlas-guided multimodal AD staging framework. The revised framework combines atlas-derived MRI regional features with core clinical variables, evaluates classification on a locked AIBL heldout cohort, uses IXI as an external healthy negative-control cohort, and replaces the original attention-only clinical alignment score with an atlas-region neurodegeneration consistency analysis. This design directly tests whether the model can generalize beyond ADNI, preserve specificity in healthy subjects, and produce disease-consistent regional MRI patterns.

## External Validation Protocol

All splits were defined at the subject level. ADNI scans were divided into training, validation, and internal test sets. AIBL was divided into adaptation training, adaptation validation, and a locked heldout split. OASIS was retained as an external stress-test cohort, and IXI was used as a healthy external negative-control cohort. The resulting splits contained 1,686 ADNI training scans, 355 ADNI validation scans, 360 ADNI internal test scans, 719 AIBL adaptation-training scans, 191 AIBL adaptation-validation scans, 397 locked AIBL heldout scans, 99 OASIS scans, and 581 IXI healthy scans.

## Main Classification Result

The original v3 model failed to support cross-dataset generalization, achieving AIBL balanced accuracy of 0.399 and macro AUC of 0.597, while retaining only 0.439 of IXI healthy controls as CN. In contrast, the recommended atlas-guided multimodal model achieved AIBL heldout accuracy of 0.882, balanced accuracy of 0.741, macro AUC of 0.942, and AD-vs-CN AUC of 0.990. Per-class recall on AIBL heldout was 0.964 for CN, 0.528 for MCI, and 0.732 for AD. On the IXI healthy negative-control cohort, the same model retained 0.998 of scans as CN. These results were reproduced across four seeds, with AIBL heldout balanced accuracy of 0.741 +/- 0.000 and IXI CN retention of 0.998 +/- 0.000.

## Comparator Result

We additionally evaluated MRI/atlas-only, cascade, clinical-only, and biomarker-enhanced sensitivity models. The atlas-only HGB model improved healthy specificity but remained limited for staging, with AIBL heldout balanced accuracy of 0.479 and MCI recall of 0.151. The cascade model achieved perfect IXI CN retention but failed to detect AIBL heldout MCI. The strongest classifier was the clinical-only random forest, which achieved AIBL heldout balanced accuracy of 0.830 +/- 0.003 and IXI CN retention of 1.000. We report this model as a clinical comparator and upper-bound rather than the main atlas-guided model, because it does not retain the atlas-derived MRI component that is central to the study's mechanistic objective.

## CAS Replacement / Biological Consistency

The original attention-based CAS was removed because it was below the uniform AD-key region null and therefore did not support clinical alignment. We instead evaluated an atlas-region neurodegeneration consistency score based on structural MRI volume changes in a priori AD-relevant regions. In the locked AIBL heldout split, the AD-key volume score was 0.510, compared with a uniform null of 0.286, with score difference 0.225, bootstrap CI [0.479, 0.526], and permutation p=0.026. This analysis supports disease-consistent structural change in AD-relevant regions, while avoiding the unsupported claim that attention mass alone constitutes a validated biomarker.

## Braak-Alternative Framing

The revised manuscript does not claim direct Braak staging validation. The original Braak correlation was non-significant, and the available labels do not support a direct neuropathological staging analysis. We therefore frame the biological validation as a Braak-alternative MRI neurodegeneration proxy: the analysis tests whether disease-related atlas features concentrate in regions known to show AD-related atrophy and ventricular expansion. This is a weaker but empirically supported claim and is stated as such in the revised limitations.

## Limitations Paragraph

Several limitations remain. First, the best-performing external result is obtained under an AIBL-adapted heldout protocol rather than pure ADNI-to-AIBL zero-shot transfer; ADNI-only models remain weak for AIBL AD detection. Second, OASIS transfer remains poor and should be interpreted as an unresolved external stress-test result rather than a successful validation. Third, the biological analysis is an MRI neurodegeneration proxy and not direct Braak staging. Fourth, the 21-region atlas is anatomically interpretable but coarse, and future work should test finer parcellations. Finally, the clinical-only comparator outperforms the atlas-guided model on AIBL heldout, indicating that clinical variables contain substantial diagnostic signal and that the atlas-guided model should be interpreted as a multimodal, mechanistically grounded model rather than as the absolute strongest classifier.
