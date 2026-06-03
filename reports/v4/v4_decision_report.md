# V4 Decision Report

## Recommendation

Use a two-tier story, not a single overclaimed model:

1. MRI/atlas-only evidence shows the original v3 failure mode was fixed for healthy specificity but not fully for MCI staging.
2. The main atlas-guided multimodal model should be `aibl_adapted/atlas_core_clinical__hgb`: it keeps the atlas MRI signal, uses core clinical variables, and performs well on locked AIBL heldout plus IXI.
3. `aibl_adapted/clinical_core_only__rf_balanced` is the strongest classifier but should be presented as a clinical-only comparator or upper-bound, not as the atlas-guided model.
4. `atlas_biomarker_enhanced__hgb` is a biomarker-enhanced sensitivity analysis, not the clean main model.
5. ADNI-only zero-shot models remain weak on AIBL AD detection; do not claim pure zero-shot cross-cohort staging is solved.

## Data Split Evidence

- train: scans=1686, subjects=450, CN/MCI/AD=514/819/353
- val: scans=355, subjects=97, CN/MCI/AD=125/166/64
- internal_test: scans=360, subjects=96, CN/MCI/AD=111/171/78
- aibl_adapt_train: scans=719, subjects=385, CN/MCI/AD=536/105/78
- aibl_adapt_val: scans=191, subjects=105, CN/MCI/AD=147/25/19
- aibl_heldout: scans=397, subjects=210, CN/MCI/AD=303/53/41
- oasis_external: scans=99, subjects=99, CN/MCI/AD=59/29/11
- ixi_external: scans=581, subjects=581, CN/MCI/AD=581/0/0

## Old v3 Failure Baseline

- aibl: Acc=0.606, BAcc=0.399, AUC=0.5974888400448501, pred={'CN': 918, 'MCI': 44, 'AD': 345}
- ixi: Acc=0.439, BAcc=0.439, AUC=None, pred={'CN': 255, 'MCI': 41, 'AD': 285}
- oasis: Acc=0.606, BAcc=0.364, AUC=0.6627818217239608, pred={'CN': 97, 'MCI': 0, 'AD': 2}

## Classification Decision

### Recommended atlas-guided multimodal model: `atlas_core_clinical__hgb`

- internal_test: Acc=0.619, BAcc=0.548, AUC=0.780, ADvCN_AUC=0.923, recall_CN/MCI/AD=0.748/0.754/0.141, pred={'CN': 114, 'MCI': 217, 'AD': 29}
- aibl_adapt_val: Acc=0.901, BAcc=0.824, AUC=0.938, ADvCN_AUC=1.000, recall_CN/MCI/AD=0.952/0.520/1.000, pred={'CN': 150, 'MCI': 20, 'AD': 21}
- aibl_heldout: Acc=0.882, BAcc=0.741, AUC=0.942, ADvCN_AUC=0.990, recall_CN/MCI/AD=0.964/0.528/0.732, pred={'CN': 311, 'MCI': 47, 'AD': 39}
- ixi_external: Acc=0.998, BAcc=0.998, AUC=NA, CN_retention=0.998, recall_CN/MCI/AD=0.998/0.000/0.000, pred={'CN': 580, 'MCI': 1, 'AD': 0}
- oasis_external: Acc=0.576, BAcc=0.322, AUC=0.539, ADvCN_AUC=0.497, recall_CN/MCI/AD=0.966/0.000/0.000, pred={'CN': 94, 'MCI': 3, 'AD': 2}

### Strong clinical-only comparator: `clinical_core_only__rf_balanced`

- internal_test: Acc=0.483, BAcc=0.411, AUC=0.569, ADvCN_AUC=0.583, recall_CN/MCI/AD=0.396/0.696/0.141, pred={'CN': 81, 'MCI': 233, 'AD': 46}
- aibl_adapt_val: Acc=0.906, BAcc=0.882, AUC=0.964, ADvCN_AUC=1.000, recall_CN/MCI/AD=0.925/0.720/1.000, pred={'CN': 141, 'MCI': 29, 'AD': 21}
- aibl_heldout: Acc=0.922, BAcc=0.835, AUC=0.957, ADvCN_AUC=0.997, recall_CN/MCI/AD=0.970/0.755/0.780, pred={'CN': 297, 'MCI': 58, 'AD': 42}
- ixi_external: Acc=1.000, BAcc=1.000, AUC=NA, CN_retention=1.000, recall_CN/MCI/AD=1.000/0.000/0.000, pred={'CN': 581, 'MCI': 0, 'AD': 0}
- oasis_external: Acc=0.596, BAcc=0.333, AUC=0.500, ADvCN_AUC=0.500, recall_CN/MCI/AD=1.000/0.000/0.000, pred={'CN': 99, 'MCI': 0, 'AD': 0}

### Biomarker-enhanced sensitivity model: `atlas_biomarker_enhanced__hgb`

- internal_test: Acc=0.625, BAcc=0.570, AUC=0.791, ADvCN_AUC=0.947, recall_CN/MCI/AD=0.784/0.708/0.218, pred={'CN': 120, 'MCI': 199, 'AD': 41}
- aibl_adapt_val: Acc=0.911, BAcc=0.836, AUC=0.938, ADvCN_AUC=0.999, recall_CN/MCI/AD=0.959/0.600/0.947, pred={'CN': 151, 'MCI': 21, 'AD': 19}
- aibl_heldout: Acc=0.861, BAcc=0.703, AUC=0.942, ADvCN_AUC=0.990, recall_CN/MCI/AD=0.957/0.396/0.756, pred={'CN': 314, 'MCI': 40, 'AD': 43}
- ixi_external: Acc=0.997, BAcc=0.997, AUC=NA, CN_retention=0.997, recall_CN/MCI/AD=0.997/0.000/0.000, pred={'CN': 579, 'MCI': 2, 'AD': 0}
- oasis_external: Acc=0.192, BAcc=0.310, AUC=0.561, ADvCN_AUC=0.443, recall_CN/MCI/AD=0.169/0.034/0.727, pred={'CN': 18, 'MCI': 4, 'AD': 77}

## MRI/Atlas-Only Baseline

- Best atlas-only model `hgb` AIBL heldout: BAcc=0.479, AUC=0.732, recall_CN/MCI/AD=0.944/0.151/0.341.
- IXI healthy specificity: CN retention=0.983.
- Interpretation: useful specificity and AD-vs-CN signal, but MCI/AD staging is insufficient without clinical adaptation.

## Cascade Baseline

- Best cascade `rf__logreg` AIBL heldout: BAcc=0.391, AUC=0.756, recall_CN/MCI/AD=0.954/0.000/0.220.
- IXI CN retention=1.000.
- Interpretation: excellent healthy specificity, but it fails MCI detection on heldout AIBL.

## CAS / Braak-Alternative Biological Validation

- aibl_heldout: AD-key volume score=0.510, uniform=0.286, delta=0.225, CI=[0.479, 0.526], permutation p=0.0260.
- aibl_adapt_heldout: AD-key volume score=0.512, uniform=0.286, delta=0.226, CI=[0.493, 0.525], permutation p=0.0350.
- all_labeled_ad: AD-key volume score=0.426, uniform=0.286, delta=0.141, CI=[0.348, 0.478], permutation p=0.0207.
- adni_val_internal_test: AD-key volume score=0.342, uniform=0.286, delta=0.056, CI=[0.258, 0.503], permutation p=0.1843.
- Interpretation: valid as an MRI neurodegeneration/Braak-proxy check, especially in AIBL and pooled labeled AD; ADNI validation alone is not significant, so do not claim direct Braak staging.

## Stability Status

- Multi-seed replicate check completed.
- atlas_biomarker_enhanced__hgb: AIBL heldout BAcc=0.703+/-0.000, IXI CN retention=0.997+/-0.000 (n=4).
- atlas_core_clinical__hgb: AIBL heldout BAcc=0.741+/-0.000, IXI CN retention=0.998+/-0.000 (n=4).
- clinical_core_only__rf_balanced: AIBL heldout BAcc=0.830+/-0.003, IXI CN retention=1.000+/-0.000 (n=4).

## Manuscript Implication

This is now a substantive new work if framed correctly: a cross-cohort atlas-guided and clinically adapted AD staging framework with locked AIBL heldout evaluation, IXI healthy negative-control specificity, and atlas-region biological validation. Multi-seed confirmation is complete for the key hybrid models; OASIS transfer remains the main unresolved weakness.
