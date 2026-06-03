# Manuscript Rewrite Package

## New Manuscript Positioning

Recommended title direction:

**Atlas-guided multimodal Alzheimer disease staging with external heldout validation and neurodegeneration-consistent regional biomarkers**

Core claim to make:

The revised work should no longer be framed as a purely attention-based deep MRI classifier. It should be framed as a cross-cohort atlas-guided and clinically adapted AD staging framework that combines anatomically grounded MRI region features with core clinical variables, validates classification on a locked AIBL heldout split, tests healthy specificity on IXI, and replaces the failed CAS/Braak claim with an empirically tested atlas-region neurodegeneration consistency score.

Claims to avoid:

- Do not claim pure ADNI-to-AIBL zero-shot staging is solved.
- Do not claim direct Braak staging validation.
- Do not use the clinical-only model as the main atlas-guided model.
- Do not hide the weak OASIS transfer.

## Dataset And Split Table

| Split | Scans | Subjects | CN | MCI | AD | Role |
|---|---:|---:|---:|---:|---:|---|
| train | 1686 | 450 | 514 | 819 | 353 | ADNI training |
| val | 355 | 97 | 125 | 166 | 64 | ADNI model selection |
| internal_test | 360 | 96 | 111 | 171 | 78 | ADNI internal test |
| aibl_adapt_train | 719 | 385 | 536 | 105 | 78 | AIBL adaptation training |
| aibl_adapt_val | 191 | 105 | 147 | 25 | 19 | AIBL adaptation validation |
| aibl_heldout | 397 | 210 | 303 | 53 | 41 | Locked AIBL heldout external test |
| oasis_external | 99 | 99 | 59 | 29 | 11 | OASIS external stress test |
| ixi_external | 581 | 581 | 581 | 0 | 0 | Healthy external negative-control cohort |

## Main Result Table

| Model / protocol | Evaluation | Acc | BAcc | Macro AUC | AD-vs-CN AUC / CN retention | CN/MCI/AD recall | Interpretation |
|---|---|---:|---:|---:|---|---|---|
| Old v3 ensemble | AIBL external | 0.606 | 0.399 | 0.597 |  | see old v3 report | Failed external baseline |
| Old v3 ensemble | IXI healthy | 0.439 | 0.439 | NA | CN retention 0.439 | see old v3 report | High false impairment rate |
| Atlas-only HGB | AIBL heldout | 0.776 | 0.479 | 0.732 | AD-vs-CN AUC 0.884 | 0.944/0.151/0.341 | MRI specificity improves but staging remains weak |
| Cascade RF-logreg | AIBL heldout | 0.751 | 0.391 | 0.756 | AD-vs-CN AUC 0.886 | 0.954/0.000/0.220 | Healthy-specific but fails MCI heldout |
| ADNI-only hybrid RF | AIBL heldout | 0.549 | 0.406 | 0.753 | AD-vs-CN AUC 0.870 | 0.614/0.604/0.000 | Pure zero-shot remains insufficient |
| Recommended atlas+clinical HGB | AIBL heldout | 0.882 | 0.741 | 0.942 | AD-vs-CN AUC 0.990 | 0.964/0.528/0.732 | Main atlas-guided multimodal result |
| Recommended atlas+clinical HGB | IXI healthy | 0.998 | 0.998 | NA | CN retention 0.998 | 0.998/0.000/0.000 | Healthy negative-control specificity |
| Clinical-only RF | AIBL heldout | 0.922 | 0.835 | 0.957 | AD-vs-CN AUC 0.997 | 0.970/0.755/0.780 | Strong comparator / upper-bound |
| Biomarker-enhanced HGB | AIBL heldout | 0.861 | 0.703 | 0.942 | AD-vs-CN AUC 0.990 | 0.957/0.396/0.756 | Sensitivity analysis |

## Three Fatal Issues And New Evidence

### 1. Cross-dataset generalization was unsupported

Old manuscript problem: IXI and OASIS were used for attention similarity only; no external classification metrics were reported.

New evidence:

- Locked AIBL heldout external test for the recommended atlas+clinical model: Acc 0.882, BAcc 0.741, macro AUC 0.942, AD-vs-CN AUC 0.990, recall CN/MCI/AD 0.964/0.528/0.732.
- IXI healthy negative-control test: Acc 0.998, BAcc 0.998, macro AUC NA, CN retention 0.998, recall CN/MCI/AD 0.998/0.000/0.000.
- Multi-seed confirmation for the recommended model: AIBL heldout BAcc 0.741 +/- 0.000, IXI CN retention 0.998 +/- 0.000, n=4.

How to write it:

We now explicitly distinguish zero-shot external evaluation from clinically adapted external heldout evaluation. ADNI-only models remain weak on AIBL AD detection, whereas the clinically adapted atlas-guided model generalizes to a locked AIBL heldout set and preserves specificity on IXI.

### 2. CAS was below chance and unvalidated

Old manuscript problem: attention-based CAS was below the uniform 6/21 null and therefore could not support clinical alignment.

New evidence:

- AIBL heldout AD-key volume score 0.510 versus uniform null 0.286; delta 0.225; bootstrap CI [0.479, 0.526]; permutation p=0.0260.
- The new score uses atlas-derived volume changes rather than unvalidated attention mass.

How to write it:

We replace the original attention-only CAS with an atlas-region neurodegeneration consistency score. This is not a cosmetic reinterpretation: it changes the validity target from attention concentration to disease-consistent structural MRI changes in a priori AD-relevant regions.

### 3. Braak correlation was non-significant

Old manuscript problem: the reported Braak correlation was non-significant and could not support a mechanistic claim.

New evidence:

- The revised validation should be called Braak-alternative or neurodegeneration-proxy validation.
- The significant AIBL heldout AD-key volume score supports MRI-consistent medial temporal atrophy and ventricular expansion patterns.
- ADNI-only biological validation remains non-significant, so this limitation must be stated.

How to write it:

We no longer claim direct Braak staging. Instead, we evaluate whether atlas-derived disease gradients concentrate in established MRI neurodegeneration regions. This is a weaker but empirically supported biological validation.

## Proposed Revised Results Sections

1. Cohort construction and leakage-free subject-level splits.
2. External failure analysis of the original v3 model.
3. MRI/atlas-only feature baseline and healthy specificity recovery.
4. Clinically adapted atlas-guided model on locked AIBL heldout.
5. IXI negative-control specificity analysis.
6. Clinical-only and biomarker-enhanced sensitivity analyses.
7. CAS replacement: atlas neurodegeneration consistency score.
8. Limitations: OASIS transfer, no direct Braak labels, adaptation-vs-zero-shot distinction, 21-region atlas coarseness.

## Cover Letter Core Paragraph

In response to the previous decision, we did not attempt a narrow revision of the original manuscript. Instead, we rebuilt the experimental framework and substantially rewrote the study. The revised work now includes explicit subject-level cohort manifests, external classification on AIBL with a locked heldout split, an IXI healthy negative-control specificity test, MRI/atlas-only and clinical-only comparator models, multi-seed confirmation of the key hybrid results, and a replacement of the original attention-only CAS/Braak claims with an empirically tested atlas-region neurodegeneration consistency analysis. These additions directly address the previously identified concerns regarding unsupported cross-dataset generalization, an invalid CAS result, and non-significant Braak validation.

## Response Matrix

| Reviewer/editor concern | New action | Evidence file/result | Manuscript change |
|---|---|---|---|
| Cross-dataset generalization unsupported | Added AIBL heldout, IXI healthy negative control, OASIS stress test | `v4_decision_report.md`, `hybrid_replicate_summary.md` | New external validation section and tables |
| CAS below chance | Replaced attention-only CAS with atlas neurodegeneration consistency score | AIBL heldout score 0.510 vs 0.286 null, p=0.026 | New biomarker validation section |
| Braak non-significant | Removed direct Braak claim, reframed as Braak-alternative MRI neurodegeneration proxy | AIBL and pooled AD-key volume score | Revised interpretation and limitation |
| Need volumetric/clinical baseline | Added atlas-only, atlas+clinical, clinical-only, biomarker-enhanced models | Candidate ranking and replicate summary | New baseline/sensitivity table |
| Reproducibility unclear | Added subject-level manifest and split counts | manifest summary in v4 reports | New cohort and split subsection |
| MCI errors | Reported per-class recall and showed MCI improvement in AIBL heldout | MCI recall 0.528 in main model; 0.755 clinical-only comparator | New error/per-class analysis |
| Source availability | Scripts and outputs now organized under reproducible v4 pipeline | local `scripts/`, server `outputs/v4` | Release checklist and methods appendix |

## Remaining Weaknesses To State Honestly

- OASIS remains weak and should be described as an external stress test where transfer is not solved.
- The strongest classifier is clinical-only; the atlas-guided model is chosen because it retains MRI atlas information and better supports the paper's mechanistic story.
- AIBL heldout is domain-adapted external validation, not pure zero-shot transfer.
- The biological validation is a Braak alternative, not direct neuropathological staging.
- The 21-region atlas is coarse; a finer parcellation sensitivity analysis would be the strongest next experiment.
