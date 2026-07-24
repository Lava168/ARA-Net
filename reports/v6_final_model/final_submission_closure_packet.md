# Final Submission Closure Packet

## Purpose

This packet closes the remaining manuscript-integration issues that were still open after the V6 model lock: Figure 1 wording, OASIS handling, reviewer-requested citations, terminology cleanup, and the exact wording boundaries for CAS/Braak/clinical-use claims.

## Final Figure Blueprint Decision

Do not generate new figures until the figure blueprint is approved. Use `final_figure_blueprint.md` as the authoritative plan for what each main and supplementary figure should contain.

| figure | planned content | manuscript role |
|---|---|---|
| Figure 1 | Revised framework and claim-boundary overview. | Shows that the work is a substantive V6 rebuild. |
| Figure 2 | External rescue result: v3 failure, v4 rebuild, final scan/subject result, clinical comparator. | Addresses external classification. |
| Figure 3 | Subject-level confusion matrices and error transitions. | Addresses AD/MCI error risk. |
| Figure 4 | Bootstrap stability and confidence intervals. | Addresses robustness. |
| Figure 5 | CAS/Braak replacement with atlas structural neurodegeneration consistency. | Addresses biological validation without overclaiming. |

### Figure 1 Caption

**Figure 1. Revised ARA-Net V6 workflow.** The revised workflow uses leakage-free subject-level cohort construction, atlas-guided multimodal feature extraction, a locked rescue probability ensemble, and subject-level probability averaging. The primary endpoint is locked AIBL heldout CN/MCI/AD staging, with IXI healthy-control CN retention as a specificity endpoint. OASIS is retained as an external stress test rather than a successful validation cohort. The original attention-only CAS and direct Braak-stage claims are replaced by atlas structural neurodegeneration consistency analysis and explicit claim-boundary language.

## OASIS Handling Decision

Do not use the old OASIS figure as a main success figure. Replace it with a brief limitation table or a short stress-test paragraph. The OASIS result is important because it shows the remaining domain-transfer boundary, but it should not visually compete with the locked AIBL primary endpoint.

Recommended manuscript placement:

- Results: one short "OASIS stress test" subsection after the main AIBL/IXI results.
- Discussion: one limitation sentence explaining that OASIS remains unresolved and needs larger/cleaner external cohorts.
- Supplement: optional OASIS figure only if the journal requests all external cohorts in figure form.

Recommended table text:

| cohort | n | Acc | BAcc | macro AUC | AD-vs-CN AUC | recall CN/MCI/AD | interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| OASIS stress test | 99 | 0.586 | 0.334 | 0.554 | 0.371 | 0.966/0.034/0.000 | Unresolved external transfer limitation; not a validation success. |

## Reviewer-Requested Citations

Add both references to the final manuscript reference list and use them in the Introduction/Discussion interpretability paragraph.

1. Pfeifer, B., Gevaert, A., Loecher, M., & Holzinger, A. (2025). Tree smoothing: Post-hoc regularization of tree ensembles for interpretable machine learning. *Information Sciences, 690*, 121564. https://doi.org/10.1016/j.ins.2024.121564

2. Retzlaff, C. O., Angerschmid, A., Saranti, A., Schneeberger, D., Roettger, R., Mueller, H., & Holzinger, A. (2024). Post-hoc vs ante-hoc explanations: xAI design guidelines for data scientists. *Cognitive Systems Research, 86*, 101243. https://doi.org/10.1016/j.cogsys.2024.101243

Suggested insertion:

> Recent xAI work distinguishes post-hoc explanation from ante-hoc interpretability and emphasizes that explanation tools require task-specific validation. Accordingly, the revised manuscript no longer treats attention concentration as a validated biomarker. Instead, the biological analysis is limited to atlas-region structural neurodegeneration consistency.

## Terminology Lock

Use these terms consistently:

| use | avoid |
|---|---|
| atlas-guided multimodal AD staging framework | attention model as the main claim |
| locked AIBL heldout subject-level endpoint | pure zero-shot success |
| domain-adapted external heldout validation | universal cross-cohort generalization |
| IXI healthy negative-control specificity | IXI AD/MCI validation |
| OASIS stress-test limitation | OASIS external validation success |
| atlas structural neurodegeneration consistency | CAS validates attention as a biomarker |
| MRI neurodegeneration proxy | direct Braak-stage validation |
| open-source deployable research prototype | deployment-ready clinical diagnostic device |

## Word Manuscript Replacement Rules

Apply these edits to the submitted Word manuscript before resubmission.

| old wording to remove | replacement |
|---|---|
| CAS validates clinical alignment of attention. | The original attention-only CAS was removed because it was below the uniform AD-key regional null. |
| Attention maps are biologically validated biomarkers. | The revised analysis evaluates atlas structural neurodegeneration consistency, not attention as a biomarker. |
| Direct Braak-stage validation / Braak-stage correlation supports the model. | No neuropathological Braak-stage labels are available; the biological claim is limited to an MRI neurodegeneration proxy. |
| OASIS external validation succeeded. | OASIS remains an unresolved stress-test limitation. |
| The model is ready for clinical deployment. | The model is an open-source research prototype requiring prospective validation and regulatory assessment before clinical use. |

## Formula And Reference Closure

The public V6 rewrite package uses three final formulas:

1. Log-probability ensemble pooling.
2. Temperature and class-offset calibration.
3. Subject-level probability averaging.

The public Markdown package has no `Error! Reference source not found` placeholders. The remaining required action is a final Word-document pass to remove stale equation boxes, old CAS equations, and old direct Braak wording from the submitted `.docx`.

## Reviewer-Safe Summary

The revised work is no longer a small v3 patch. It now presents a locked subject-level external AIBL endpoint, IXI healthy specificity, bootstrap stability, MCI/AD error analysis, a public deployment wrapper, a claim-boundary audit, and a replacement of invalid attention-only CAS/Braak claims with a structural MRI neurodegeneration proxy. The strongest claim is domain-adapted external AD staging, not pure zero-shot generalization, direct Braak validation, OASIS success, or clinical deployment readiness.
