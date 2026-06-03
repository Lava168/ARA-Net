# Review Response / Rewrite Matrix

This document is a planning draft for a substantial resubmission. It should not be framed as a minor revision response, because the original decision was reject. Use it to guide the rewritten manuscript, cover letter, and response-to-reviewers language if the journal allows resubmission.

## Editor / AE Fatal Concerns

| Concern | Problem in old manuscript | New evidence | How to write the response |
|---|---|---|---|
| Cross-dataset generalization unsupported | IXI and OASIS were used only for attention similarity; no external classification metrics supported the generalization claim. | Added locked AIBL heldout classification and IXI healthy negative-control test. Main atlas+clinical model: AIBL heldout Acc 0.882, BAcc 0.741, AUC 0.942, AD-vs-CN AUC 0.990; IXI CN retention 0.998. | State that the revised manuscript no longer relies on attention-profile similarity as evidence of generalization. It now reports external classification metrics, confusion patterns, per-class recall, and healthy-control specificity. |
| CAS below chance | Original attention CAS was 0.204, below uniform 6/21 = 0.286, so it could not support clinical alignment. | Replaced with atlas neurodegeneration consistency score. AIBL heldout AD-key volume score 0.510 vs 0.286 null, CI [0.479, 0.526], p=0.026. | State that the old CAS was removed/reframed. The new validity target is disease-consistent structural MRI change, not attention mass. |
| Braak validation non-significant | Original Braak correlation was rho=0.174, p=0.451. | Removed direct Braak claim. Added MRI neurodegeneration/Braak-alternative validation using hippocampus/amygdala/ventricle volume trends. | Say clearly that the revised manuscript does not claim direct Braak staging. It offers a weaker but empirically supported MRI proxy validation. |

## Reviewer 2

| Reviewer 2 point | New action | Evidence / manuscript insertion |
|---|---|---|
| Accuracy too low for clinical deployment; discuss use case. | Reframe as decision-support/staging research, not deployment-ready clinical software. Add limitations paragraph. | Main AIBL heldout BAcc 0.741; clinical-only comparator BAcc 0.830; OASIS limitation remains. |
| MCI recall concerning; provide error analysis. | Add per-class recall to every result table and explicitly discuss MCI. | Main AIBL heldout MCI recall 0.528; clinical-only comparator MCI recall 0.741 +/- 0.008. |
| Compare with simple volumetric + clinical baseline. | Added atlas-only, atlas+clinical, clinical-only, and biomarker-enhanced models. | `hybrid_candidate_ranking.md`, `hybrid_replicate_summary.md`. |
| Explain why explainable 67% model is useful vs black-box 71%. | Avoid defending old 67% result. Replace with new framework and stronger external metrics. | Old v3 is now treated as a failure baseline. |
| Anatomical regularization design unclear. | If keeping deep ARA-Net material, move it to secondary/ablation; do not make it the central evidence. | Deep v4 did not outperform hybrid and should not anchor the revised paper. |
| Equation/figure/reference errors. | Full manuscript cleanup required. | Fix malformed equations, reference placeholders, figure labels. |

## Reviewer 4

| Reviewer 4 point | New action | Evidence / manuscript insertion |
|---|---|---|
| Clarify which subjects/scans were included and how selected. | Add manifest-based cohort table and subject-level split protocol. | Dataset table: ADNI train/val/internal, AIBL adapt train/val/heldout, OASIS, IXI. |
| ADNI selection criteria unclear. | Describe source table/label mapping and subject-level split. | Use `manifest_v4_summary.json` and manifest CSV as reproducibility evidence. |
| OASIS selection criteria unclear. | Add OASIS as external stress test; state limitations. | OASIS remains weak, do not overclaim. |
| Figure parcellation misleading. | Replace figure with actual 21-region atlas visualization or remove misleading parcellation. | Required figure rewrite. |
| Why 21 regions, not full DK? | Add limitation and future/finer parcellation sensitivity plan. | 21-region coarseness is explicitly listed as limitation. |
| Code unavailable. | Prepare scripts and manifest for release/reviewer access. | Local `scripts/`, server `outputs/v4`. |
| AD-key regions not matching top attention. | Remove attention-CAS centrality; use AD-key neurodegeneration volume score instead. | AIBL AD-key score 0.510 vs null. |
| Broken reference placeholders. | Full manuscript cleanup. | Required before submission. |
| Computational workstation limitations. | Add a short deployment/runtime limitation paragraph. | Discuss FastSurfer/atlas preprocessing burden and workstation constraints. |

## Reviewer 5

| Reviewer 5 point | New action | Evidence / manuscript insertion |
|---|---|---|
| CAS is unvalidated and below chance. | Replace CAS with atlas neurodegeneration consistency score. | AIBL heldout score 0.510 vs 0.286 null, p=0.026. |
| No external classification on IXI/OASIS. | Add AIBL heldout external classification and IXI healthy specificity; keep OASIS as stress test. | AIBL BAcc 0.741/AUC 0.942; IXI CN retention 0.998; OASIS weak. |
| Braak correlation non-significant. | Remove direct Braak claim; use proxy validation. | State no direct Braak staging. |
| Baselines undertuned. | Do not rely on ViT comparison. Add strong classical baselines and clinical-only comparator. | Clinical-only RF BAcc 0.830 +/- 0.003 on AIBL heldout. |
| Presentation issues: equations, fonts, placeholders. | Full formatting pass required. | Fix before resubmission. |

## Recommended Revised Manuscript Structure

1. Introduction: state new problem as cross-cohort, atlas-guided, clinically adapted AD staging with verifiable biological consistency.
2. Cohorts and preprocessing: include explicit subject/scans table and label definitions.
3. Feature/model framework: describe atlas MRI features, core clinical variables, and model protocols.
4. Evaluation protocol: distinguish ADNI-only zero-shot, AIBL-adapted heldout, IXI healthy negative control, and OASIS stress test.
5. Results 1: failure analysis of old v3 and pure zero-shot baselines.
6. Results 2: main atlas+clinical model on AIBL heldout and IXI.
7. Results 3: clinical-only and biomarker-enhanced sensitivity analyses.
8. Results 4: atlas neurodegeneration consistency score.
9. Discussion: what is solved, what is not solved, and why OASIS/Braak/direct clinical deployment remain limitations.

## Exact Claim Boundaries

Allowed:

- "The revised atlas-guided multimodal model achieved strong performance on a locked AIBL heldout split."
- "IXI was used as a healthy external negative-control cohort to estimate false impairment."
- "The original attention-only CAS was replaced by a structural neurodegeneration consistency score."
- "The biological validation is consistent with MRI neurodegeneration patterns in AD-relevant regions."

Not allowed:

- "The model solved pure zero-shot cross-dataset AD staging."
- "The model is directly Braak-stage validated."
- "Attention alone is a validated biomarker."
- "OASIS external transfer is solved."
- "The model is ready for clinical deployment."
