# V6 Reviewer Response Matrix

## One-Sentence Position

The revised manuscript is no longer a patched attention-classifier paper; it is a rebuilt atlas-guided multimodal AD staging study with locked AIBL subject-level validation, IXI healthy specificity, bootstrap stability, MCI/AD error analysis, and a replacement biological-consistency analysis.

## Editor / Reviewer Critical Issues

| concern | status after v6 | evidence | manuscript action |
|---|---|---|---|
| Cross-dataset generalization unsupported | Largely solved, with honest boundary | Final subject-level AIBL heldout BAcc 0.833, AUC 0.937, AD-vs-CN AUC 1.000; IXI CN retention 1.000 | Replace attention-similarity generalization with explicit external classification and healthy specificity |
| No real external classification on IXI/OASIS | Partly solved | IXI is solved as healthy negative control; OASIS remains weak | Report IXI as specificity validation; report OASIS only as stress-test limitation |
| CAS below chance | Solved by removal/replacement | Original CAS removed; AD-key volume consistency score 0.510 vs 0.286 null, p=0.026 | Do not call old CAS a biomarker; use structural atlas neurodegeneration consistency |
| Braak correlation non-significant | Solved by claim boundary, not by direct Braak proof | Direct Braak claim removed; MRI neurodegeneration proxy retained | Replace "Braak validation" with "Braak-alternative MRI neurodegeneration proxy" |
| MCI recall too low | Materially improved | v4 MCI recall 0.528 -> final subject-level MCI recall 0.686; scan-level 0.642 | Add MCI/AD per-class recall, confusion matrix, and error profiles |
| AD recall weak internally | Improved but still limited | Internal subject-level AD recall 0.550; internal AD-to-CN errors are 0, but many AD become MCI | State internal calibration limitation; emphasize locked AIBL as primary endpoint |
| Clinical deployment claim too strong | Solved by reframing | Results strong externally but OASIS/internal limitations remain | Reframe as decision-support research, not deployment-ready software |
| Need clinical/volumetric comparator | Solved | Clinical-only RF BAcc 0.835; v4 atlas+clinical and final ensemble shown | Present clinical-only as comparator/upper bound, not main model |
| Need error analysis | Solved | AIBL transitions and feature groups in `final_model_error_analysis.md` | Add Results subsection and Figure 5 |
| Formula placeholders / unclear formula | Solved in v6 text | Log-probability pooling, temperature, class offsets, subject averaging formulas written | Replace old formula placeholders with Methods equations |
| Figure 1 / parcellation / workflow unclear | Solved at blueprint level | V6 Figure 1 workflow/caption specified; V6 Figures 2-5 already available | Generate Figure 1 only after blueprint approval |
| Code unavailable | Needs final repository packaging | Scripts are present locally and on server | Add Code Availability statement and prepare GitHub release folder |
| Computational resources unclear | Needs text only | Server training and local post-hoc scripts used | Add Methods appendix paragraph describing CPU/GPU/server environment once exact hardware is known |
| Citation updates requested | Solved for public package | Pfeifer 2025 Tree Smoothing and Retzlaff 2024 post-hoc vs ante-hoc citations verified in closure packet | Add both references to Word manuscript reference list |

## Reviewer-Safe Claim Language

Use:

- "domain-adapted external heldout validation"
- "locked AIBL heldout subject-level evaluation"
- "healthy negative-control specificity"
- "MRI neurodegeneration proxy"
- "atlas-region structural consistency"
- "not deployment-ready"
- "OASIS stress-test failure"

Avoid:

- "pure zero-shot generalization is solved"
- "OASIS external validation succeeded"
- "CAS validates attention as a biomarker"
- "direct Braak-stage validation"
- "clinical deployment"
- "fully interpretable biological mechanism"

## What Is Now Strong

- AIBL subject-level BAcc 0.833 is strong enough to lead the revised paper.
- AIBL AD-vs-CN AUC is effectively perfect at subject level, but write "1.000 in this heldout cohort" rather than implying universal perfection.
- IXI retention 1.000 answers the false-impairment problem.
- AIBL AD-to-CN errors are zero, which is a strong clinical-error-analysis point.
- Bootstrap CIs make the result look less like a single lucky split.

## What Must Stay Honest

- Internal test subject-level BAcc is only 0.448, mainly because CN subjects are often shifted toward MCI.
- OASIS is not solved.
- Clinical-only comparator is very strong, so the paper cannot pretend MRI atlas features alone dominate.
- Biological validation is structural MRI consistency, not neuropathology.
