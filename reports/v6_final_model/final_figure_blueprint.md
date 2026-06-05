# Final Figure Blueprint

## Purpose

This blueprint specifies what each manuscript figure should show before any new artwork is generated. The figure set is designed to make the revised work look like a substantive new study: RC-SPE algorithmic evidence, locked external classification, subject-level evaluation, error analysis, and replacement of the invalid attention-only CAS/Braak claim with a structural MRI neurodegeneration proxy.

## Main Figure Set

### Figure 1. Revised Study Framework

**Purpose:** Show that the paper is no longer a small v3 attention-model patch. This figure should establish the new study design, final model, evaluation endpoints, and claim boundaries.

| panel | content | visual form | key message |
|---|---|---|---|
| A | Cohort and split design: ADNI train/validation/internal test; AIBL adaptation train/validation/locked heldout; IXI healthy negative-control; OASIS stress test. | Flow diagram with cohort blocks and subject counts. | The protocol is subject-level and leakage-aware. |
| B | Atlas-guided multimodal features: 21-region MRI atlas summaries plus core clinical variables. | Feature pipeline schematic. | The revised model is atlas-guided multimodal staging, not an attention-only model. |
| C | RC-SPE final algorithm: base-model probabilities, log-probability pooling, class offsets, temperature calibration, risk-constrained selection, and subject-level probability averaging. | Model block diagram. | The locked primary model is RC-SPE, a risk-constrained subject-level probability ensemble. |
| D | Evaluation endpoints: AIBL heldout primary endpoint, IXI specificity endpoint, OASIS stress-test limitation. | Endpoint branch diagram. | The external claim is AIBL/IXI, while OASIS is explicitly bounded. |
| E | Biological interpretation boundary: old CAS removed; atlas structural neurodegeneration consistency; no direct Braak-stage validation. | Before/after claim-boundary schematic. | The biological claim is a structural MRI proxy, not attention as biomarker or direct Braak proof. |

**Suggested caption:**
**Figure 1. Revised ARA-Net V6 study framework.** The revised workflow uses subject-level cohort construction, atlas-guided multimodal feature extraction, RC-SPE risk-constrained probability pooling, and subject-level probability averaging. The primary endpoint is locked AIBL heldout CN/MCI/AD staging, with IXI healthy-control CN retention as a specificity endpoint. OASIS is retained as an external stress test rather than a successful validation cohort. The original attention-only CAS and direct Braak-stage claims are replaced by atlas structural neurodegeneration consistency analysis and explicit claim-boundary language.

### Figure 2. Data Line And Endpoint Design

**Purpose:** Show that the data protocol is a real rebuild and that the final endpoint is leakage-aware.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | ADNI train/validation/internal test inventory. | Cohort block diagram. | `reports/v6_final_model/tables/final_model_classification_table.md` and Methods text. | ADNI is used for development and internal testing. |
| B | AIBL adaptation train/validation and locked heldout inventory. | Split timeline or cohort blocks. | Methods text. | AIBL heldout is the primary external endpoint after adaptation data are separated. |
| C | IXI healthy negative-control cohort. | Single-cohort specificity block. | `reports/v6_final_model/tables/final_model_classification_table.md` | IXI tests false impairment in healthy controls. |
| D | OASIS stress-test cohort. | Small bounded limitation block. | same as A | OASIS is not used for final model selection and is not framed as solved. |
| E | Scan-to-subject endpoint aggregation and leakage control. | Scan-to-subject schematic. | Methods text. | Repeated scans are aggregated at the subject-level endpoint. |

### Figure 3. RC-SPE Algorithm And Risk Constraint

**Purpose:** Make the algorithmic contribution visible and separate RC-SPE from a generic average ensemble.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | Six base-model probability streams: atlas+bio HGB, atlas+clinical HGB, clinical+bio RF, clinical HGB, clinical RF, cascade RF-logreg. | Model stream diagram. | `reports/v6_algorithm_innovation/algorithm_innovation_evidence.md` | RC-SPE begins from heterogeneous probability evidence streams. |
| B | RC-SPE objective: external BAcc, MCI/AD recall, AD-to-CN error avoidance, IXI healthy specificity, and calibration. | Equation/objective schematic. | Methods text and `scripts/generate_algorithm_innovation_evidence.py` | The selection is risk-constrained, not only accuracy-maximizing. |
| C | Log-probability pooling with non-negative weights, class offsets, and temperature scaling. | Equation-to-block schematic. | Methods text. | RC-SPE is calibrated probability pooling, not a binary classifier. |
| D | Scan-level to subject-level probability averaging. | Aggregation schematic. | Methods text. | The primary decision is made at the subject-level endpoint. |
| E | CN/MCI/AD output with confidence and margin. | Three-class probability output. | `deployment/research_inference.py` and frontend. | The deployed research wrapper remains three-class. |

**Suggested caption:**
**Figure 3. RC-SPE risk-constrained subject-level probability ensemble.** RC-SPE combines heterogeneous base-model probability streams through log-probability pooling, class offsets, temperature scaling, and subject-level probability averaging. Its selection objective balances AIBL external staging, MCI/AD rescue, AD-to-CN error avoidance, calibration, and IXI healthy specificity.

### Figure 4. External Classification Performance

**Purpose:** Address the reviewer/editor concern that the original manuscript did not provide real external classification evidence.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | AIBL heldout balanced accuracy across old v3, v4 atlas+clinical, final scan-level, final subject-level, and clinical-only comparator. | Bar plot or point plot. | `reports/v6_final_model/tables/final_model_classification_table.md` | Final subject-level model reaches BAcc 0.833. |
| B | AIBL heldout macro AUC across the same models. | Bar plot or point plot. | same as A | Final subject-level macro AUC is 0.937. |
| C | AIBL heldout CN/MCI/AD recall for the final model, with v4 comparator if space allows. | Grouped bars. | same as A | The main residual weakness is MCI, not total AD collapse. |
| D | IXI healthy CN retention and false-impairment rate. | Two bars or retention/failure gauge. | same as A | Healthy specificity is preserved: CN retention 1.000. |

**Design note:** Clinical-only RF should appear as a comparator or upper bound, not as the central model.

### Figure 5. Confusion Matrix And Error Pattern

**Purpose:** Show what the model gets wrong and why the errors are scientifically defensible.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | AIBL heldout subject-level CN/MCI/AD confusion matrix. | Heatmap with counts and row percentages. | `reports/v6_final_model/tables/aibl_heldout_confusion_transitions.csv` | No AD subject is misclassified as CN. |
| B | Internal test subject-level CN/MCI/AD confusion matrix. | Heatmap with counts and row percentages. | `reports/v6_final_model/tables/internal_test_confusion_transitions.csv` | Internal calibration remains weaker and should be discussed honestly. |
| C | AIBL heldout true-to-predicted transition flow. | Sankey/alluvial or compact flow diagram. | `aibl_heldout_confusion_transitions.csv` | Errors concentrate around MCI/AD boundaries. |
| D | MCI/AD error subtype summary: MCI->CN, MCI->AD, AD->MCI, AD->CN. | Small multiples or stacked bars. | same as A | AD-to-CN is absent; residual AD errors are AD-to-MCI. |

### Figure 6. Algorithmic Evidence, Stability, And Uncertainty

**Purpose:** Show that RC-SPE is not a simple average ensemble, that risk choices were explicit, and that the final result is not dependent on one fragile base stream.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | Algorithmic ablation: best single, arithmetic mean, equal log-pooling, partial RC-SPE, full scan-level RC-SPE, full subject-level RC-SPE. | Grouped bars. | `reports/v6_algorithm_innovation/algorithm_ablation_table.csv` | Full subject-level RC-SPE improves BAcc and removes AD-to-CN errors. |
| B | Calibration comparison for best single, equal log-pooling, and final RC-SPE. | Reliability curve or ECE/NLL panel. | `reports/v6_algorithm_innovation/calibration_table.csv` | Temperature scaling improves calibration while preserving staging performance. |
| C | Risk-profile tradeoff: MCI rescue vs IXI false impairment. | Scatterplot with locked RC-SPE highlighted. | `reports/v6_algorithm_innovation/risk_constraint_candidates.csv` | The highest MCI recall profile is rejected because it increases healthy-control false impairment. |
| D | Leave-one-model-out sensitivity. | Point/range plot. | `reports/v6_algorithm_innovation/leave_one_model_out_table.csv` | Dropping any one stream keeps BAcc 0.823-0.835 and zero AD-to-CN errors. |
| E | Bootstrap BAcc, MCI recall, and AD recall with 95% CIs. | Forest plot or compact violins. | `reports/v6_final_model/final_rescue_model_summary_public.json` | The locked result is stable enough to report, with MCI as the main residual uncertainty. |

### Figure 7. Biological Consistency And CAS/Braak Replacement

**Purpose:** Address CAS validity and Braak/biological validation without overclaiming.

| panel | content | visual form | data source | key message |
|---|---|---|---|---|
| A | Old attention-only CAS failure: old CAS below uniform AD-key regional null. | Bar plot or failure callout. | `reports/v4/review_response_rebuild_matrix.md` | The old CAS should be removed, not defended. |
| B | AD-key region definition: bilateral hippocampus, amygdala, and lateral ventricles. | Atlas-region schematic or labeled region list. | `reports/v4/tables/table4_neurodegeneration.csv` and Methods text. | The replacement target is anatomically grounded. |
| C | AIBL heldout atlas structural neurodegeneration consistency: score 0.510 vs null 0.286. | Bar/point plot with bootstrap CI and p-value. | `reports/v4/tables/table4_neurodegeneration.csv` | AIBL heldout supports structural MRI consistency, p=0.026. |
| D | Cohort-level biological consistency: AIBL heldout, all labeled AD, ADNI internal check. | Multi-row point plot. | same as C | Evidence is strongest externally and pooled; ADNI-only internal check remains non-significant. |
| E | Claim-boundary summary. | Small text panel or badge row. | `reports/v6_final_model/claim_boundary_audit.md` | MRI proxy only; not attention biomarker; not direct Braak validation. |

## Supplementary Figure Set

### Supplementary Figure S1. OASIS Stress Test

Use this only in supplement unless the journal demands all external cohorts in main figures.

| panel | content | visual form | key message |
|---|---|---|---|
| A | OASIS subject-level confusion matrix. | Heatmap. | OASIS is weak and should not be claimed as validation success. |
| B | OASIS recall CN/MCI/AD. | Bar plot. | CN retention is high, but MCI/AD recall collapses. |
| C | AIBL vs OASIS comparison. | Paired metric plot. | Domain transfer remains unresolved. |
| D | Limitation summary. | Text callout. | More external harmonization/cohort cleaning is required. |

### Supplementary Figure S2. Comparator And Ablation Summary

| panel | content | visual form | key message |
|---|---|---|---|
| A | Atlas-only, cascade, atlas+clinical, clinical-only, biomarker-enhanced, final RC-SPE. | Ranking plot. | The final model should be compared transparently. |
| B | Scan-level vs subject-level final RC-SPE. | Paired bars. | Subject-level averaging is the primary endpoint. |
| C | AIBL adaptation validation vs locked heldout. | Metric comparison. | Heldout performance is not from evaluating on the adaptation-validation set. |
| D | Alternative high-MCI and high-AD rescue profiles. | Tradeoff plot. | The locked RC-SPE profile is selected by risk balance rather than single-class maximization. |

### Supplementary Figure S3. Open-Source Research Deployment

| panel | content | visual form | key message |
|---|---|---|---|
| A | CSV probability input -> CLI/API/frontend -> subject-level output. | Deployment flow. | The code is deployable as a research prototype. |
| B | Three-class output example: CN/MCI/AD probability report. | Example table. | The public demo is three-class, not binary. |
| C | Clinical-use boundary. | Warning box. | Not a medical device; not cleared or approved for clinical use. |

## What Not To Draw

- Do not draw OASIS as a successful external validation figure.
- Do not draw attention heatmaps as validated biomarkers.
- Do not draw a direct Braak-stage validation plot unless neuropathological Braak labels become available.
- Do not make clinical deployment readiness a visual claim.
- Do not let clinical-only RF appear to be the central ARA-Net model; it is a comparator/upper bound.

## Priority Order

1. Figure 3 and Figure 4 are the highest-impact reviewer-facing figures.
2. Figure 7 is essential for the CAS/Braak criticism.
3. Figure 1 makes the work look new and coherent.
4. Figure 6 strengthens confidence but can be shortened if journal space is tight.
5. OASIS should stay in supplement or in a concise limitation table.
