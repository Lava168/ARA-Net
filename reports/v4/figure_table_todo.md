# Figure And Table Production Status

Generated from v4 experiment outputs with reproducible scripts.

- Table script: `scripts/export_manuscript_tables.py`
- Figure script: `scripts/generate_manuscript_figures.py`
- Local table directory: `reports/v4/tables/`
- Local figure directory: `reports/v4/figures/`
- Server figure directory: `outputs/v4/manuscript_figures/`

## Tables

### Table 1. Cohort and split summary - done

Outputs:

- `reports/v4/tables/table1_cohort_splits.csv`
- `reports/v4/tables/table1_cohort_splits.md`

Purpose: answer Reviewer 4 concerns about subject inclusion, scan selection, and leakage-free split design.

### Table 2. Main classification results - done

Outputs:

- `reports/v4/tables/table2_classification.csv`
- `reports/v4/tables/table2_classification.md`

Required rows:

- Old v3 ensemble on AIBL
- Old v3 ensemble on IXI
- Atlas-only HGB on AIBL heldout
- Cascade RF-logreg on AIBL heldout
- ADNI-only hybrid RF on AIBL heldout
- Recommended atlas+clinical HGB on AIBL heldout and IXI
- Clinical-only RF on AIBL heldout
- Biomarker-enhanced HGB on AIBL heldout

### Table 3. Candidate model ranking / replicate stability - done

Outputs:

- `reports/v4/tables/table3_replicates.csv`
- `reports/v4/tables/table3_replicates.md`

Related narrative source: `reports/v4/hybrid_candidate_ranking.md`.

Purpose: show that model selection was not cherry-picked and clarify ADNI-only versus AIBL-adapted protocols.

### Table 4. Neurodegeneration consistency validation - done

Outputs:

- `reports/v4/tables/table4_neurodegeneration.csv`
- `reports/v4/tables/table4_neurodegeneration.md`

Rows:

- AIBL heldout
- AIBL adaptation + heldout
- All labeled AD
- ADNI validation + internal test

Columns:

- AD-key score
- Uniform null
- Delta
- Bootstrap CI
- Permutation p

## Figures

### Figure 1. Revised study design - done

Outputs:

- `reports/v4/figures/figure1_revised_study_design.png`
- `reports/v4/figures/figure1_revised_study_design.pdf`

Panel A: cohorts and subject-level splits.

Panel B: atlas MRI feature extraction.

Panel C: core clinical variable integration.

Panel D: AIBL heldout / IXI healthy negative-control evaluation.

Panel E: atlas neurodegeneration consistency analysis.

### Figure 2. External classification improvement - done

Outputs:

- `reports/v4/figures/figure2_external_classification_improvement.png`
- `reports/v4/figures/figure2_external_classification_improvement.pdf`

Bar chart:

- AIBL BAcc
- AIBL macro AUC
- IXI CN retention

Models:

- Old v3
- Atlas-only HGB
- Cascade RF-logreg
- Recommended atlas+clinical HGB
- Clinical-only RF

### Figure 3. AIBL heldout confusion matrices - done

Outputs:

- `reports/v4/figures/figure3_aibl_confusion_matrices.png`
- `reports/v4/figures/figure3_aibl_confusion_matrices.pdf`

Two matrices:

- Recommended atlas+clinical HGB
- Clinical-only RF comparator

Add per-class recall next to each matrix.

### Figure 4. Neurodegeneration consistency score - done

Outputs:

- `reports/v4/figures/figure4_neurodegeneration_consistency.png`
- `reports/v4/figures/figure4_neurodegeneration_consistency.pdf`

Panel A: AD-key score versus uniform null for AIBL heldout.

Panel B: bootstrap CI plot.

Panel C: region-level disease gradient plot for hippocampus, amygdala, and ventricles.

### Figure 5. Honest stress-test / limitation figure - done

Outputs:

- `reports/v4/figures/figure5_oasis_stress_test.png`
- `reports/v4/figures/figure5_oasis_stress_test.pdf`

Show OASIS results and label them explicitly as an unresolved transfer stress test.

This figure is optional but useful because it preempts reviewer concerns about selective reporting.

## Completed Script Needs

1. Generate publication-style CSV tables from JSON/MD summaries - done.
2. Generate Figure 2 bar chart from summary JSON - done.
3. Generate AIBL confusion matrices from prediction CSV files - done.
4. Generate neurodegeneration consistency plots from biomarker summary JSON - done.
5. Export all figures as PDF and 300+ dpi PNG - done.

## Next Integration Steps

1. Insert Tables 1-4 and Figures 1-5 into the rewritten manuscript skeleton.
2. Convert figure captions from `reports/v4/figures/figure_captions.json` into manuscript-ready captions.
3. Keep Figure 5 in the main manuscript or supplement depending on target journal space limits.
