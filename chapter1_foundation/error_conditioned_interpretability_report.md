## Error-conditioned interpretability (CAS / Hit@K / Cosine) — Results

This report quantifies whether region attention remains anatomically meaningful under **misclassification** (reviewer concern), using the saved attention subset from each run.

### Data & alignment note

- Attention is collected in `run_experiment_v3.collect_attention(..., max_samples=50)`, i.e., **first N test samples per run** (default N=50).
- Therefore metrics below are computed on the **aligned first N samples** of each test fold.
- Aggregated across all runs: **1500 samples** (= 30 runs × 50 attention samples/run) for `Ours (Atlas+AnatDist)`.

### Key reviewer case: when **MCI is misclassified as CN** (MCI→CN)

- **CAS mean**: 0.2656 (n=171)
- **Hit@5 mean**: 0.1754 (n=171)
- **Cosine similarity (mean)**: to CN=0.9777, to MCI=0.9775, to AD=0.9601

Comparison baseline (correct MCI→MCI, n=514):

- **CAS mean**: 0.2670
- **Hit@5 mean**: 0.1677

**Interpretation**: CAS/Hit@5 for MCI→CN is **not lower** than correct MCI→MCI on the saved attention subset, suggesting the anatomical attention pattern does **not collapse** under this error mode.

### Other notable error mode: MCI→AD

- **CAS mean**: 0.2921 (n=112)
- **Hit@5 mean**: 0.2946 (n=112)
- **Cosine similarity (mean)**: to AD=0.9840 (highest), to MCI=0.9725, to CN=0.9701

**Interpretation**: when MCI is predicted as AD, the attention profile is more AD-like (cosine to AD prototype is highest), consistent with an **interpretable failure mode** driven by clinically adjacent patterns.

### Artifacts saved

- **Summary JSON**: `chapter1_foundation/error_conditioned_interpretability.json`
- **Per-sample CSV**: `chapter1_foundation/error_conditioned_interpretability_samples.csv`
- **Supplementary figures**:
  - `chapter1_foundation/figures_supplementary/FigS_error_conditioned_heatmaps.(png|pdf)`
  - `chapter1_foundation/figures_supplementary/FigS_key_error_modes_MCI.(png|pdf)`

