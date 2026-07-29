# Gap analysis vs `atlas-guided-ad-staging/` checklist

Compared against GitHub `Lava168/ARA-Net` `origin/master` and manuscript
`manuscript(2).docx` (ARA-Net / RC-SPE).

## Already present on GitHub (no structural gap)

| Spec item | Status |
|-----------|--------|
| README, LICENSE, CITATION.cff, CODE_OF_CONDUCT, environment.yml, requirements, .gitignore | Present |
| configs/{adni_development,aibl_adaptation,aibl_heldout,ixi_specificity,oasis_stress_test}.yaml | Present |
| src/{data,atlas,models,fusion,calibration,constraints,aggregation,evaluation,interpretation} | Present |
| scripts/{prepare_features,train_base_models,fit_rc_spe,evaluate_*,reproduce_*} | Present |
| data/README + example_metadata + synthetic/ | Present |
| tests/{test_fusion,test_subject_aggregation,test_metrics} | Present |
| outputs/expected_results/ | Present |

## Gaps filled in this branch

| Item | Action |
|------|--------|
| Thin `src/atlas/regions.py` | Expanded to full 21-region FS-lite atlas + secondary pathology |
| Missing `src/atlas/features.py` | Added synthetic feature helpers used by `prepare_features.py` |
| Thin `src/models/probability_streams.py` | Added stream metadata for six locked RC-SPE streams |
| Thin `src/interpretation` | Added AD-key concentration score helper (null 6/21) |
| Thin `src/calibration` | Added NLL + ECE helpers |
| Package identity | Added `pyproject.toml` name `atlas-guided-ad-staging`, `CONTRIBUTING.md`, `docs/SUBMISSION_LAYOUT.md` |
| Tests | Added `tests/test_atlas_interpretation.py` |

## Intentionally out of public package

- Restricted MRI / FastSurfer volumes
- Row-level AIBL/ADNI subject prediction CSVs
- Private trained HGB/RF checkpoints (public path uses locked JSON RC-SPE head)
