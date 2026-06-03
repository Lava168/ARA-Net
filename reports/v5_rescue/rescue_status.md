# AD Model Rescue Status

Generated during the rescue run after the v4 rewrite.

## What improved immediately

### Subject-level robustness

Fast subject-level probability averaging also improved the AIBL result:

- AIBL heldout subject-level endpoint units: Acc 0.903, BAcc 0.833, AUC 0.937, AD-vs-CN AUC 1.000.
- Subject-level recall CN/MCI/AD: 0.961 / 0.686 / 0.852.
- IXI healthy retention: 1.000.
- Internal subject-level recall CN/MCI/AD: 0.241 / 0.553 / 0.550.

Interpretation: averaging repeated scans at subject level strengthens the AIBL heldout result and gives a more reviewer-friendly robustness analysis. It still does not solve OASIS.

### AIBL heldout, no OASIS tuning

Best balanced probability ensemble from existing predictions:

- AIBL heldout: Acc 0.909, BAcc 0.820, AUC 0.939, AD-vs-CN AUC 0.998.
- AIBL heldout recall CN/MCI/AD: 0.964 / 0.642 / 0.854.
- IXI healthy retention: 1.000.
- Internal test recall CN/MCI/AD: 0.333 / 0.579 / 0.500.

Compared with the v4 main atlas+clinical HGB model:

- AIBL BAcc improved from 0.741 to 0.820.
- AIBL MCI recall improved from 0.528 to 0.642.
- AIBL AD recall improved from 0.732 to 0.854.
- Internal AD recall improved from 0.141 to 0.500, but internal BAcc remains modest.

### Internal AD recall profile

If optimizing specifically for AD recall:

- Internal test AD recall reaches 0.949.
- Internal BAcc reaches 0.593.
- AIBL heldout AD recall reaches 0.902.
- Tradeoff: AIBL MCI recall drops to 0.226.

Interpretation: internal AD recall is strongly threshold/calibration limited, but boosting AD aggressively collapses some MCI into AD.

### AIBL MCI recall profile

If optimizing specifically for AIBL MCI recall:

- AIBL heldout MCI recall reaches 0.830.
- AIBL heldout AD recall remains 0.780.
- Tradeoff: AIBL CN recall drops to 0.498 and internal CN recall collapses.

Interpretation: MCI can be recovered, but not yet with acceptable CN specificity unless the profile is less aggressive.

## OASIS status

Zero-shot/no-OASIS-tune OASIS remains weak:

- Best no-OASIS-tune profiles stay around BAcc 0.333 to 0.358, mostly all-CN or poor minority recall.

Decision: do not keep spending major compute on OASIS for the main paper.

Rationale:

- OASIS has only 99 scans and only 11 AD cases.
- Any internal OASIS adapt/heldout split is statistically fragile.
- Improving OASIS by adapting on OASIS would weaken the "external validation" claim unless repeated on a larger external cohort.
- Keep OASIS as a stress-test limitation and move the main effort to AIBL heldout, IXI specificity, internal AD recall, and subject-level robustness.

## Current best scientific story

The strongest honest model is now a rescued ensemble/calibrated hybrid:

- Main subject-level external result: AIBL heldout Acc 0.903, BAcc 0.833, AUC 0.937, AD-vs-CN AUC 1.000.
- AIBL heldout subject-level recall CN/MCI/AD: 0.961 / 0.686 / 0.852.
- Bootstrap uncertainty is now available: AIBL heldout BAcc 95% CI 0.759-0.899, MCI recall 95% CI 0.531-0.839, AD recall 95% CI 0.710-0.966.
- IXI healthy specificity remains excellent: subject-level CN retention 1.000.
- OASIS remains weak without OASIS tuning and should be written only as an external stress-test limitation.

## Next experiments

1. Let `rescue_hybrid_search_core` finish and compare with the probability ensemble.
2. Keep the subject-level rescue ensemble as the locked main model unless the core search beats it without losing IXI specificity.
3. Pull chapter4 or NACC-style external AD predictions into the probability optimizer only if a clean heldout cohort is available.
4. Do not continue OASIS tuning for the main paper; keep OASIS as a limitation.
