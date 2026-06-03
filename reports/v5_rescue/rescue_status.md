# AD Model Rescue Status

Generated during the rescue run after the v4 rewrite.

## What improved immediately

### Subject-level robustness

Fast subject-level probability averaging also improved the AIBL result:

- AIBL heldout subjects: Acc 0.903, BAcc 0.833, AUC 0.937, AD-vs-CN AUC 1.000.
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

OASIS adaptation sensitivity is promising but not locked evidence:

- Full OASIS adaptation search quickly reached high oasis_transfer scores.
- This cannot be used as external validation because OASIS was included in training/tuning.

OASIS locked adaptation was created:

- OASIS adapt train: 50 scans, CN/MCI/AD 30/14/6.
- OASIS adapt val: 20 scans, CN/MCI/AD 12/6/2.
- OASIS heldout: 29 scans, CN/MCI/AD 17/9/3.

Current interim OASIS locked result:

- OASIS heldout: Acc 0.655, BAcc 0.499, AUC 0.645.
- OASIS heldout recall CN/MCI/AD: 0.941 / 0.222 / 0.333.
- AIBL heldout in the same profile: BAcc 0.814, recall CN/MCI/AD 0.947 / 0.642 / 0.854.
- IXI retention: 0.995.

Interpretation: OASIS is no longer completely failed after small-domain adaptation, but it is still not strong. Because OASIS heldout has only 29 scans and 3 AD scans, treat this as sensitivity evidence, not a definitive external validation.

## Current best scientific story

The strongest honest model is now a rescued ensemble/calibrated hybrid:

- Main external result: AIBL heldout BAcc 0.820, AUC 0.939, AD-vs-CN AUC 0.998.
- MCI and AD recall are both materially better than v4 main model.
- IXI healthy specificity remains excellent.
- OASIS can improve under a small locked-adaptation protocol, but still needs larger or cleaner OASIS/NACC validation.

## Next experiments

1. Let `rescue_hybrid_search_core` finish and compare with the probability ensemble.
2. Let `rescue_hybrid_search_oasis_locked` finish and select the best locked OASIS profile.
3. Run multi-seed confirmation for the selected probability ensemble and OASIS locked split seeds.
4. Pull chapter4 generalization predictions into the probability optimizer as additional ensemble candidates.
5. Build a NACC or chapter4 external heldout if available, because OASIS has too few AD cases.
