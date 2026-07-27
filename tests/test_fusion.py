from __future__ import annotations

import numpy as np

from src.fusion.rc_spe import ensemble_scan_probabilities


def test_rc_spe_probabilities_sum_to_one():
    config = {
        "classes": ["CN", "MCI", "AD"],
        "base_models": ["m1", "m2"],
        "weights": [0.6, 0.4],
        "offsets": {"CN": 0.0, "MCI": 0.0, "AD": 0.0},
        "temperature": 1.0,
        "epsilon": 1e-8,
    }
    rows = [
        {
            "subject_id": "s1",
            "m1__prob_CN": "0.80",
            "m1__prob_MCI": "0.15",
            "m1__prob_AD": "0.05",
            "m2__prob_CN": "0.70",
            "m2__prob_MCI": "0.20",
            "m2__prob_AD": "0.10",
        }
    ]
    probs = ensemble_scan_probabilities(rows, config)
    assert probs.shape == (1, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert int(probs.argmax(axis=1)[0]) == 0
