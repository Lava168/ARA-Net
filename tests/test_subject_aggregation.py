from __future__ import annotations

import numpy as np

from src.aggregation import aggregate_subject_probabilities


def test_subject_aggregation_averages_repeated_scans():
    rows = [
        {"subject_id": "s1", "scan_id": "a", "y_true": "AD"},
        {"subject_id": "s1", "scan_id": "b", "y_true": "AD"},
        {"subject_id": "s2", "scan_id": "a", "y_true": "CN"},
    ]
    probs = np.asarray([
        [0.10, 0.20, 0.70],
        [0.20, 0.20, 0.60],
        [0.80, 0.10, 0.10],
    ])
    out_rows, out_probs = aggregate_subject_probabilities(rows, probs, ["CN", "MCI", "AD"])
    assert len(out_rows) == 2
    assert out_rows[0]["subject_id"] == "s1"
    assert out_rows[0]["predicted_label"] == "AD"
    assert np.allclose(out_probs[0], [0.15, 0.20, 0.65])
