from __future__ import annotations

import numpy as np

from src.evaluation import classification_metrics


def test_classification_metrics_reports_balanced_accuracy():
    y_true = ["CN", "MCI", "AD"]
    probs = np.asarray([
        [0.90, 0.08, 0.02],
        [0.10, 0.80, 0.10],
        [0.05, 0.20, 0.75],
    ])
    metrics = classification_metrics(y_true, probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
