"""Subject-level probability aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def aggregate_subject_probabilities(
    rows: Sequence[Mapping[str, object]],
    scan_probs: np.ndarray,
    classes: Sequence[str],
) -> tuple[list[dict[str, object]], np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["subject_id"])].append(index)

    out_rows: list[dict[str, object]] = []
    out_probs: list[np.ndarray] = []
    for subject_id, indices in sorted(grouped.items()):
        prob = np.asarray(scan_probs[indices], dtype=float).mean(axis=0)
        first = rows[indices[0]]
        row: dict[str, object] = {
            "subject_id": subject_id,
            "scan_id": f"{subject_id}__subject_mean",
            "prediction_unit": "subject",
            "n_scans": len(indices),
        }
        for key in ("dataset", "split", "y_true"):
            if key in first:
                values = {str(rows[i].get(key, "")) for i in indices}
                row[key] = first.get(key, "") if len(values) == 1 else "mixed"
        pred_idx = int(np.argmax(prob))
        ordered = sorted([float(x) for x in prob], reverse=True)
        row["predicted_label"] = classes[pred_idx]
        row["confidence"] = ordered[0]
        row["margin"] = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        for cls, value in zip(classes, prob):
            row[f"prob_{cls}"] = float(value)
        out_rows.append(row)
        out_probs.append(prob)
    return out_rows, np.vstack(out_probs) if out_probs else np.empty((0, len(classes)))
