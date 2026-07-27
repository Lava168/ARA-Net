"""Simple safety-oriented summary constraints."""

from __future__ import annotations

from typing import Sequence


def ad_to_cn_error_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    ad_indices = [index for index, label in enumerate(y_true) if label == "AD"]
    if not ad_indices:
        return 0.0
    errors = sum(1 for index in ad_indices if y_pred[index] == "CN")
    return errors / len(ad_indices)


def cn_retention_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    cn_indices = [index for index, label in enumerate(y_true) if label == "CN"]
    if not cn_indices:
        return 0.0
    retained = sum(1 for index in cn_indices if y_pred[index] == "CN")
    return retained / len(cn_indices)


def passes_floor(value: float, floor: float) -> bool:
    return float(value) >= float(floor)
