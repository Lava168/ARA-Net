"""Temperature scaling and class-offset calibration."""

from __future__ import annotations

import math

import numpy as np

from src.fusion.rc_spe import softmax


def apply_temperature_and_offsets(logits: np.ndarray, offsets: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-4)
    calibrated = np.asarray(logits, dtype=float) / temperature
    calibrated = calibrated + np.asarray(offsets, dtype=float).reshape(1, -1)
    return softmax(calibrated)


def negative_log_likelihood(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    """Mean categorical NLL for calibrated probability rows."""
    y = np.asarray(y_true_idx, dtype=int)
    p = np.asarray(probs, dtype=float)
    rows = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(p[rows, y], 1e-12, 1.0))))


def expected_calibration_error(y_true_idx: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Simple confidence ECE (max-prob bins)."""
    y = np.asarray(y_true_idx, dtype=int)
    p = np.asarray(probs, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
        if not np.any(mask):
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * (mask.mean())
    return float(ece) if math.isfinite(ece) else float("nan")
