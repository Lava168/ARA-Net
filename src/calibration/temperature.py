"""Temperature scaling and class-offset calibration."""

from __future__ import annotations

import numpy as np

from src.fusion.rc_spe import softmax


def apply_temperature_and_offsets(logits: np.ndarray, offsets: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-4)
    calibrated = np.asarray(logits, dtype=float) / temperature
    calibrated = calibrated + np.asarray(offsets, dtype=float).reshape(1, -1)
    return softmax(calibrated)
