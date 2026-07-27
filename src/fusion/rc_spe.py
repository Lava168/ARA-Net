"""Risk-constrained subject-level probability ensemble (RC-SPE)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.data.probability_table import parse_float, probability_column_names


CLASS_NAMES = ("CN", "MCI", "AD")


def load_ensemble_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_ensemble_config(config)
    return config


def validate_ensemble_config(config: Mapping[str, object]) -> None:
    classes = list(config["classes"])
    base_models = list(config["base_models"])
    weights = np.asarray(config["weights"], dtype=float)
    if len(weights) != len(base_models):
        raise ValueError("Number of weights does not match number of base models.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Ensemble weights must sum to 1.")
    offsets = dict(config["offsets"])
    missing = [cls for cls in classes if cls not in offsets]
    if missing:
        raise ValueError(f"Missing offsets for classes: {missing}")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def normalize_probability_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Probabilities contain NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError("Probabilities must be non-negative.")
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Every probability vector must have positive mass.")
    return values / row_sum


def ensemble_scan_probabilities(rows: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> np.ndarray:
    """Fuse base-model probability streams into calibrated scan-level probabilities."""

    classes = list(config["classes"])
    weights = np.asarray(config["weights"], dtype=float)
    logits = np.zeros((len(rows), len(classes)), dtype=float)
    epsilon = float(config.get("epsilon", 1e-8))
    for weight, model in zip(weights, config["base_models"]):
        columns = probability_column_names(str(model), classes)
        values = np.asarray(
            [[parse_float(row[column]) for column in columns] for row in rows],
            dtype=float,
        )
        values = normalize_probability_matrix(values)
        logits += float(weight) * np.log(np.clip(values, epsilon, 1.0))
    temperature = max(float(config["temperature"]), 1e-4)
    logits /= temperature
    offsets = np.asarray([float(config["offsets"][cls]) for cls in classes], dtype=float)
    logits += offsets.reshape(1, -1)
    return softmax(logits)
