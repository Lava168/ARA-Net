"""Public RC-SPE evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.aggregation import aggregate_subject_probabilities
from src.data import read_csv_rows, validate_probability_table, write_csv_rows
from src.evaluation.metrics import classification_metrics
from src.fusion import ensemble_scan_probabilities, load_ensemble_config


def format_scan_predictions(
    rows: Sequence[Mapping[str, object]],
    probs: np.ndarray,
    classes: Sequence[str],
    clinical_use_notice: str = "",
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for row, prob in zip(rows, probs):
        pred_idx = int(np.argmax(prob))
        ordered = sorted([float(x) for x in prob], reverse=True)
        payload: dict[str, object] = {
            "subject_id": row["subject_id"],
            "scan_id": row.get("scan_id", ""),
            "prediction_unit": "scan",
            "predicted_label": classes[pred_idx],
            "confidence": ordered[0],
            "margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
            "clinical_use_notice": clinical_use_notice,
        }
        for key in ("dataset", "split", "y_true"):
            if key in row:
                payload[key] = row.get(key, "")
        for cls, value in zip(classes, prob):
            payload[f"prob_{cls}"] = float(value)
        predictions.append(payload)
    return predictions


def run_rc_spe_evaluation(
    input_csv: Path,
    config_json: Path,
    output_csv: Path,
    unit: str = "subject",
    metrics_json: Path | None = None,
) -> dict[str, object]:
    config = load_ensemble_config(config_json)
    rows = read_csv_rows(input_csv)
    validate_probability_table(rows, config)
    classes = list(config["classes"])
    scan_probs = ensemble_scan_probabilities(rows, config)
    if unit == "scan":
        predictions = format_scan_predictions(rows, scan_probs, classes, config.get("clinical_use_notice", ""))
        metric_probs = scan_probs
    elif unit == "subject":
        predictions, metric_probs = aggregate_subject_probabilities(rows, scan_probs, classes)
    else:
        raise ValueError("unit must be 'scan' or 'subject'")

    write_csv_rows(output_csv, predictions)
    result: dict[str, object] = {
        "input_csv": str(input_csv),
        "config_json": str(config_json),
        "output_csv": str(output_csv),
        "unit": unit,
        "n_predictions": len(predictions),
    }
    labels = [str(row.get("y_true", "")) for row in predictions]
    if labels and all(label in classes for label in labels):
        result["metrics"] = classification_metrics(labels, metric_probs, classes)
    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
