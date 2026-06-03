#!/usr/bin/env python3
"""Research inference CLI for the ARA-Net v6 ensemble.

This is a deployment-oriented research wrapper around the final probability
ensemble. It accepts base-model class probabilities and returns calibrated
scan-level and subject-level predictions. It does not load raw MRI data and is
not intended for clinical diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


REQUIRED_META_COLUMNS = ["subject_id"]
OPTIONAL_META_COLUMNS = ["scan_id", "dataset", "split", "y_true"]


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data["classes"]
    weights = np.asarray(data["weights"], dtype=float)
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("Ensemble weights must sum to 1.")
    if len(weights) != len(data["base_models"]):
        raise ValueError("Number of weights does not match number of base models.")
    for cls in classes:
        if cls not in data["offsets"]:
            raise ValueError(f"Missing class offset for {cls}")
    return data


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def probability_column_names(model: str, classes: Sequence[str]) -> List[str]:
    return [f"{model}__prob_{cls}" for cls in classes]


def validate_rows(rows: Sequence[dict], config: dict) -> None:
    if not rows:
        raise ValueError("Input file has no rows.")
    columns = set(rows[0])
    for col in REQUIRED_META_COLUMNS:
        if col not in columns:
            raise ValueError(f"Missing required metadata column: {col}")
    missing = []
    for model in config["base_models"]:
        for col in probability_column_names(model, config["classes"]):
            if col not in columns:
                missing.append(col)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise ValueError(f"Missing required probability columns: {preview}{suffix}")


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        raise ValueError("Probabilities contain NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError("Probabilities must be non-negative.")
    row_sum = values.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Each probability vector must have positive sum.")
    return values / row_sum


def ensemble_scan_probabilities(rows: Sequence[dict], config: dict) -> np.ndarray:
    classes = config["classes"]
    logits = np.zeros((len(rows), len(classes)), dtype=float)
    eps = float(config.get("epsilon", 1e-8))
    for weight, model in zip(config["weights"], config["base_models"]):
        values = np.asarray(
            [
                [parse_float(row[col]) for col in probability_column_names(model, classes)]
                for row in rows
            ],
            dtype=float,
        )
        values = normalize_probabilities(values)
        logits += float(weight) * np.log(np.clip(values, eps, 1.0))
    logits /= max(float(config["temperature"]), 1e-4)
    offsets = np.asarray([float(config["offsets"][cls]) for cls in classes], dtype=float)
    logits += offsets.reshape(1, -1)
    return softmax(logits)


def format_prediction_rows(rows: Sequence[dict], probs: np.ndarray, config: dict) -> List[dict]:
    classes = config["classes"]
    out = []
    for row, prob in zip(rows, probs):
        pred_idx = int(np.argmax(prob))
        ordered = sorted([float(x) for x in prob], reverse=True)
        payload = {
            "subject_id": row["subject_id"],
            "scan_id": row.get("scan_id", ""),
            "prediction_unit": "scan",
            "predicted_label": classes[pred_idx],
            "confidence": ordered[0],
            "margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
            "clinical_use_notice": config["clinical_use_notice"],
        }
        for meta in ["dataset", "split", "y_true"]:
            if meta in row:
                payload[meta] = row.get(meta, "")
        for cls, value in zip(classes, prob):
            payload[f"prob_{cls}"] = float(value)
        out.append(payload)
    return out


def aggregate_subjects(scan_rows: Sequence[dict], scan_probs: np.ndarray, config: dict) -> List[dict]:
    classes = config["classes"]
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(scan_rows):
        grouped[row["subject_id"]].append(idx)
    out = []
    for subject_id, indices in sorted(grouped.items()):
        prob = scan_probs[indices].mean(axis=0)
        pred_idx = int(np.argmax(prob))
        ordered = sorted([float(x) for x in prob], reverse=True)
        first = scan_rows[indices[0]]
        payload = {
            "subject_id": subject_id,
            "scan_id": f"{subject_id}__subject_mean",
            "prediction_unit": "subject",
            "n_scans": len(indices),
            "predicted_label": classes[pred_idx],
            "confidence": ordered[0],
            "margin": ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0],
            "clinical_use_notice": config["clinical_use_notice"],
        }
        for meta in ["dataset", "split", "y_true"]:
            if meta in first:
                values = {scan_rows[i].get(meta, "") for i in indices}
                payload[meta] = first.get(meta, "") if len(values) == 1 else "mixed"
        for cls, value in zip(classes, prob):
            payload[f"prob_{cls}"] = float(value)
        out.append(payload)
    return out


def json_payload(rows: Sequence[dict], config: dict) -> dict:
    return {
        "model": config["name"],
        "version": config["version"],
        "intended_use": config["intended_use"],
        "clinical_use_notice": config["clinical_use_notice"],
        "predictions": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with base-model probability columns.")
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--output", type=Path, required=True, help="Output CSV or JSON path.")
    parser.add_argument("--unit", choices=["scan", "subject"], default="subject")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_csv_rows(args.input_csv)
    validate_rows(rows, config)
    scan_probs = ensemble_scan_probabilities(rows, config)
    if args.unit == "scan":
        predictions = format_prediction_rows(rows, scan_probs, config)
    else:
        predictions = aggregate_subjects(rows, scan_probs, config)

    if args.output.suffix.lower() == ".json":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_payload(predictions, config), indent=2), encoding="utf-8")
    else:
        write_csv(args.output, predictions)
    print(f"[saved] {args.output}")
    print(config["clinical_use_notice"])


if __name__ == "__main__":
    main()
