#!/usr/bin/env python3
"""Merge sharded ARA-Net external-validation outputs.

Each shard evaluates a subset of checkpoints and writes a per-checkpoint
prediction CSV. This script averages probabilities over all checkpoint
predictions per subject, recomputes ensemble metrics, and writes one final
reviewer-grade JSON/CSV package.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        mat[int(true), int(pred)] += 1
    return mat


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true_binary == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return ece


def classification_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, object]:
    y_true_np = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true_np, y_pred, len(CLASS_NAMES))
    support = cm.sum(axis=1)
    pred_counts = Counter(int(x) for x in y_pred)

    per_class: Dict[str, Dict[str, float]] = {}
    recalls_present = []
    f1_present = []
    f1_all = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(support[idx]),
        }
        f1_all.append(f1)
        if support[idx] > 0:
            recalls_present.append(recall)
            f1_present.append(f1)

    aucs: Dict[str, object] = {}
    valid_aucs = []
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y_true_np == idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc = float("nan")
        elif roc_auc_score is not None:
            auc = float(roc_auc_score(y_bin, probs[:, idx]))
        else:
            auc = binary_auc(y_bin, probs[:, idx])
        aucs[name] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)

    clipped = np.clip(probs, 1e-8, 1.0)
    one_hot = np.eye(len(CLASS_NAMES))[y_true_np]
    metrics = {
        "n_samples": int(len(y_true_np)),
        "label_counts": {CLASS_NAMES[i]: int(v) for i, v in enumerate(support) if v > 0},
        "prediction_distribution": {CLASS_NAMES[i]: int(pred_counts.get(i, 0)) for i in range(len(CLASS_NAMES))},
        "accuracy": float((y_pred == y_true_np).mean()),
        "balanced_accuracy_present": float(np.mean(recalls_present)) if recalls_present else None,
        "macro_f1_present": float(np.mean(f1_present)) if f1_present else None,
        "macro_f1_all": float(np.mean(f1_all)),
        "macro_auc_ovr_valid": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "nll": float(-np.log(clipped[np.arange(len(y_true_np)), y_true_np]).mean()),
        "brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_15bin": expected_calibration_error(y_true_np, probs),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    if len(set(y_true_np.tolist())) == 1 and int(y_true_np[0]) == 0:
        metrics["ixi_cn_retention_rate"] = metrics["accuracy"]
        metrics["ixi_false_impairment_rate"] = float(1.0 - metrics["accuracy"])
    return metrics


def load_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def read_prediction_csv(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_prediction_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = [
        "dataset", "checkpoint", "subject_id", "y_true", "y_pred",
        "prob_CN", "prob_MCI", "prob_AD",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge(args: argparse.Namespace) -> None:
    shard_jsons = sorted(Path().glob(args.shard_json_glob))
    if not shard_jsons:
        raise FileNotFoundError(f"No shard JSONs matched: {args.shard_json_glob}")

    shard_data = [load_json(path) for path in shard_jsons]
    pred_csvs = [path.with_suffix(".per_checkpoint_predictions.csv") for path in shard_jsons]
    missing = [str(path) for path in pred_csvs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing per-checkpoint CSVs: {missing}")
    pred_rows = read_prediction_csv(pred_csvs)

    grouped: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    checkpoint_names = set()
    for row in pred_rows:
        dataset = row["dataset"]
        subject = row["subject_id"]
        checkpoint_names.add(row["checkpoint"])
        prob = np.array([float(row["prob_CN"]), float(row["prob_MCI"]), float(row["prob_AD"])])
        entry = grouped[dataset].setdefault(
            subject,
            {
                "y_true": CLASS_TO_IDX[row["y_true"]],
                "probs": [],
            },
        )
        entry["probs"].append(prob)

    ensemble: Dict[str, object] = {}
    ensemble_rows: List[Dict[str, object]] = []
    for dataset, subjects in sorted(grouped.items()):
        y_true = []
        probs = []
        for subject_id, entry in sorted(subjects.items()):
            subject_probs = np.stack(entry["probs"], axis=0)
            mean_prob = subject_probs.mean(axis=0)
            y_true.append(int(entry["y_true"]))
            probs.append(mean_prob)
            ensemble_rows.append({
                "dataset": dataset,
                "checkpoint": "ensemble_mean_probability",
                "subject_id": subject_id,
                "y_true": CLASS_NAMES[int(entry["y_true"])],
                "y_pred": CLASS_NAMES[int(mean_prob.argmax())],
                "prob_CN": float(mean_prob[0]),
                "prob_MCI": float(mean_prob[1]),
                "prob_AD": float(mean_prob[2]),
            })
        ensemble[dataset] = classification_metrics(y_true, np.stack(probs, axis=0))

    per_checkpoint: Dict[str, object] = defaultdict(dict)
    for data in shard_data:
        for dataset, metrics_by_ckpt in data.get("per_checkpoint", {}).items():
            per_checkpoint[dataset].update(metrics_by_ckpt)

    datasets: Dict[str, object] = {}
    for data in shard_data:
        datasets.update(data.get("datasets", {}))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_json.with_suffix(".ensemble_predictions.csv")
    final = {
        "merged_from": [str(path) for path in shard_jsons],
        "n_shards": len(shard_jsons),
        "n_checkpoints": len(checkpoint_names),
        "checkpoints": sorted(checkpoint_names),
        "datasets": datasets,
        "model_meta": shard_data[0].get("model_meta"),
        "per_checkpoint": per_checkpoint,
        "ensemble": ensemble,
    }
    with out_json.open("w") as handle:
        json.dump(final, handle, indent=2)
    write_prediction_csv(out_csv, ensemble_rows)
    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")
    for dataset, metrics in ensemble.items():
        print(
            f"[ensemble] {dataset}: "
            f"Acc={metrics['accuracy']:.4f}, "
            f"BAcc={metrics['balanced_accuracy_present']:.4f}, "
            f"AUC={metrics['macro_auc_ovr_valid']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-json-glob", required=True)
    parser.add_argument("--output-json", required=True)
    merge(parser.parse_args())


if __name__ == "__main__":
    main()
