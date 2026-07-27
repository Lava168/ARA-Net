"""Small dependency-light classification metrics for CN/MCI/AD staging."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def binary_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = y_true_binary == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(
    y_true: Sequence[str],
    probs: np.ndarray,
    classes: Sequence[str] = ("CN", "MCI", "AD"),
) -> dict[str, object]:
    classes = list(classes)
    label_to_idx = {label: index for index, label in enumerate(classes)}
    y = np.asarray([label_to_idx[label] for label in y_true], dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for yi, pi in zip(y, pred):
        cm[int(yi), int(pi)] += 1

    recalls = []
    f1s = []
    per_class: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(classes):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(cm[idx, :].sum())
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": support,
        }
        if support:
            recalls.append(recall)
            f1s.append(f1)

    aucs: dict[str, float | None] = {}
    valid_aucs = []
    for idx, label in enumerate(classes):
        auc = binary_auc((y == idx).astype(int), probs[:, idx])
        aucs[label] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)

    out: dict[str, object] = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_auc_ovr": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": {classes[i]: int((pred == i).sum()) for i in range(len(classes))},
        "per_class": per_class,
    }
    if "CN" in label_to_idx and "AD" in label_to_idx:
        cn = y == label_to_idx["CN"]
        ad = y == label_to_idx["AD"]
        if cn.sum() and ad.sum():
            score = probs[:, label_to_idx["AD"]] - probs[:, label_to_idx["CN"]]
            yy = np.concatenate([np.zeros(int(cn.sum())), np.ones(int(ad.sum()))])
            ss = np.concatenate([score[cn], score[ad]])
            out["ad_vs_cn_auc"] = binary_auc(yy, ss)
    if len(set(y.tolist())) == 1 and len(y) and classes[int(y[0])] == "CN":
        out["cn_retention_rate"] = out["accuracy"]
        out["false_impairment_rate"] = float(1.0 - float(out["accuracy"]))
    return out
