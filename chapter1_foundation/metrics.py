"""Comprehensive Evaluation Metrics for AD Classification."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

CLASS_NAMES = ["CN", "MCI", "AD"]

def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.)
    specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.)
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "support": cm.sum(axis=1)}

def accuracy(y_true, y_pred):
    return float((np.array(y_true) == np.array(y_pred)).mean())

def _binary_auc(y_true_binary, scores):
    order = np.argsort(-scores)
    ys = y_true_binary[order]
    n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
    if n_pos == 0 or n_neg == 0: return 0.5
    tp = fp = auc = 0.
    tpr_prev = fpr_prev = 0.; prev = -np.inf
    for i in range(len(ys)):
        if scores[order[i]] != prev:
            tpr, fpr = tp / n_pos, fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr; prev = scores[order[i]]
        if ys[i] == 1: tp += 1
        else: fp += 1
    tpr, fpr = tp / n_pos, fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
    return float(auc)

def multiclass_auc_ovr(y_true, y_prob, n_classes=3):
    aucs = {}
    for c in range(n_classes):
        aucs[CLASS_NAMES[c]] = _binary_auc((y_true == c).astype(int), y_prob[:, c])
    aucs["macro"] = float(np.mean([v for k, v in aucs.items() if k != "macro"]))
    return aucs

def roc_curve(y_true_binary, scores, n_thresholds=200):
    thresholds = np.linspace(scores.max() + 1e-8, scores.min() - 1e-8, n_thresholds)
    n_pos, n_neg = y_true_binary.sum(), len(y_true_binary) - y_true_binary.sum()
    fprs, tprs = [], []
    for th in thresholds:
        pred = (scores >= th).astype(int)
        fprs.append(((pred == 1) & (y_true_binary == 0)).sum() / max(n_neg, 1))
        tprs.append(((pred == 1) & (y_true_binary == 1)).sum() / max(n_pos, 1))
    return np.array(fprs), np.array(tprs)

def classification_report(y_true, y_pred, y_prob=None, class_names=None):
    class_names = class_names or CLASS_NAMES
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    pcm = per_class_metrics(cm)
    report = {
        "accuracy": accuracy(y_true, y_pred), "confusion_matrix": cm,
        "per_class": {class_names[i]: {k: float(v[i]) for k, v in pcm.items()} for i in range(n)},
        "macro": {k: float(v.mean()) for k, v in pcm.items() if k != "support"},
        "n_samples": len(y_true),
    }
    if y_prob is not None:
        report["auc"] = multiclass_auc_ovr(y_true, y_prob, n)
    return report

def bootstrap_ci(y_true, y_pred, metric_fn=accuracy, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed); n = len(y_true); scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores); alpha = (1 - ci) / 2
    return float(scores.mean()), float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))
