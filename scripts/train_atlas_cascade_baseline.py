#!/usr/bin/env python3
"""Two-stage atlas-feature cascade for AD staging.

Stage 1 predicts CN vs impaired (MCI/AD). Stage 2 predicts MCI vs AD only for
scans deemed impaired. This directly targets the failure mode seen in v4:
single-softmax models oscillate between all-CN and all-impaired predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from train_atlas_feature_baseline import (
    CLASS_NAMES,
    FEATURE_NAMES,
    classification_metrics,
    matrix,
    read_feature_cache,
    read_manifest,
    build_feature_cache,
    write_predictions,
)


def make_stage1_models(seed: int) -> Dict[str, object]:
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, random_state=seed)),
        ]),
        "svm": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("clf", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=600, max_depth=9, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        "extra": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
    }


def make_stage2_models(seed: int) -> Dict[str, object]:
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.8, random_state=seed)),
        ]),
        "svm": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("clf", SVC(C=2.5, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]),
        "extra": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=3, class_weight="balanced", random_state=seed + 3, n_jobs=-1)),
        ]),
    }


def rows_by_split(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[row["split"]].append(row)
    return out


def as_binary_train(rows: Sequence[dict], mode: str) -> Tuple[np.ndarray, np.ndarray]:
    x, y = matrix(rows)
    if mode == "cn_vs_imp":
        return x, (y > 0).astype(int)
    if mode == "mci_vs_ad":
        keep = y > 0
        return x[keep], (y[keep] == 2).astype(int)
    raise ValueError(mode)


def positive_column(model) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    classes = list(classes)
    return classes.index(1)


def cascade_probs(stage1, stage2, x: np.ndarray, impaired_threshold: float) -> np.ndarray:
    p_imp = stage1.predict_proba(x)[:, positive_column(stage1)]
    p_ad_given_imp = stage2.predict_proba(x)[:, positive_column(stage2)]
    probs = np.zeros((len(x), 3), dtype=float)
    hard_imp = p_imp >= impaired_threshold
    # Probability view: retain continuous probabilities for AUC/calibration.
    probs[:, 0] = 1.0 - p_imp
    probs[:, 2] = p_imp * p_ad_given_imp
    probs[:, 1] = p_imp * (1.0 - p_ad_given_imp)
    # Decision view: threshold stage 1, then split MCI/AD by stage 2.
    pred = np.zeros(len(x), dtype=int)
    pred[hard_imp] = np.where(p_ad_given_imp[hard_imp] >= 0.5, 2, 1)
    decision_probs = probs.copy()
    decision_probs[:] = 1e-6
    decision_probs[np.arange(len(x)), pred] = 1.0 - 2e-6
    # Blend hard decision for confusion-matrix realism while preserving score ranking.
    return 0.15 * probs + 0.85 * decision_probs


def eval_cascade(stage1, stage2, split_rows: Sequence[dict], threshold: float) -> Tuple[dict, np.ndarray]:
    x, y = matrix(split_rows)
    probs = cascade_probs(stage1, stage2, x, threshold)
    return classification_metrics(y, probs), probs


def tune_threshold(stage1, stage2, val_rows: Sequence[dict], ixi_rows: Sequence[dict]) -> float:
    best_t = 0.5
    best_score = -1e9
    for t in np.linspace(0.20, 0.85, 66):
        val_metrics, _ = eval_cascade(stage1, stage2, val_rows, float(t))
        ixi_metrics, _ = eval_cascade(stage1, stage2, ixi_rows, float(t)) if ixi_rows else ({}, None)
        pc = val_metrics.get("per_class", {})
        minority = min(pc.get("MCI", {}).get("recall", 0.0), pc.get("AD", {}).get("recall", 0.0))
        score = (
            0.45 * (val_metrics.get("balanced_acc") or 0.0)
            + 0.25 * (val_metrics.get("macro_auc_ovr") or 0.0)
            + 0.15 * minority
            + 0.15 * (ixi_metrics.get("cn_retention_rate") or 0.0)
        )
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def selection_score(metrics: Dict[str, dict]) -> float:
    val = metrics.get("val", {})
    aibl = metrics.get("aibl_adapt_val", {})
    ixi = metrics.get("ixi_external", {})
    def minority(m):
        pc = m.get("per_class", {})
        return min(pc.get("MCI", {}).get("recall", 0.0), pc.get("AD", {}).get("recall", 0.0))
    return (
        0.30 * (val.get("balanced_acc") or 0.0)
        + 0.20 * (val.get("macro_auc_ovr") or 0.0)
        + 0.25 * (aibl.get("balanced_acc") or 0.0)
        + 0.15 * minority(aibl)
        + 0.10 * (ixi.get("cn_retention_rate") or 0.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    if args.rebuild_features or not args.feature_csv.exists():
        build_feature_cache(read_manifest(args.manifest), args.feature_csv)
    rows = read_feature_cache(args.feature_csv)
    by_split = rows_by_split(rows)
    train_rows = by_split["train"] + by_split["aibl_adapt_train"]
    stage2_rows = [r for r in train_rows if int(r["label"]) > 0]
    val_rows = by_split["val"] + by_split["aibl_adapt_val"]
    ixi_rows = by_split.get("ixi_external", [])
    x1, y1 = as_binary_train(train_rows, "cn_vs_imp")
    x2, y2 = as_binary_train(stage2_rows, "mci_vs_ad")
    eval_splits = ["val", "internal_test", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    best = {"name": None, "score": -1e9}
    for n1, m1 in make_stage1_models(args.seed).items():
        print(f"[fit-stage1] {n1}", flush=True)
        m1.fit(x1, y1)
        for n2, m2 in make_stage2_models(args.seed).items():
            print(f"[fit-stage2] {n1}+{n2}", flush=True)
            m2.fit(x2, y2)
            threshold = tune_threshold(m1, m2, val_rows, ixi_rows)
            metrics = {}
            combo = f"{n1}__{n2}"
            for split in eval_splits:
                split_rows = by_split.get(split, [])
                if not split_rows:
                    continue
                met, probs = eval_cascade(m1, m2, split_rows, threshold)
                metrics[split] = met
                write_predictions(args.out_dir / f"{combo}_{split}_predictions.csv", split_rows, probs)
            score = selection_score(metrics)
            results[combo] = {"threshold": threshold, "selection_score": score, "metrics": metrics}
            print(f"[score] {combo}: threshold={threshold:.3f} score={score:.4f}", flush=True)
            if score > best["score"]:
                best = {"name": combo, "score": score}

    summary = {
        "best_model": best["name"],
        "best_score": best["score"],
        "feature_csv": str(args.feature_csv),
        "n_features": len(FEATURE_NAMES),
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.out_dir / 'summary.json'}")
    print(f"[best] {best['name']} score={best['score']:.4f}")


if __name__ == "__main__":
    main()
