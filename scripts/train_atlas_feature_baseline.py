#!/usr/bin/env python3
"""Atlas-feature baselines for robust external AD classification.

This is a deliberately strong, interpretable companion to the v4 deep model.
It extracts per-region volume and intensity statistics from the same 21-region
atlas caches, then trains calibrated classical classifiers with subject-level
splits from the v4 manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.svm import SVC
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required: {exc}")


CLASS_NAMES = ["CN", "MCI", "AD"]
FS_LABELS = [
    0,
    2, 3, 4, 10, 11, 12, 13, 16, 17, 18, 26,
    41, 42, 43, 49, 50, 51, 52, 53, 54, 58,
]
REGION_NAMES = [
    "L-WM", "L-Cortex", "L-Lat-Ventricle", "L-Thalamus", "L-Caudate",
    "L-Putamen", "L-Pallidum", "Brain-Stem", "L-Hippocampus", "L-Amygdala",
    "L-Accumbens", "R-WM", "R-Cortex", "R-Lat-Ventricle", "R-Thalamus",
    "R-Caudate", "R-Putamen", "R-Pallidum", "R-Hippocampus", "R-Amygdala",
    "R-Accumbens",
]
AD_KEY_REGIONS = {
    "L-Hippocampus", "R-Hippocampus",
    "L-Amygdala", "R-Amygdala",
    "L-Lat-Ventricle", "R-Lat-Ventricle",
}


def read_manifest(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def feature_names() -> List[str]:
    names = []
    for region in REGION_NAMES:
        names.extend([
            f"vol_{region}",
            f"mean_{region}",
            f"std_{region}",
            f"q10_{region}",
            f"q90_{region}",
        ])
    pairs = [
        ("Hippocampus", "L-Hippocampus", "R-Hippocampus"),
        ("Amygdala", "L-Amygdala", "R-Amygdala"),
        ("LatVentricle", "L-Lat-Ventricle", "R-Lat-Ventricle"),
        ("Thalamus", "L-Thalamus", "R-Thalamus"),
        ("Cortex", "L-Cortex", "R-Cortex"),
        ("WM", "L-WM", "R-WM"),
    ]
    for short, _, _ in pairs:
        names.append(f"asym_vol_{short}")
        names.append(f"asym_mean_{short}")
    names.extend([
        "ad_key_volume_sum",
        "hippocampus_volume_sum",
        "ventricle_volume_sum",
        "hippocampus_to_ventricle_ratio",
    ])
    return names


FEATURE_NAMES = feature_names()


def extract_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        image = data["image"].astype(np.float32)
        seg = data["seg"].astype(np.int32)
    brain_mask = seg > 0
    brain_n = float(brain_mask.sum()) if brain_mask.any() else float(np.prod(seg.shape))
    feats: Dict[str, float] = {}
    region_stats: Dict[str, Tuple[float, float]] = {}
    for label, region in zip(FS_LABELS[1:], REGION_NAMES):
        mask = seg == int(label)
        vol = float(mask.sum() / max(brain_n, 1.0))
        vals = image[mask]
        if vals.size:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            q10 = float(np.percentile(vals, 10))
            q90 = float(np.percentile(vals, 90))
        else:
            mean = std = q10 = q90 = np.nan
        feats[f"vol_{region}"] = vol
        feats[f"mean_{region}"] = mean
        feats[f"std_{region}"] = std
        feats[f"q10_{region}"] = q10
        feats[f"q90_{region}"] = q90
        region_stats[region] = (vol, mean)

    for short, left, right in [
        ("Hippocampus", "L-Hippocampus", "R-Hippocampus"),
        ("Amygdala", "L-Amygdala", "R-Amygdala"),
        ("LatVentricle", "L-Lat-Ventricle", "R-Lat-Ventricle"),
        ("Thalamus", "L-Thalamus", "R-Thalamus"),
        ("Cortex", "L-Cortex", "R-Cortex"),
        ("WM", "L-WM", "R-WM"),
    ]:
        lv, lm = region_stats[left]
        rv, rm = region_stats[right]
        feats[f"asym_vol_{short}"] = float((lv - rv) / (lv + rv + 1e-8))
        feats[f"asym_mean_{short}"] = float((lm - rm) / (abs(lm) + abs(rm) + 1e-8))

    ad_key_volume = sum(feats[f"vol_{region}"] for region in AD_KEY_REGIONS)
    hipp = feats["vol_L-Hippocampus"] + feats["vol_R-Hippocampus"]
    vent = feats["vol_L-Lat-Ventricle"] + feats["vol_R-Lat-Ventricle"]
    feats["ad_key_volume_sum"] = float(ad_key_volume)
    feats["hippocampus_volume_sum"] = float(hipp)
    feats["ventricle_volume_sum"] = float(vent)
    feats["hippocampus_to_ventricle_ratio"] = float(hipp / (vent + 1e-8))
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


def build_feature_cache(manifest_rows: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "subject_id", "scan_id", "label", "label_name", "path",
        *FEATURE_NAMES,
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(manifest_rows, 1):
            feats = extract_features(Path(row["path"]))
            out = {
                "dataset": row["dataset"],
                "split": row["split"],
                "subject_id": row["subject_id"],
                "scan_id": row["scan_id"],
                "label": row["label"],
                "label_name": row["label_name"],
                "path": row["path"],
            }
            out.update({name: float(value) for name, value in zip(FEATURE_NAMES, feats)})
            writer.writerow(out)
            if idx % 500 == 0:
                print(f"[features] {idx}/{len(manifest_rows)}", flush=True)
    print(f"[saved] {output_csv}", flush=True)


def read_feature_cache(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def matrix(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return x, y


def classification_metrics(y_true: Sequence[int], probs: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for yi, pi in zip(y, pred):
        cm[int(yi), int(pi)] += 1
    support = cm.sum(axis=1)
    per_class = {}
    recalls = []
    f1s = []
    for idx, name in enumerate(CLASS_NAMES):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(support[idx]),
        }
        if support[idx] > 0:
            recalls.append(recall)
            f1s.append(f1)
    valid_aucs = []
    aucs = {}
    for idx, name in enumerate(CLASS_NAMES):
        yb = (y == idx).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            auc = None
        else:
            auc = float(roc_auc_score(yb, probs[:, idx]))
            valid_aucs.append(auc)
        aucs[name] = auc
    out = {
        "n": int(len(y)),
        "label_counts": {CLASS_NAMES[i]: int(support[i]) for i in range(3) if support[i] > 0},
        "prediction_distribution": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(3)},
        "acc": float((pred == y).mean()),
        "balanced_acc": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_auc_ovr": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_class_auc_ovr": aucs,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    cn = y == 0
    ad = y == 2
    if cn.sum() and ad.sum():
        out["ad_vs_cn_auc"] = float(roc_auc_score(np.r_[np.zeros(cn.sum()), np.ones(ad.sum())], np.r_[probs[cn, 2] - probs[cn, 0], probs[ad, 2] - probs[ad, 0]]))
    if len(set(y.tolist())) == 1 and int(y[0]) == 0:
        out["cn_retention_rate"] = out["acc"]
        out["false_impairment_rate"] = float(1.0 - out["acc"])
    return out


def make_models(seed: int) -> Dict[str, object]:
    return {
        "logreg_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, multi_class="auto", random_state=seed)),
        ]),
        "svm_rbf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("clf", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]),
        "rf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=600, max_depth=8, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        "extratrees_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
        "hgb": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, l2_regularization=0.05, random_state=seed)),
        ]),
    }


def calibrate_if_possible(model, x_val: np.ndarray, y_val: np.ndarray):
    # Keep this simple and robust: only calibrate if every class is present.
    if len(set(y_val.tolist())) < 3:
        return model
    try:
        return CalibratedClassifierCV(model, cv="prefit", method="sigmoid").fit(x_val, y_val)
    except Exception:
        return model


def write_predictions(path: Path, rows: Sequence[dict], probs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "subject_id", "scan_id", "y_true", "y_pred",
        "prob_CN", "prob_MCI", "prob_AD",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, prob in zip(rows, probs):
            writer.writerow({
                "dataset": row["dataset"],
                "split": row["split"],
                "subject_id": row["subject_id"],
                "scan_id": row["scan_id"],
                "y_true": CLASS_NAMES[int(row["label"])],
                "y_pred": CLASS_NAMES[int(np.argmax(prob))],
                "prob_CN": float(prob[0]),
                "prob_MCI": float(prob[1]),
                "prob_AD": float(prob[2]),
            })


def select_score(metrics_by_split: Dict[str, dict]) -> float:
    val = metrics_by_split.get("val", {})
    aibl = metrics_by_split.get("aibl_adapt_val", {})
    ixi = metrics_by_split.get("ixi_external", {})
    def minority(m):
        pc = m.get("per_class", {})
        return min(pc.get("MCI", {}).get("recall", 0.0), pc.get("AD", {}).get("recall", 0.0))
    return (
        0.35 * (val.get("balanced_acc") or 0.0)
        + 0.20 * (val.get("macro_auc_ovr") or 0.0)
        + 0.20 * (aibl.get("balanced_acc") or 0.0)
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

    manifest_rows = read_manifest(args.manifest)
    if args.rebuild_features or not args.feature_csv.exists():
        build_feature_cache(manifest_rows, args.feature_csv)
    rows = read_feature_cache(args.feature_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    train_rows = by_split["train"] + by_split["aibl_adapt_train"]
    val_rows = by_split["val"] + by_split["aibl_adapt_val"]
    x_train, y_train = matrix(train_rows)
    x_val, y_val = matrix(val_rows)
    eval_splits = ["val", "internal_test", "aibl_adapt_val", "aibl_heldout", "oasis_external", "ixi_external"]

    results = {}
    best_name = None
    best_score = -1e9
    for name, model in make_models(args.seed).items():
        print(f"[fit] {name}", flush=True)
        model.fit(x_train, y_train)
        calibrated = calibrate_if_possible(model, x_val, y_val)
        metrics_by_split = {}
        for split in eval_splits:
            split_rows = by_split.get(split, [])
            if not split_rows:
                continue
            x, y = matrix(split_rows)
            probs = calibrated.predict_proba(x)
            if probs.shape[1] != 3:
                aligned = np.zeros((len(x), 3), dtype=float)
                classes = getattr(calibrated, "classes_", getattr(model, "classes_", []))
                for j, cls in enumerate(classes):
                    aligned[:, int(cls)] = probs[:, j]
                probs = aligned
            metrics = classification_metrics(y, probs)
            metrics_by_split[split] = metrics
            write_predictions(args.out_dir / f"{name}_{split}_predictions.csv", split_rows, probs)
        score = select_score(metrics_by_split)
        results[name] = {"selection_score": score, "metrics": metrics_by_split}
        print(f"[score] {name}: {score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_name = name

    summary = {
        "best_model": best_name,
        "best_score": best_score,
        "feature_csv": str(args.feature_csv),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "results": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.out_dir / 'summary.json'}")
    print(f"[best] {best_name} score={best_score:.4f}")


if __name__ == "__main__":
    main()
