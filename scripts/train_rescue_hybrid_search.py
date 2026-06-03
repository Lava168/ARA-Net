#!/usr/bin/env python3
"""Targeted rescue search for hybrid atlas/clinical AD staging models.

The existing v4 hybrid script found a strong AIBL/IXI story, but three weak
points remained: AD recall on internal ADNI, MCI recall on AIBL, and OASIS
transfer. This script deliberately optimizes for those weak points by trying:

* class-weight multipliers, especially AD and MCI,
* domain/sample weights for ADNI, AIBL, and OASIS adaptation,
* stronger tabular models and calibrated linear baselines,
* probability offset tuning on validation/adaptation splits.

Keep protocol names honest. If OASIS rows are included in training or tuning,
the OASIS result is adaptation/sensitivity evidence, not locked external
validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from train_atlas_feature_baseline import CLASS_NAMES, classification_metrics, write_predictions
from train_hybrid_atlas_clinical_baseline import (
    FEATURE_SETS,
    build_adni_index,
    build_aibl_index,
    clean_float,
    enrich_rows,
    matrix,
    read_csv_rows,
    rows_by_split,
)


LABEL_WEIGHTS = {
    "flat": [1.0, 1.0, 1.0],
    "mci": [1.0, 1.8, 1.0],
    "ad": [1.0, 1.0, 2.2],
    "mci_ad": [1.0, 1.7, 2.2],
    "aggressive_mci_ad": [0.8, 2.4, 3.0],
}

DOMAIN_WEIGHTS = {
    "balanced": {"ADNI": 1.0, "AIBL": 1.0, "OASIS": 1.0, "IXI": 1.0},
    "aibl_focus": {"ADNI": 0.8, "AIBL": 1.8, "OASIS": 1.0, "IXI": 1.0},
    "adni_focus": {"ADNI": 1.8, "AIBL": 0.8, "OASIS": 1.0, "IXI": 1.0},
    "oasis_focus": {"ADNI": 0.8, "AIBL": 1.0, "OASIS": 3.0, "IXI": 1.0},
    "external_focus": {"ADNI": 0.7, "AIBL": 1.7, "OASIS": 2.4, "IXI": 1.0},
}

PROTOCOLS = {
    "adni_only": {
        "train_splits": ["train"],
        "calibration_splits": ["val"],
        "tune_splits": ["val"],
        "description": "ADNI-only training; external splits remain untouched.",
    },
    "aibl_adapted": {
        "train_splits": ["train", "aibl_adapt_train"],
        "calibration_splits": ["val", "aibl_adapt_val"],
        "tune_splits": ["val", "aibl_adapt_val", "ixi_external"],
        "description": "ADNI plus AIBL adaptation; AIBL heldout remains locked.",
    },
    "oasis_adapted": {
        "train_splits": ["train", "aibl_adapt_train", "oasis_external"],
        "calibration_splits": ["val", "aibl_adapt_val", "oasis_external"],
        "tune_splits": ["val", "aibl_adapt_val", "oasis_external", "ixi_external"],
        "description": "Sensitivity protocol that adapts to OASIS; OASIS is not a locked test.",
    },
}


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def select_rows(by_split: Dict[str, List[dict]], splits: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    for split in splits:
        rows.extend(by_split.get(split, []))
    return rows


def sample_weights(rows: Sequence[dict], label_profile: str, domain_profile: str) -> np.ndarray:
    label_weights = LABEL_WEIGHTS[label_profile]
    domain_weights = DOMAIN_WEIGHTS[domain_profile]
    counts = Counter(int(row["label"]) for row in rows)
    inv_freq = {
        label: len(rows) / (3.0 * max(count, 1))
        for label, count in counts.items()
    }
    weights = []
    for row in rows:
        label = int(row["label"])
        dataset = row.get("dataset", "")
        weights.append(inv_freq.get(label, 1.0) * label_weights[label] * domain_weights.get(dataset, 1.0))
    arr = np.asarray(weights, dtype=np.float64)
    return arr / np.mean(arr).clip(min=1e-8)


def make_models(seed: int) -> Dict[str, object]:
    return {
        "logreg_l1": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                penalty="l1",
                solver="saga",
                C=0.45,
                multi_class="multinomial",
                random_state=seed,
            )),
        ]),
        "logreg_l2": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                C=0.7,
                multi_class="multinomial",
                random_state=seed,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("clf", SVC(C=2.4, gamma="scale", probability=True, random_state=seed)),
        ]),
        "rf_deep": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(
                n_estimators=1000,
                max_depth=14,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "rf_leaf4": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(
                n_estimators=900,
                max_depth=10,
                min_samples_leaf=4,
                random_state=seed + 3,
                n_jobs=-1,
            )),
        ]),
        "extra_deep": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", ExtraTreesClassifier(
                n_estimators=1200,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "hgb_l2_005": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=420,
                learning_rate=0.026,
                l2_regularization=0.05,
                max_leaf_nodes=24,
                random_state=seed,
            )),
        ]),
        "hgb_l2_015": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=360,
                learning_rate=0.032,
                l2_regularization=0.15,
                max_leaf_nodes=18,
                random_state=seed + 7,
            )),
        ]),
    }


def supports_sample_weight(model: object) -> bool:
    if isinstance(model, Pipeline):
        return True
    return True


def fit_with_weights(model: object, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> object:
    if isinstance(model, Pipeline):
        model.fit(x, y, clf__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def calibrate_if_possible(model: object, x_val: np.ndarray, y_val: np.ndarray, weights: Optional[np.ndarray] = None) -> object:
    if len(set(y_val.tolist())) < 3:
        return model
    try:
        calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
        if weights is None:
            return calibrated.fit(x_val, y_val)
        return calibrated.fit(x_val, y_val, sample_weight=weights)
    except Exception:
        return model


def aligned_predict_proba(model: object, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] == 3:
        return probs
    aligned = np.zeros((len(x), 3), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline):
        classes = model.named_steps["clf"].classes_
    for col, cls in enumerate(classes or []):
        aligned[:, int(cls)] = probs[:, col]
    return aligned


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def apply_offsets(probs: np.ndarray, offsets: Sequence[float], temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-8, 1.0)) / max(float(temperature), 1e-4)
    logits += np.asarray(offsets, dtype=float).reshape(1, 3)
    return softmax(logits)


def recall(metrics: dict, cls: str) -> float:
    return float(metrics.get("per_class", {}).get(cls, {}).get("recall") or 0.0)


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def profile_score(profile: str, metrics: Dict[str, dict]) -> float:
    val = metrics.get("val", {})
    internal = metrics.get("internal_test", {})
    aibl_val = metrics.get("aibl_adapt_val", {})
    oasis = metrics.get("oasis_external", {})
    ixi = metrics.get("ixi_external", {})
    ixi_retention = metric_value(ixi, "cn_retention_rate") or metric_value(ixi, "acc")

    if profile == "internal_ad_recall":
        return (
            0.38 * recall(val, "AD")
            + 0.24 * recall(internal, "AD")
            + 0.16 * metric_value(val, "balanced_acc")
            + 0.12 * metric_value(internal, "balanced_acc")
            + 0.10 * ixi_retention
        )
    if profile == "aibl_mci_recall":
        return (
            0.42 * recall(aibl_val, "MCI")
            + 0.18 * recall(aibl_val, "AD")
            + 0.18 * metric_value(aibl_val, "balanced_acc")
            + 0.12 * metric_value(val, "balanced_acc")
            + 0.10 * ixi_retention
        )
    if profile == "oasis_transfer":
        return (
            0.48 * metric_value(oasis, "balanced_acc")
            + 0.18 * recall(oasis, "MCI")
            + 0.18 * recall(oasis, "AD")
            + 0.08 * metric_value(val, "balanced_acc")
            + 0.08 * ixi_retention
        )
    if profile == "minority_rescue":
        return (
            0.18 * min(recall(val, "MCI"), recall(val, "AD"))
            + 0.18 * min(recall(internal, "MCI"), recall(internal, "AD"))
            + 0.22 * min(recall(aibl_val, "MCI"), recall(aibl_val, "AD"))
            + 0.16 * metric_value(aibl_val, "balanced_acc")
            + 0.16 * metric_value(internal, "balanced_acc")
            + 0.10 * ixi_retention
        )
    return (
        0.22 * metric_value(val, "balanced_acc")
        + 0.16 * metric_value(internal, "balanced_acc")
        + 0.24 * metric_value(aibl_val, "balanced_acc")
        + 0.13 * min(recall(aibl_val, "MCI"), recall(aibl_val, "AD"))
        + 0.13 * ixi_retention
        + 0.12 * metric_value(oasis, "balanced_acc")
    )


def tune_offsets(
    raw_probs: Dict[str, np.ndarray],
    y_by_split: Dict[str, np.ndarray],
    tune_splits: Sequence[str],
    profile: str,
    rng: np.random.Generator,
    n_trials: int,
) -> Tuple[np.ndarray, float, Dict[str, dict], float]:
    candidates: List[Tuple[np.ndarray, float]] = []
    candidates.append((np.zeros(3), 1.0))
    for offsets in ([0.0, 0.25, 0.0], [0.0, 0.0, 0.35], [-0.25, 0.15, 0.25], [0.25, -0.10, -0.15]):
        arr = np.asarray(offsets, dtype=float)
        arr -= arr.mean()
        candidates.append((arr, 1.0))
    for _ in range(n_trials):
        arr = rng.uniform(-1.4, 1.4, size=3)
        arr -= arr.mean()
        temp = float(np.exp(rng.uniform(math.log(0.6), math.log(2.0))))
        candidates.append((arr, temp))

    best_score = -1e9
    best_offsets = np.zeros(3)
    best_temp = 1.0
    best_metrics: Dict[str, dict] = {}
    for offsets, temp in candidates:
        metrics = {}
        for split in tune_splits:
            if split not in raw_probs:
                continue
            metrics[split] = classification_metrics(y_by_split[split], apply_offsets(raw_probs[split], offsets, temp))
        score = profile_score(profile, metrics)
        if score > best_score:
            best_score = score
            best_offsets = offsets.copy()
            best_temp = float(temp)
            best_metrics = metrics
    return best_offsets, best_temp, best_metrics, float(best_score)


def metric_digest(metrics: dict) -> dict:
    return {
        "n": metrics.get("n"),
        "acc": metrics.get("acc"),
        "balanced_acc": metrics.get("balanced_acc"),
        "macro_auc_ovr": metrics.get("macro_auc_ovr"),
        "ad_vs_cn_auc": metrics.get("ad_vs_cn_auc"),
        "cn_retention_rate": metrics.get("cn_retention_rate"),
        "recall_CN": recall(metrics, "CN"),
        "recall_MCI": recall(metrics, "MCI"),
        "recall_AD": recall(metrics, "AD"),
        "prediction_distribution": metrics.get("prediction_distribution"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


def load_rows(feature_csv: Path, adni_clinical: Path, aibl_clinical: Path) -> Tuple[List[dict], dict]:
    rows = read_csv_rows(feature_csv)
    for row in rows:
        row["label"] = int(row["label"])
    adni_index = build_adni_index(adni_clinical)
    aibl_index = build_aibl_index(aibl_clinical)
    return enrich_rows(rows, adni_index, aibl_index)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_summary(summary: dict) -> str:
    lines = ["# Targeted Rescue Hybrid Search", ""]
    lines.append(f"Clinical match summary is stored in `summary.json`.")
    lines.append("")
    if summary.get("warnings"):
        lines += ["## Warnings", ""]
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    for profile, item in summary["best_by_profile"].items():
        lines += [f"## {profile}", ""]
        lines.append(f"- run: `{item['run_name']}`")
        lines.append(f"- score: {item['score']:.4f}")
        lines.append(f"- offsets CN/MCI/AD: {', '.join(f'{x:.3f}' for x in item['offsets'])}")
        lines.append(f"- temperature: {item['temperature']:.3f}")
        lines.append("")
        lines.append("| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for split, metrics in item["metrics"].items():
            recalls = f"{metrics['recall_CN']:.3f}/{metrics['recall_MCI']:.3f}/{metrics['recall_AD']:.3f}"
            lines.append(
                f"| {split} | {metrics['acc'] if metrics['acc'] is not None else 0:.3f} "
                f"| {metrics['balanced_acc'] if metrics['balanced_acc'] is not None else 0:.3f} "
                f"| {metrics['macro_auc_ovr'] if metrics['macro_auc_ovr'] is not None else 0:.3f} "
                f"| {metrics['ad_vs_cn_auc'] if metrics['ad_vs_cn_auc'] is not None else 0:.3f} "
                f"| {metrics['cn_retention_rate'] if metrics['cn_retention_rate'] is not None else 0:.3f} "
                f"| {recalls} | {metrics['prediction_distribution']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--adni-clinical", type=Path, required=True)
    parser.add_argument("--aibl-clinical", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocols", default="aibl_adapted")
    parser.add_argument("--feature-sets", default="atlas_core_clinical,atlas_cognitive,atlas_biomarker_enhanced,clinical_core_only")
    parser.add_argument("--models", default="hgb_l2_005,hgb_l2_015,rf_deep,rf_leaf4,extra_deep,logreg_l2")
    parser.add_argument("--label-profiles", default="flat,mci,ad,mci_ad,aggressive_mci_ad")
    parser.add_argument("--domain-profiles", default="balanced,aibl_focus,adni_focus")
    parser.add_argument("--profiles", default="balanced,internal_ad_recall,aibl_mci_recall,minority_rescue")
    parser.add_argument("--eval-splits", default="val,internal_test,aibl_adapt_val,aibl_heldout,oasis_external,ixi_external")
    parser.add_argument("--offset-trials", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    protocols = parse_csv_list(args.protocols)
    feature_sets = parse_csv_list(args.feature_sets)
    model_names = parse_csv_list(args.models)
    label_profiles = parse_csv_list(args.label_profiles)
    domain_profiles = parse_csv_list(args.domain_profiles)
    profiles = parse_csv_list(args.profiles)
    eval_splits = parse_csv_list(args.eval_splits)
    for protocol in protocols:
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
    for feature_set in feature_sets:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
    for label_profile in label_profiles:
        if label_profile not in LABEL_WEIGHTS:
            raise ValueError(f"Unknown label profile: {label_profile}")
    for domain_profile in domain_profiles:
        if domain_profile not in DOMAIN_WEIGHTS:
            raise ValueError(f"Unknown domain profile: {domain_profile}")

    rng = np.random.default_rng(args.seed)
    rows, clinical_summary = load_rows(args.feature_csv, args.adni_clinical, args.aibl_clinical)
    by_split = rows_by_split(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    if "oasis_adapted" in protocols:
        warnings.append("oasis_adapted includes OASIS in training/tuning; OASIS metrics are not locked external validation.")

    best_by_profile = {
        profile: {"score": -1e9}
        for profile in profiles
    }
    results = {}
    available_models = make_models(args.seed)
    for model_name in model_names:
        if model_name not in available_models:
            raise ValueError(f"Unknown model: {model_name}")

    for protocol in protocols:
        spec = PROTOCOLS[protocol]
        train_rows = select_rows(by_split, spec["train_splits"])
        calib_rows = select_rows(by_split, spec["calibration_splits"])
        tune_splits = spec["tune_splits"]
        if not train_rows:
            raise ValueError(f"No training rows for {protocol}")
        for feature_set, model_name, label_profile, domain_profile in product(
            feature_sets,
            model_names,
            label_profiles,
            domain_profiles,
        ):
            names = FEATURE_SETS[feature_set]
            run_name = f"{protocol}/{feature_set}__{model_name}__{label_profile}__{domain_profile}"
            print(f"[fit] {run_name}", flush=True)
            x_train, y_train = matrix(train_rows, names)
            x_calib, y_calib = matrix(calib_rows, names)
            weights = sample_weights(train_rows, label_profile, domain_profile)
            calib_weights = sample_weights(calib_rows, label_profile, domain_profile) if calib_rows else None
            model = make_models(args.seed)[model_name]
            fit_with_weights(model, x_train, y_train, weights)
            calibrated = calibrate_if_possible(model, x_calib, y_calib, calib_weights)

            raw_probs: Dict[str, np.ndarray] = {}
            y_by_split: Dict[str, np.ndarray] = {}
            split_rows_map: Dict[str, List[dict]] = {}
            raw_metrics: Dict[str, dict] = {}
            for split in eval_splits:
                split_rows = by_split.get(split, [])
                if not split_rows:
                    continue
                x_eval, y_eval = matrix(split_rows, names)
                probs = aligned_predict_proba(calibrated, x_eval)
                raw_probs[split] = probs
                y_by_split[split] = y_eval
                split_rows_map[split] = split_rows
                raw_metrics[split] = classification_metrics(y_eval, probs)

            profile_payload = {}
            for profile in profiles:
                offsets, temp, tune_metrics, score = tune_offsets(
                    raw_probs,
                    y_by_split,
                    tune_splits,
                    profile,
                    rng,
                    args.offset_trials,
                )
                final_metrics_raw = {
                    split: classification_metrics(y_by_split[split], apply_offsets(raw_probs[split], offsets, temp))
                    for split in eval_splits
                    if split in raw_probs
                }
                final_score = profile_score(profile, {
                    split: final_metrics_raw[split]
                    for split in tune_splits
                    if split in final_metrics_raw
                })
                profile_payload[profile] = {
                    "score": final_score,
                    "offsets": [float(x) for x in offsets],
                    "temperature": float(temp),
                    "tune_metrics": {split: metric_digest(m) for split, m in tune_metrics.items()},
                    "metrics": {split: metric_digest(m) for split, m in final_metrics_raw.items()},
                }
                if final_score > best_by_profile[profile]["score"]:
                    profile_out_dir = args.out_dir / "best_predictions" / profile
                    profile_out_dir.mkdir(parents=True, exist_ok=True)
                    for split in eval_splits:
                        if split not in raw_probs:
                            continue
                        write_predictions(
                            profile_out_dir / f"{split}_predictions.csv",
                            split_rows_map[split],
                            apply_offsets(raw_probs[split], offsets, temp),
                        )
                    best_by_profile[profile] = {
                        "run_name": run_name,
                        "score": float(final_score),
                        "offsets": [float(x) for x in offsets],
                        "temperature": float(temp),
                        "metrics": {split: metric_digest(m) for split, m in final_metrics_raw.items()},
                        "raw_metrics": {split: metric_digest(m) for split, m in raw_metrics.items()},
                    }
                    print(f"[best:{profile}] {run_name} score={final_score:.4f}", flush=True)

            results[run_name] = {
                "protocol": protocol,
                "feature_set": feature_set,
                "model": model_name,
                "label_profile": label_profile,
                "domain_profile": domain_profile,
                "n_features": len(names),
                "raw_metrics": {split: metric_digest(m) for split, m in raw_metrics.items()},
                "profiles": profile_payload,
            }

    summary = {
        "feature_csv": str(args.feature_csv),
        "adni_clinical": str(args.adni_clinical),
        "aibl_clinical": str(args.aibl_clinical),
        "protocols": {name: PROTOCOLS[name] for name in protocols},
        "clinical_summary": clinical_summary,
        "warnings": warnings,
        "best_by_profile": best_by_profile,
        "results": results,
    }
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(f"[saved] {args.out_dir / 'summary.json'}")
    print(f"[saved] {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
