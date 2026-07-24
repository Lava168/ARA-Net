#!/usr/bin/env python3
"""Hybrid atlas + clinical baselines for external AD classification.

The goal is not to replace the MRI-only atlas model, but to test whether a
clinically enhanced version can materially improve the weak points exposed by
external validation, especially AIBL MCI/AD staging and IXI healthy specificity.
The script keeps two protocols separate:

* adni_only: train/select on ADNI only, then evaluate external cohorts.
* aibl_adapted: train on ADNI plus AIBL adaptation subjects, then evaluate the
  locked AIBL heldout split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from train_atlas_feature_baseline import (
    CLASS_NAMES,
    FEATURE_NAMES,
    classification_metrics,
    read_feature_cache,
    write_predictions,
)


CLINICAL_FEATURES = [
    "clin_age",
    "clin_sex_male",
    "clin_education",
    "clin_apoe4",
    "clin_mmse",
    "clin_cdrsb",
    "clin_adas13",
    "clin_faq",
    "clin_csf_abeta42",
    "clin_csf_tau",
    "clin_csf_ptau",
    "clin_wmh_total",
    "clin_amy_suvr",
    "clin_amy_centiloids",
    "clin_amy_status",
    "clin_hippocampus",
    "clin_entorhinal",
    "clin_wholebrain",
    "clin_icv",
    "clin_hippocampus_icv_ratio",
    "clin_entorhinal_icv_ratio",
    "clin_wholebrain_icv_ratio",
]

DEMOGRAPHIC_FEATURES = [
    "clin_age",
    "clin_sex_male",
    "clin_education",
    "clin_apoe4",
]
CORE_COGNITIVE_FEATURES = [
    "clin_mmse",
    "clin_cdrsb",
]
EXTENDED_COGNITIVE_FEATURES = [
    "clin_adas13",
    "clin_faq",
]
BIOMARKER_FEATURES = [
    "clin_csf_abeta42",
    "clin_csf_tau",
    "clin_csf_ptau",
    "clin_wmh_total",
    "clin_amy_suvr",
    "clin_amy_centiloids",
    "clin_amy_status",
    "clin_hippocampus",
    "clin_entorhinal",
    "clin_wholebrain",
    "clin_icv",
    "clin_hippocampus_icv_ratio",
    "clin_entorhinal_icv_ratio",
    "clin_wholebrain_icv_ratio",
]

FEATURE_SETS = {
    "atlas_only": FEATURE_NAMES,
    "atlas_demographic": FEATURE_NAMES + DEMOGRAPHIC_FEATURES,
    "atlas_core_clinical": FEATURE_NAMES + DEMOGRAPHIC_FEATURES + CORE_COGNITIVE_FEATURES,
    "atlas_cognitive": FEATURE_NAMES + DEMOGRAPHIC_FEATURES + CORE_COGNITIVE_FEATURES + EXTENDED_COGNITIVE_FEATURES,
    "atlas_biomarker_enhanced": FEATURE_NAMES + DEMOGRAPHIC_FEATURES + CORE_COGNITIVE_FEATURES + EXTENDED_COGNITIVE_FEATURES + BIOMARKER_FEATURES,
    "clinical_core_only": DEMOGRAPHIC_FEATURES + CORE_COGNITIVE_FEATURES,
    "clinical_biomarker_only": DEMOGRAPHIC_FEATURES + CORE_COGNITIVE_FEATURES + EXTENDED_COGNITIVE_FEATURES + BIOMARKER_FEATURES,
}

PROTOCOLS = {
    "adni_only": {
        "train_splits": ["train"],
        "calibration_splits": ["val"],
        "selection_mode": "internal_only",
    },
    "aibl_adapted": {
        "train_splits": ["train", "aibl_adapt_train"],
        "calibration_splits": ["val", "aibl_adapt_val"],
        "selection_mode": "adaptation_val_plus_specificity",
    },
}

ADNI_IMAGE_RE = re.compile(r"_I([0-9]+)$")
AIBL_SCAN_RE = re.compile(r"^AIBL_([0-9]+)_([^_]+)_I[0-9]+$")


def clean_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def sex_to_male(value: object) -> float:
    text = str(value).strip().lower()
    if not text:
        return math.nan
    if text.startswith("m"):
        return 1.0
    if text.startswith("f"):
        return 0.0
    numeric = clean_float(text)
    if math.isnan(numeric):
        return math.nan
    if int(numeric) == 1:
        return 1.0
    if int(numeric) == 2:
        return 0.0
    return numeric


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def image_id_from_scan(scan_id: str) -> Optional[str]:
    match = ADNI_IMAGE_RE.search(scan_id)
    return match.group(1) if match else None


def safe_ratio(num: float, den: float) -> float:
    if math.isnan(num) or math.isnan(den) or abs(den) < 1e-8:
        return math.nan
    return num / den


def empty_clinical() -> Dict[str, float]:
    return {name: math.nan for name in CLINICAL_FEATURES}


def adni_record(row: dict) -> Dict[str, float]:
    record = empty_clinical()
    record.update({
        "clin_age": clean_float(row.get("AGE")),
        "clin_sex_male": sex_to_male(row.get("GENDER")),
        "clin_education": clean_float(row.get("EDUCATION")),
        "clin_apoe4": clean_float(row.get("APOE_E4")),
        "clin_csf_abeta42": clean_float(row.get("CSF_ABETA42")),
        "clin_csf_tau": clean_float(row.get("CSF_TAU")),
        "clin_csf_ptau": clean_float(row.get("CSF_PTAU")),
        "clin_wmh_total": clean_float(row.get("WMH_TOTAL")),
        "clin_amy_suvr": clean_float(row.get("AMY_SUVR")),
        "clin_amy_centiloids": clean_float(row.get("AMY_CENTILOIDS")),
        "clin_amy_status": clean_float(row.get("AMY_STATUS")),
    })
    return record


def aibl_record(row: dict) -> Dict[str, float]:
    record = empty_clinical()
    hipp = clean_float(row.get("Hippocampus"))
    ent = clean_float(row.get("Entorhinal"))
    whole = clean_float(row.get("WholeBrain"))
    icv = clean_float(row.get("ICV"))
    record.update({
        "clin_age": clean_float(row.get("AGE")),
        "clin_sex_male": sex_to_male(row.get("PTGENDER")),
        "clin_education": clean_float(row.get("PTEDUCAT")),
        "clin_apoe4": clean_float(row.get("APOE4")),
        "clin_mmse": clean_float(row.get("MMSE")),
        "clin_cdrsb": clean_float(row.get("CDRSB")),
        "clin_adas13": clean_float(row.get("ADAS13")),
        "clin_faq": clean_float(row.get("FAQ")),
        "clin_hippocampus": hipp,
        "clin_entorhinal": ent,
        "clin_wholebrain": whole,
        "clin_icv": icv,
        "clin_hippocampus_icv_ratio": safe_ratio(hipp, icv),
        "clin_entorhinal_icv_ratio": safe_ratio(ent, icv),
        "clin_wholebrain_icv_ratio": safe_ratio(whole, icv),
    })
    return record


def median_record(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return empty_clinical()
    out = {}
    for name in CLINICAL_FEATURES:
        values = [r[name] for r in records if not math.isnan(r[name])]
        out[name] = float(median(values)) if values else math.nan
    return out


def build_adni_index(path: Path) -> dict:
    rows = read_csv_rows(path)
    by_image: Dict[Tuple[str, str], Dict[str, float]] = {}
    by_subject_values: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        subject = str(row.get("PTID", "")).strip()
        if not subject:
            continue
        record = adni_record(row)
        image_id = clean_float(row.get("image_id"))
        if not math.isnan(image_id):
            by_image[(subject, str(int(image_id)))] = record
        by_subject_values[subject].append(record)
    return {
        "by_image": by_image,
        "by_subject": {subject: median_record(values) for subject, values in by_subject_values.items()},
    }


def build_aibl_index(path: Path) -> dict:
    rows = read_csv_rows(path)
    by_visit: Dict[Tuple[str, str], Dict[str, float]] = {}
    by_subject_values: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("RID") or row.get("PTID") or "").strip()
        if not rid:
            continue
        subject = f"AIBL_{int(float(rid))}" if rid.replace(".", "", 1).isdigit() else f"AIBL_{rid}"
        visit = str(row.get("VISCODE", "")).strip()
        record = aibl_record(row)
        if visit:
            by_visit[(subject, visit)] = record
        by_subject_values[subject].append(record)
    return {
        "by_visit": by_visit,
        "by_subject": {subject: median_record(values) for subject, values in by_subject_values.items()},
    }


def match_clinical(row: dict, adni_index: dict, aibl_index: dict) -> Tuple[Dict[str, float], str]:
    dataset = row["dataset"]
    if dataset == "ADNI":
        subject = row["subject_id"]
        image_id = image_id_from_scan(row["scan_id"])
        if image_id and (subject, image_id) in adni_index["by_image"]:
            return adni_index["by_image"][(subject, image_id)], "adni_image"
        if subject in adni_index["by_subject"]:
            return adni_index["by_subject"][subject], "adni_subject"
    if dataset == "AIBL":
        match = AIBL_SCAN_RE.match(row["scan_id"])
        if match:
            subject = f"AIBL_{match.group(1)}"
            visit = match.group(2)
            if (subject, visit) in aibl_index["by_visit"]:
                return aibl_index["by_visit"][(subject, visit)], "aibl_visit"
            if subject in aibl_index["by_subject"]:
                return aibl_index["by_subject"][subject], "aibl_subject"
        if row["subject_id"] in aibl_index["by_subject"]:
            return aibl_index["by_subject"][row["subject_id"]], "aibl_subject"
    return empty_clinical(), "missing"


def enrich_rows(rows: Sequence[dict], adni_index: dict, aibl_index: dict) -> Tuple[List[dict], dict]:
    enriched = []
    match_counts = Counter()
    coverage = defaultdict(Counter)
    for row in rows:
        record, match_level = match_clinical(row, adni_index, aibl_index)
        copied = dict(row)
        copied.update(record)
        copied["clinical_match_level"] = match_level
        enriched.append(copied)
        match_counts[(row["dataset"], row["split"], match_level)] += 1
        for name in CLINICAL_FEATURES:
            if not math.isnan(record[name]):
                coverage[(row["dataset"], row["split"])][name] += 1
    coverage_out = {
        f"{dataset}/{split}": {name: int(counts.get(name, 0)) for name in CLINICAL_FEATURES}
        for (dataset, split), counts in sorted(coverage.items())
    }
    match_out = {
        f"{dataset}/{split}/{match}": int(count)
        for (dataset, split, match), count in sorted(match_counts.items())
    }
    return enriched, {"match_counts": match_out, "non_missing_counts": coverage_out}


def rows_by_split(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[row["split"]].append(row)
    return out


def select_rows(by_split: Dict[str, List[dict]], splits: Iterable[str]) -> List[dict]:
    selected: List[dict] = []
    for split in splits:
        selected.extend(by_split.get(split, []))
    return selected


def matrix(rows: Sequence[dict], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[clean_float(row.get(name)) for name in names] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return x, y


def make_models(seed: int) -> Dict[str, object]:
    return {
        "logreg_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=0.7, random_state=seed)),
        ]),
        "svm_rbf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
            ("clf", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]),
        "rf_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", RandomForestClassifier(n_estimators=700, max_depth=9, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
        "extratrees_balanced": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", ExtraTreesClassifier(n_estimators=900, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
        "hgb": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", HistGradientBoostingClassifier(max_iter=320, learning_rate=0.03, l2_regularization=0.08, random_state=seed)),
        ]),
    }


def calibrate_if_possible(model, x_val: np.ndarray, y_val: np.ndarray):
    if len(set(y_val.tolist())) < 3:
        return model
    try:
        return CalibratedClassifierCV(model, cv="prefit", method="sigmoid").fit(x_val, y_val)
    except Exception:
        return model


def aligned_predict_proba(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] == 3:
        return probs
    aligned = np.zeros((len(x), 3), dtype=float)
    for col, cls in enumerate(getattr(model, "classes_", [])):
        aligned[:, int(cls)] = probs[:, col]
    return aligned


def minority_recall(metrics: dict) -> float:
    per_class = metrics.get("per_class", {})
    return min(
        per_class.get("MCI", {}).get("recall", 0.0),
        per_class.get("AD", {}).get("recall", 0.0),
    )


def selection_score(protocol: str, metrics: Dict[str, dict]) -> float:
    val = metrics.get("val", {})
    aibl = metrics.get("aibl_adapt_val", {})
    ixi = metrics.get("ixi_external", {})
    if protocol == "adni_only":
        return (
            0.55 * (val.get("balanced_acc") or 0.0)
            + 0.30 * (val.get("macro_auc_ovr") or 0.0)
            + 0.15 * minority_recall(val)
        )
    return (
        0.25 * (val.get("balanced_acc") or 0.0)
        + 0.15 * (val.get("macro_auc_ovr") or 0.0)
        + 0.30 * (aibl.get("balanced_acc") or 0.0)
        + 0.15 * minority_recall(aibl)
        + 0.15 * (ixi.get("cn_retention_rate") or 0.0)
    )


def grouped_eval_sets(by_split: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    groups = {
        "val": by_split.get("val", []),
        "internal_test": by_split.get("internal_test", []),
        "aibl_adapt_val": by_split.get("aibl_adapt_val", []),
        "aibl_heldout": by_split.get("aibl_heldout", []),
        "oasis_external": by_split.get("oasis_external", []),
        "ixi_external": by_split.get("ixi_external", []),
    }
    groups["aibl_all"] = (
        by_split.get("aibl_adapt_train", [])
        + by_split.get("aibl_adapt_val", [])
        + by_split.get("aibl_heldout", [])
    )
    return {name: rows for name, rows in groups.items() if rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--adni-clinical", type=Path, required=True)
    parser.add_argument("--aibl-clinical", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocols", default=",".join(PROTOCOLS), help="Comma-separated protocol names to run.")
    parser.add_argument("--feature-sets", default=",".join(FEATURE_SETS), help="Comma-separated feature-set names to run.")
    parser.add_argument("--models", default="", help="Comma-separated model names to run. Empty means all models.")
    args = parser.parse_args()

    selected_protocols = [name.strip() for name in args.protocols.split(",") if name.strip()]
    selected_feature_sets = [name.strip() for name in args.feature_sets.split(",") if name.strip()]
    selected_models = [name.strip() for name in args.models.split(",") if name.strip()]
    for name in selected_protocols:
        if name not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {name}")
    for name in selected_feature_sets:
        if name not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {name}")
    model_names = set(make_models(args.seed))
    for name in selected_models:
        if name not in model_names:
            raise ValueError(f"Unknown model: {name}")

    rows = read_feature_cache(args.feature_csv)
    adni_index = build_adni_index(args.adni_clinical)
    aibl_index = build_aibl_index(args.aibl_clinical)
    rows, clinical_summary = enrich_rows(rows, adni_index, aibl_index)
    by_split = rows_by_split(rows)
    eval_sets = grouped_eval_sets(by_split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    best_by_protocol = {}
    for protocol in selected_protocols:
        spec = PROTOCOLS[protocol]
        train_rows = select_rows(by_split, spec["train_splits"])
        calib_rows = select_rows(by_split, spec["calibration_splits"])
        protocol_results = {}
        best = {"name": None, "score": -1e9}
        for feature_set in selected_feature_sets:
            names = FEATURE_SETS[feature_set]
            x_train, y_train = matrix(train_rows, names)
            x_calib, y_calib = matrix(calib_rows, names)
            for model_name, model in make_models(args.seed).items():
                if selected_models and model_name not in selected_models:
                    continue
                run_name = f"{feature_set}__{model_name}"
                print(f"[fit] {protocol}/{run_name}", flush=True)
                model.fit(x_train, y_train)
                calibrated = calibrate_if_possible(model, x_calib, y_calib)
                split_metrics = {}
                for split_name, split_rows in eval_sets.items():
                    x_eval, y_eval = matrix(split_rows, names)
                    probs = aligned_predict_proba(calibrated, x_eval)
                    split_metrics[split_name] = classification_metrics(y_eval, probs)
                    write_predictions(
                        args.out_dir / f"{protocol}_{run_name}_{split_name}_predictions.csv",
                        split_rows,
                        probs,
                    )
                score = selection_score(protocol, split_metrics)
                protocol_results[run_name] = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "selection_score": score,
                    "n_features": len(names),
                    "metrics": split_metrics,
                }
                print(f"[score] {protocol}/{run_name}: {score:.4f}", flush=True)
                if score > best["score"]:
                    best = {"name": run_name, "score": score}
        results[protocol] = protocol_results
        best_by_protocol[protocol] = best

    summary = {
        "feature_csv": str(args.feature_csv),
        "adni_clinical": str(args.adni_clinical),
        "aibl_clinical": str(args.aibl_clinical),
        "clinical_features": CLINICAL_FEATURES,
        "feature_sets": {name: list(features) for name, features in FEATURE_SETS.items()},
        "protocols": PROTOCOLS,
        "clinical_summary": clinical_summary,
        "best_by_protocol": best_by_protocol,
        "results": results,
    }
    output = args.out_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {output}")
    print(json.dumps(best_by_protocol, indent=2))


if __name__ == "__main__":
    main()
