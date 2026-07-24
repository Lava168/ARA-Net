#!/usr/bin/env python3
"""Post-hoc rescue search for AD staging predictions.

This script is intentionally model-agnostic. It consumes prediction CSV files
already produced by the v4 pipelines, then searches log-linear ensembles,
temperature scaling, and class-specific logit offsets. The goal is to quickly
test whether the current weak points are mostly threshold/calibration problems:

* low AD recall on internal ADNI tests,
* low MCI recall on AIBL heldout,
* poor OASIS transfer.

It never retrains a feature extractor. Use it first because it is cheap; then
use the winning profile to decide whether a full server training run is worth
the cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]
SPLITS = [
    "val",
    "internal_test",
    "aibl_adapt_val",
    "aibl_heldout",
    "oasis_external",
    "ixi_external",
]
LABEL_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}


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
    aucs = {}
    valid_aucs = []
    for idx, name in enumerate(CLASS_NAMES):
        y_bin = (y == idx).astype(int)
        auc = binary_auc(y_bin, probs[:, idx])
        aucs[name] = None if math.isnan(auc) else auc
        if not math.isnan(auc):
            valid_aucs.append(auc)
    out = {
        "n": int(len(y)),
        "label_counts": {CLASS_NAMES[i]: int(support[i]) for i in range(3) if support[i] > 0},
        "prediction_distribution": {CLASS_NAMES[i]: int((pred == i).sum()) for i in range(3)},
        "acc": float((pred == y).mean()) if len(y) else 0.0,
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
        score = probs[:, 2] - probs[:, 0]
        yy = np.concatenate([np.zeros(int(cn.sum())), np.ones(int(ad.sum()))])
        ss = np.concatenate([score[cn], score[ad]])
        out["ad_vs_cn_auc"] = binary_auc(yy, ss)
    if len(set(y.tolist())) == 1 and len(y) and int(y[0]) == 0:
        out["cn_retention_rate"] = out["acc"]
        out["false_impairment_rate"] = float(1.0 - out["acc"])
    return out


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def parse_prediction_name(path: Path) -> Tuple[str, str] | None:
    name = path.name
    if not name.endswith("_predictions.csv"):
        return None
    stem = name[: -len("_predictions.csv")]
    for split in sorted(SPLITS, key=len, reverse=True):
        suffix = f"_{split}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], split
    return None


def read_prediction_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["prob_CN"] = float(row["prob_CN"])
        row["prob_MCI"] = float(row["prob_MCI"])
        row["prob_AD"] = float(row["prob_AD"])
    return rows


def row_key(row: dict) -> Tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("split", ""),
        row.get("subject_id", ""),
        row.get("scan_id", ""),
        row.get("y_true", ""),
    )


def discover_predictions(pred_dirs: Sequence[Path], wanted_runs: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    wanted = set(wanted_runs)
    found: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for pred_dir in pred_dirs:
        for path in sorted(pred_dir.glob("*_predictions.csv")):
            parsed = parse_prediction_name(path)
            if parsed is None:
                continue
            run, split = parsed
            if wanted and run not in wanted:
                continue
            found[run][split] = path
    return {run: splits for run, splits in sorted(found.items())}


def load_aligned_run(run: str, split_paths: Dict[str, Path]) -> Dict[str, List[dict]]:
    return {split: read_prediction_csv(path) for split, path in split_paths.items()}


def common_runs(
    discovered: Dict[str, Dict[str, Path]],
    required_splits: Iterable[str],
) -> Dict[str, Dict[str, Path]]:
    required = set(required_splits)
    return {
        run: paths
        for run, paths in discovered.items()
        if required.issubset(paths.keys())
    }


def align_split(
    runs: Sequence[str],
    rows_by_run: Dict[str, Dict[str, List[dict]]],
    split: str,
) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    base_rows = rows_by_run[runs[0]][split]
    keys = [row_key(row) for row in base_rows]
    arrays: Dict[str, np.ndarray] = {}
    for run in runs:
        by_key = {row_key(row): row for row in rows_by_run[run][split]}
        missing = [key for key in keys if key not in by_key]
        if missing:
            raise ValueError(f"{run}/{split} missing {len(missing)} rows relative to {runs[0]}")
        arrays[run] = np.array(
            [
                [
                    by_key[key]["prob_CN"],
                    by_key[key]["prob_MCI"],
                    by_key[key]["prob_AD"],
                ]
                for key in keys
            ],
            dtype=np.float64,
        )
    return base_rows, arrays


def aggregate_subject_rows(rows: Sequence[dict], probs: np.ndarray) -> Tuple[List[dict], np.ndarray]:
    grouped: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row.get("dataset", ""), row.get("split", ""), row.get("subject_id", ""), row.get("y_true", ""))].append(idx)
    out_rows = []
    out_probs = []
    for (dataset, split, subject, y_true), indices in sorted(grouped.items()):
        row = dict(rows[indices[0]])
        row["dataset"] = dataset
        row["split"] = split
        row["subject_id"] = subject
        row["scan_id"] = f"{subject}__subject_mean"
        row["y_true"] = y_true
        out_rows.append(row)
        out_probs.append(probs[indices].mean(axis=0))
    return out_rows, np.vstack(out_probs)


def pooled_probs(
    arrays_by_run: Dict[str, np.ndarray],
    runs: Sequence[str],
    weights: np.ndarray,
    offsets: np.ndarray,
    temperature: float,
) -> np.ndarray:
    logits = np.zeros_like(arrays_by_run[runs[0]], dtype=np.float64)
    for weight, run in zip(weights, runs):
        logits += float(weight) * np.log(np.clip(arrays_by_run[run], 1e-8, 1.0))
    logits = logits / max(float(temperature), 1e-4)
    logits += offsets.reshape(1, 3)
    return softmax(logits)


def y_true_from_rows(rows: Sequence[dict]) -> np.ndarray:
    return np.array([LABEL_TO_INT[row["y_true"]] for row in rows], dtype=int)


def recall(metrics: dict, cls: str) -> float:
    return float(metrics.get("per_class", {}).get(cls, {}).get("recall") or 0.0)


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def score_metrics(profile: str, metrics_by_split: Dict[str, dict]) -> float:
    val = metrics_by_split.get("val", {})
    aibl_val = metrics_by_split.get("aibl_adapt_val", {})
    ixi = metrics_by_split.get("ixi_external", {})
    oasis = metrics_by_split.get("oasis_external", {})

    ixi_retention = metric_value(ixi, "cn_retention_rate") or metric_value(ixi, "acc")
    val_minority = min(recall(val, "MCI"), recall(val, "AD"))
    aibl_minority = min(recall(aibl_val, "MCI"), recall(aibl_val, "AD"))

    if profile == "internal_ad_recall":
        return (
            0.44 * recall(val, "AD")
            + 0.22 * metric_value(val, "balanced_acc")
            + 0.16 * metric_value(val, "macro_auc_ovr")
            + 0.10 * recall(val, "MCI")
            + 0.08 * ixi_retention
        )
    if profile == "aibl_mci_recall":
        return (
            0.42 * recall(aibl_val, "MCI")
            + 0.18 * recall(aibl_val, "AD")
            + 0.18 * metric_value(aibl_val, "balanced_acc")
            + 0.12 * metric_value(aibl_val, "macro_auc_ovr")
            + 0.10 * ixi_retention
        )
    if profile == "oasis_transfer":
        # Only meaningful if OASIS is included in tune-splits. The script
        # reports overlap warnings so this cannot be mistaken for locked eval.
        return (
            0.48 * metric_value(oasis, "balanced_acc")
            + 0.25 * metric_value(oasis, "macro_auc_ovr")
            + 0.15 * min(recall(oasis, "MCI"), recall(oasis, "AD"))
            + 0.07 * ixi_retention
            + 0.05 * metric_value(val, "balanced_acc")
        )
    if profile == "minority_rescue":
        return (
            0.23 * metric_value(val, "balanced_acc")
            + 0.15 * metric_value(val, "macro_auc_ovr")
            + 0.20 * val_minority
            + 0.22 * metric_value(aibl_val, "balanced_acc")
            + 0.15 * aibl_minority
            + 0.05 * ixi_retention
        )
    # balanced
    return (
        0.25 * metric_value(val, "balanced_acc")
        + 0.15 * metric_value(val, "macro_auc_ovr")
        + 0.25 * metric_value(aibl_val, "balanced_acc")
        + 0.12 * aibl_minority
        + 0.13 * ixi_retention
        + 0.10 * metric_value(oasis, "balanced_acc")
    )


def evaluate_transform(
    split_payload: Dict[str, Tuple[List[dict], Dict[str, np.ndarray]]],
    runs: Sequence[str],
    weights: np.ndarray,
    offsets: np.ndarray,
    temperature: float,
    splits: Sequence[str],
    aggregate_subjects: bool,
) -> Dict[str, dict]:
    out = {}
    for split in splits:
        if split not in split_payload:
            continue
        rows, arrays = split_payload[split]
        probs = pooled_probs(arrays, runs, weights, offsets, temperature)
        metric_rows = rows
        metric_probs = probs
        if aggregate_subjects:
            metric_rows, metric_probs = aggregate_subject_rows(rows, probs)
        out[split] = classification_metrics(y_true_from_rows(metric_rows), metric_probs)
    return out


def random_trial(rng: np.random.Generator, n_runs: int, offset_scale: float) -> Tuple[np.ndarray, np.ndarray, float]:
    weights = rng.dirichlet(np.ones(n_runs))
    offsets = rng.uniform(-offset_scale, offset_scale, size=3)
    offsets -= offsets.mean()
    temperature = float(np.exp(rng.uniform(math.log(0.55), math.log(2.2))))
    return weights, offsets, temperature


def named_trials(n_runs: int) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
    trials = []
    equal = np.ones(n_runs, dtype=float) / max(n_runs, 1)
    trials.append(("equal_no_offset", equal, np.zeros(3), 1.0))
    for idx in range(n_runs):
        weights = np.zeros(n_runs, dtype=float)
        weights[idx] = 1.0
        trials.append((f"single_{idx}_no_offset", weights, np.zeros(3), 1.0))
    for label, offsets in {
        "boost_ad": [0.0, -0.15, 0.35],
        "boost_mci": [-0.10, 0.35, -0.10],
        "boost_mci_ad": [-0.30, 0.25, 0.25],
        "conservative_cn": [0.30, -0.10, -0.20],
    }.items():
        arr = np.array(offsets, dtype=float)
        arr -= arr.mean()
        trials.append((label, equal, arr, 1.0))
    return trials


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


def write_profile_predictions(
    out_dir: Path,
    profile: str,
    split_payload: Dict[str, Tuple[List[dict], Dict[str, np.ndarray]]],
    runs: Sequence[str],
    weights: np.ndarray,
    offsets: np.ndarray,
    temperature: float,
    splits: Sequence[str],
) -> None:
    for split in splits:
        if split not in split_payload:
            continue
        rows, arrays = split_payload[split]
        probs = pooled_probs(arrays, runs, weights, offsets, temperature)
        path = out_dir / f"{profile}_{split}_predictions.csv"
        with path.open("w", newline="") as handle:
            fieldnames = [
                "dataset", "split", "subject_id", "scan_id", "y_true", "y_pred",
                "prob_CN", "prob_MCI", "prob_AD",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row, prob in zip(rows, probs):
                writer.writerow({
                    "dataset": row.get("dataset", ""),
                    "split": row.get("split", split),
                    "subject_id": row.get("subject_id", ""),
                    "scan_id": row.get("scan_id", ""),
                    "y_true": row["y_true"],
                    "y_pred": CLASS_NAMES[int(np.argmax(prob))],
                    "prob_CN": float(prob[0]),
                    "prob_MCI": float(prob[1]),
                    "prob_AD": float(prob[2]),
                })


def markdown_summary(results: dict) -> str:
    lines = ["# Rescue Probability Optimizer", ""]
    if results.get("warnings"):
        lines += ["## Warnings", ""]
        for warning in results["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    for profile, item in results["profiles"].items():
        lines += [f"## {profile}", ""]
        lines.append(f"- score: {item['score']:.4f}")
        lines.append(f"- runs: {', '.join(item['runs'])}")
        lines.append(f"- weights: {', '.join(f'{w:.3f}' for w in item['weights'])}")
        lines.append(f"- offsets CN/MCI/AD: {', '.join(f'{v:.3f}' for v in item['offsets'])}")
        lines.append(f"- temperature: {item['temperature']:.3f}")
        lines.append("")
        lines.append("| split | Acc | BAcc | AUC | ADvCN AUC | CN retention | recall CN/MCI/AD | pred |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for split, metrics in item["eval_metrics"].items():
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
    parser.add_argument("--pred-dir", type=Path, action="append", required=True, help="Directory containing *_predictions.csv files. Can be repeated.")
    parser.add_argument("--runs", default="", help="Comma-separated run prefixes. Empty means use all discovered runs.")
    parser.add_argument("--tune-splits", default="val,aibl_adapt_val,ixi_external")
    parser.add_argument("--eval-splits", default="val,internal_test,aibl_adapt_val,aibl_heldout,oasis_external,ixi_external")
    parser.add_argument("--profiles", default="balanced,internal_ad_recall,aibl_mci_recall,minority_rescue")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-runs", type=int, default=8, help="If runs are not specified, keep this many runs ranked by baseline tune score.")
    parser.add_argument("--offset-scale", type=float, default=1.6)
    parser.add_argument("--aggregate-subjects", action="store_true", help="Optimize/evaluate after subject-level probability averaging.")
    args = parser.parse_args()

    tune_splits = parse_csv_list(args.tune_splits)
    eval_splits = parse_csv_list(args.eval_splits)
    profiles = parse_csv_list(args.profiles)
    wanted_runs = parse_csv_list(args.runs)
    required_splits = sorted(set(tune_splits + eval_splits))

    discovered = discover_predictions(args.pred_dir, wanted_runs)
    discovered = common_runs(discovered, required_splits)
    if not discovered:
        raise SystemExit(
            "No complete runs found. Check --pred-dir, --runs, and split names. "
            f"Required splits: {required_splits}"
        )

    rows_by_run = {run: load_aligned_run(run, paths) for run, paths in discovered.items()}
    all_runs = list(rows_by_run)

    split_payload_all_runs = {
        split: align_split(all_runs, rows_by_run, split)
        for split in required_splits
    }

    if not wanted_runs and len(all_runs) > args.top_runs:
        scored = []
        for idx, run in enumerate(all_runs):
            weights = np.zeros(len(all_runs), dtype=float)
            weights[idx] = 1.0
            metrics = evaluate_transform(
                split_payload_all_runs,
                all_runs,
                weights,
                np.zeros(3),
                1.0,
                tune_splits,
                args.aggregate_subjects,
            )
            scored.append((score_metrics("balanced", metrics), run))
        keep = {run for _, run in sorted(scored, reverse=True)[: args.top_runs]}
        all_runs = [run for run in all_runs if run in keep]

    split_payload = {
        split: align_split(all_runs, rows_by_run, split)
        for split in required_splits
    }
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    overlap = sorted(set(tune_splits) & (set(eval_splits) - set(tune_splits)))
    # The expression above is usually empty; report the direct overlap in a
    # clearer way because some users intentionally tune on OASIS stress tests.
    direct_overlap = sorted(set(tune_splits) & set(eval_splits))
    locked_like = {"internal_test", "aibl_heldout", "oasis_external"}
    risky = sorted(set(tune_splits) & locked_like)
    if risky:
        warnings.append(
            "Tune splits include locked/stress-test splits "
            f"{risky}; those metrics are optimization targets, not unbiased heldout estimates."
        )
    if overlap:
        warnings.append(f"Internal overlap note: {overlap}")

    profile_results = {}
    deterministic = named_trials(len(all_runs))
    random_trials = [
        (f"random_{idx}", *random_trial(rng, len(all_runs), args.offset_scale))
        for idx in range(args.n_trials)
    ]
    trials = deterministic + random_trials

    for profile in profiles:
        best = {"score": -1e9, "name": None, "weights": None, "offsets": None, "temperature": None}
        for trial_name, weights, offsets, temperature in trials:
            metrics = evaluate_transform(
                split_payload,
                all_runs,
                weights,
                offsets,
                temperature,
                tune_splits,
                args.aggregate_subjects,
            )
            score = score_metrics(profile, metrics)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "name": trial_name,
                    "weights": weights.copy(),
                    "offsets": offsets.copy(),
                    "temperature": float(temperature),
                    "tune_metrics_raw": metrics,
                }
        eval_metrics_raw = evaluate_transform(
            split_payload,
            all_runs,
            best["weights"],
            best["offsets"],
            best["temperature"],
            eval_splits,
            args.aggregate_subjects,
        )
        write_profile_predictions(
            args.out_dir,
            profile,
            split_payload,
            all_runs,
            best["weights"],
            best["offsets"],
            best["temperature"],
            eval_splits,
        )
        profile_results[profile] = {
            "trial": best["name"],
            "score": best["score"],
            "runs": all_runs,
            "weights": [float(x) for x in best["weights"]],
            "offsets": [float(x) for x in best["offsets"]],
            "temperature": float(best["temperature"]),
            "tune_metrics": {split: metric_digest(m) for split, m in best["tune_metrics_raw"].items()},
            "eval_metrics": {split: metric_digest(m) for split, m in eval_metrics_raw.items()},
        }

    result = {
        "pred_dirs": [str(path) for path in args.pred_dir],
        "runs": all_runs,
        "tune_splits": tune_splits,
        "eval_splits": eval_splits,
        "aggregate_subjects": bool(args.aggregate_subjects),
        "n_trials": int(args.n_trials),
        "warnings": warnings,
        "profiles": profile_results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown_summary(result), encoding="utf-8")
    print(f"[saved] {args.out_dir / 'summary.json'}")
    print(f"[saved] {args.out_dir / 'summary.md'}")
    for profile, item in profile_results.items():
        print(f"[best] {profile}: score={item['score']:.4f} trial={item['trial']}")


if __name__ == "__main__":
    main()
