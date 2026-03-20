#!/usr/bin/env python3
"""
Aggregate v3 experiment results: per-model stats, ensemble, bootstrap CI,
statistical tests, and publication-ready tables.

Usage:
    python -m chapter1_foundation.aggregate_results \
        chapter1_foundation/experiment_results_v3 \
        --output chapter1_foundation/aggregated_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

CLASS_NAMES = ["CN", "MCI", "AD"]

# ── helpers ──────────────────────────────────────────────────────────────────

def _balanced_acc(y_true, y_pred, n_classes=3):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == c).mean())
    return float(np.mean(recalls))


def _macro_auc_ovr(y_true, y_prob, n_classes=3):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aucs = []
    for c in range(n_classes):
        binary = (y_true == c).astype(int)
        scores = y_prob[:, c]
        if binary.sum() == 0 or binary.sum() == len(binary):
            aucs.append(0.5)
            continue
        order = np.argsort(-scores)
        ys = binary[order]
        n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
        tp = fp = auc = 0.0
        tpr_prev = fpr_prev = 0.0
        prev = -np.inf
        for i in range(len(ys)):
            if scores[order[i]] != prev:
                tpr, fpr = tp / n_pos, fp / n_neg
                auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
                tpr_prev, fpr_prev = tpr, fpr
                prev = scores[order[i]]
            if ys[i] == 1:
                tp += 1
            else:
                fp += 1
        tpr, fpr = tp / n_pos, fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        aucs.append(auc)
    return float(np.mean(aucs))


def _weighted_f1(y_true, y_pred, n_classes=3):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s, supports = [], []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        supports.append((y_true == c).sum())
    total = sum(supports)
    return float(sum(f * s for f, s in zip(f1s, supports)) / max(total, 1))


def bootstrap_ci(values, n_bootstrap=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    boot = np.array([rng.choice(values, n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


# ── ensemble prediction ─────────────────────────────────────────────────────

def ensemble_predictions(runs: List[dict], method="soft_vote") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble multiple seed runs for the same fold.
    Returns (y_true, y_pred_ensemble, y_prob_ensemble)."""
    y_true = np.array(runs[0]["test_y_true"])
    all_probs = np.array([np.array(r["test_y_prob"]) for r in runs])
    avg_prob = all_probs.mean(axis=0)
    y_pred = avg_prob.argmax(axis=1)
    return y_true, y_pred, avg_prob


# ── load results ─────────────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> dict:
    all_results = {}
    for seed_dir in sorted(results_dir.glob("seed_*")):
        for fname in ["all_results.json", "all_results_partial.json"]:
            fpath = seed_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                all_results.update(data)
                break
    return all_results


def group_by_model(all_results: dict) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for key, val in all_results.items():
        config_name = key.rsplit("__", 2)[0]
        groups[config_name].append(val)
    return dict(groups)


def group_by_model_seed_fold(all_results: dict) -> Dict[str, Dict[int, Dict[int, dict]]]:
    """Group by (config_name, seed, fold).
    Key format: 'ModelName__seedXXX__foldY'
    """
    import re
    groups = defaultdict(lambda: defaultdict(dict))
    for key, val in all_results.items():
        parts = key.split("__")
        config_name = parts[0]
        seed, fold = None, None
        for p in parts:
            ms = re.match(r"seed(\d+)", p)
            mf = re.match(r"fold(\d+)", p)
            if ms:
                seed = int(ms.group(1))
            if mf:
                fold = int(mf.group(1))
        if seed is not None and fold is not None:
            groups[config_name][seed][fold] = val
    return {k: {s: dict(folds) for s, folds in seeds.items()} for k, seeds in groups.items()}


# ── aggregate ────────────────────────────────────────────────────────────────

def compute_model_stats(runs: List[dict]) -> dict:
    baccs = [r["test_balanced_acc"] for r in runs]
    accs = [r["test_acc"] for r in runs]

    aucs = []
    wf1s = []
    per_class_recalls = {c: [] for c in CLASS_NAMES}
    for r in runs:
        y_true = np.array(r["test_y_true"])
        y_pred = np.array(r["test_y_pred"])
        y_prob = np.array(r["test_y_prob"])
        aucs.append(_macro_auc_ovr(y_true, y_prob))
        wf1s.append(_weighted_f1(y_true, y_pred))
        for i, c in enumerate(CLASS_NAMES):
            mask = y_true == i
            if mask.sum() > 0:
                per_class_recalls[c].append(float((y_pred[mask] == i).mean()))

    stats = {}
    for name, values in [("BAcc", baccs), ("Acc", accs), ("AUC", aucs), ("wF1", wf1s)]:
        mean, lo, hi = bootstrap_ci(values)
        stats[name] = {
            "mean": mean,
            "std": float(np.std(values)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_runs": len(values),
            "values": [float(v) for v in values],
        }
    for c in CLASS_NAMES:
        vals = per_class_recalls[c]
        if vals:
            mean, lo, hi = bootstrap_ci(vals)
            stats[f"Recall_{c}"] = {"mean": mean, "std": float(np.std(vals)), "ci_lo": lo, "ci_hi": hi}

    return stats


def compute_per_seed_stats(model_seed_folds: Dict[int, Dict[int, dict]]) -> dict:
    """Per-seed mean across folds, then aggregate across seeds.
    Returns stats where each 'observation' is one seed's mean over its folds."""
    seed_baccs, seed_accs, seed_aucs, seed_wf1s = [], [], [], []
    for seed_id, folds in sorted(model_seed_folds.items()):
        if len(folds) < 3:
            continue
        baccs, accs, aucs, wf1s = [], [], [], []
        for fold_id, r in sorted(folds.items()):
            baccs.append(r["test_balanced_acc"])
            accs.append(r["test_acc"])
            y_true = np.array(r["test_y_true"])
            y_prob = np.array(r["test_y_prob"])
            y_pred = np.array(r["test_y_pred"])
            aucs.append(_macro_auc_ovr(y_true, y_prob))
            wf1s.append(_weighted_f1(y_true, y_pred))
        seed_baccs.append(float(np.mean(baccs)))
        seed_accs.append(float(np.mean(accs)))
        seed_aucs.append(float(np.mean(aucs)))
        seed_wf1s.append(float(np.mean(wf1s)))

    if not seed_baccs:
        return {}

    stats = {}
    for name, values in [("BAcc", seed_baccs), ("Acc", seed_accs),
                         ("AUC", seed_aucs), ("wF1", seed_wf1s)]:
        mean, lo, hi = bootstrap_ci(values)
        stats[name] = {"mean": mean, "std": float(np.std(values)),
                       "ci_lo": lo, "ci_hi": hi, "n_seeds": len(values),
                       "values": [float(v) for v in values]}
    return stats


# ── cross-variant ensemble ────────────────────────────────────────────────────

ENSEMBLE_VARIANTS = [
    "Ours (Atlas+AnatDist)",
    "Ours (Atlas only)",
    "Ours (no atlas)",
]

def compute_variant_ensemble(
    all_results: dict,
    variants: List[str] = ENSEMBLE_VARIANTS,
    ensemble_name: str = "ARA-Net Ensemble",
) -> List[dict]:
    """For each (seed, fold), soft-vote across model variants that share
    the same train/val/test split, producing synthetic 'ensemble' runs."""
    import re
    sf_map: Dict[Tuple[int, int], Dict[str, dict]] = defaultdict(dict)
    for key, val in all_results.items():
        parts = key.split("__")
        model = parts[0]
        if model not in variants:
            continue
        seed = fold = None
        for p in parts:
            ms = re.match(r"seed(\d+)", p)
            mf = re.match(r"fold(\d+)", p)
            if ms:
                seed = int(ms.group(1))
            if mf:
                fold = int(mf.group(1))
        if seed is not None and fold is not None:
            sf_map[(seed, fold)][model] = val

    ensemble_runs = []
    for (seed, fold), models in sorted(sf_map.items()):
        runs = [models[v] for v in variants if v in models]
        if len(runs) < len(variants):
            continue
        y_true, y_pred, y_prob = ensemble_predictions(runs)
        bacc = _balanced_acc(y_true, y_pred)
        acc = float((y_pred == y_true).mean())
        auc = _macro_auc_ovr(y_true, y_prob)
        wf1 = _weighted_f1(y_true, y_pred)
        per_class = {}
        for i, c in enumerate(CLASS_NAMES):
            mask = y_true == i
            if mask.sum() > 0:
                per_class[c] = float((y_pred[mask] == i).mean())
        ensemble_runs.append({
            "test_balanced_acc": bacc,
            "test_acc": acc,
            "test_y_true": y_true.tolist(),
            "test_y_pred": y_pred.tolist(),
            "test_y_prob": y_prob.tolist(),
            "test_per_class_acc": per_class,
            "seed": seed,
            "fold": fold,
            "config_name": ensemble_name,
        })
    return ensemble_runs


# ── statistical tests ────────────────────────────────────────────────────────

def pairwise_tests(model_groups: Dict[str, List[dict]],
                   reference="Ours (Atlas+AnatDist)") -> List[dict]:
    if reference not in model_groups:
        return []

    ref_baccs = sorted(model_groups[reference], key=lambda r: (r.get("seed", 0), r.get("fold", 0)))
    ref_vals = np.array([r["test_balanced_acc"] for r in ref_baccs])
    results = []

    for model_name, runs in model_groups.items():
        if model_name == reference:
            continue
        comp_sorted = sorted(runs, key=lambda r: (r.get("seed", 0), r.get("fold", 0)))
        n = min(len(ref_vals), len(comp_sorted))
        if n < 5:
            continue
        comp_vals = np.array([r["test_balanced_acc"] for r in comp_sorted[:n]])

        try:
            stat, p = wilcoxon(ref_vals[:n], comp_vals)
        except Exception:
            stat, p = 0.0, 1.0

        results.append({
            "reference": reference,
            "comparison": model_name,
            "ref_mean": float(ref_vals[:n].mean()),
            "comp_mean": float(comp_vals.mean()),
            "delta": float(ref_vals[:n].mean() - comp_vals.mean()),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "significant_005": p < 0.05,
            "significant_001": p < 0.01,
            "n_pairs": n,
        })

    return results


# ── display ──────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "ARA-Net Ensemble",
    "Ours (Atlas+AnatDist)",
    "Ours (Atlas only)",
    "Ours (no atlas)",
    "3D ResNet-18",
    "3D ViT",
    "Plain CNN",
]

MODEL_DISPLAY = {
    "ARA-Net Ensemble": "ARA-Net Ensemble",
    "Ours (Atlas+AnatDist)": "ARA-Net (Full)",
    "Ours (Atlas only)": "ARA-Net (−AD loss)",
    "Ours (no atlas)": "ARA-Net (−Atlas)",
    "3D ResNet-18": "ResNet-18 3D",
    "3D ViT": "ViT 3D",
    "Plain CNN": "Plain CNN",
}


def print_table(model_stats: dict, ensemble_stats: dict):
    print("\n" + "=" * 100)
    print("INDIVIDUAL RUNS  (6 seeds × 5 folds)")
    print("=" * 100)
    header = f"{'Model':<25} {'BAcc':>14} {'Acc':>14} {'AUC':>14} {'wF1':>14} {'N':>4}"
    print(header)
    print("-" * 100)

    for model in MODEL_ORDER:
        if model not in model_stats:
            continue
        s = model_stats[model]
        display = MODEL_DISPLAY.get(model, model)
        row = f"{display:<25}"
        for metric in ["BAcc", "Acc", "AUC", "wF1"]:
            if metric in s:
                m = s[metric]
                row += f" {m['mean']:.3f}±{m['std']:.3f}"
            else:
                row += f" {'—':>14}"
        row += f" {s.get('BAcc', {}).get('n_runs', 0):>4}"
        print(row)

    if ensemble_stats:
        print("\n" + "=" * 100)
        print("PER-SEED SUMMARY  (mean over 5 folds per seed, then across seeds)")
        print("=" * 100)
        print(header)
        print("-" * 100)
        for model in MODEL_ORDER:
            if model not in ensemble_stats or not ensemble_stats[model]:
                continue
            s = ensemble_stats[model]
            display = MODEL_DISPLAY.get(model, model) + " (seed)"
            row = f"{display:<25}"
            for metric in ["BAcc", "Acc", "AUC", "wF1"]:
                if metric in s:
                    m = s[metric]
                    row += f" {m['mean']:.3f}±{m['std']:.3f}"
                else:
                    row += f" {'—':>14}"
            row += f" {s.get('BAcc', {}).get('n_seeds', 0):>4}"
            print(row)


def print_ci_table(model_stats: dict, ensemble_stats: dict):
    print("\n" + "=" * 100)
    print("BOOTSTRAP 95% CI")
    print("=" * 100)
    header = f"{'Model':<25} {'BAcc 95% CI':>22} {'AUC 95% CI':>22}"
    print(header)
    print("-" * 100)

    for model in MODEL_ORDER:
        for src, suffix in [(model_stats, ""), (ensemble_stats, " (ens)")]:
            if model not in src or not src[model]:
                continue
            s = src[model]
            display = MODEL_DISPLAY.get(model, model) + suffix
            bacc = s.get("BAcc", {})
            auc = s.get("AUC", {})
            bc = f"{bacc.get('mean', 0):.3f} [{bacc.get('ci_lo', 0):.3f}, {bacc.get('ci_hi', 0):.3f}]"
            ac = f"{auc.get('mean', 0):.3f} [{auc.get('ci_lo', 0):.3f}, {auc.get('ci_hi', 0):.3f}]"
            print(f"{display:<25} {bc:>22} {ac:>22}")


def print_tests(tests: List[dict]):
    if not tests:
        return
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS  (Wilcoxon signed-rank, paired by seed×fold)")
    print("=" * 100)
    print(f"{'Comparison':<35} {'Ref BAcc':>10} {'Comp BAcc':>10} {'Δ':>8} {'p':>10} {'Sig':>6}")
    print("-" * 100)
    for t in tests:
        sig = "**" if t["significant_001"] else ("*" if t["significant_005"] else "ns")
        comp_display = MODEL_DISPLAY.get(t["comparison"], t["comparison"])
        print(f"ARA-Net vs {comp_display:<23} {t['ref_mean']:>10.3f} {t['comp_mean']:>10.3f} "
              f"{t['delta']:>+8.3f} {t['p_value']:>10.4f} {sig:>6}")


def generate_latex_table(model_stats: dict, ensemble_stats: dict) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Classification performance on ADNI (CN/MCI/AD). "
        r"Mean $\pm$ std over 6 seeds $\times$ 5 folds. "
        r"Bold: best in column. $^\dagger$: cross-seed ensemble.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & BAcc (\%) & Acc (\%) & AUC & wF1 \\",
        r"\midrule",
    ]

    best = {}
    for metric in ["BAcc", "Acc", "AUC", "wF1"]:
        best_val = -1
        for model in MODEL_ORDER:
            for src in [model_stats, ensemble_stats]:
                if model in src and src[model] and metric in src[model]:
                    v = src[model][metric]["mean"]
                    if v > best_val:
                        best_val = v
        best[metric] = best_val

    for model in MODEL_ORDER:
        for src, suffix in [(model_stats, ""), (ensemble_stats, r"$^\dagger$")]:
            if model not in src or not src[model]:
                continue
            s = src[model]
            display = MODEL_DISPLAY.get(model, model) + suffix
            cells = [display]
            for metric in ["BAcc", "Acc", "AUC", "wF1"]:
                if metric not in s:
                    cells.append("—")
                    continue
                m = s[metric]
                scale = 100 if metric in ("BAcc", "Acc") else 1
                val_str = f"{m['mean'] * scale:.1f}" if scale == 100 else f"{m['mean']:.3f}"
                std_str = f"{m['std'] * scale:.1f}" if scale == 100 else f"{m['std']:.3f}"
                cell = f"{val_str}$\\pm${std_str}"
                if abs(m["mean"] - best[metric]) < 1e-4:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(" & ".join(cells) + r" \\")

        if model in ("Ours (no atlas)", "ARA-Net Ensemble"):
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate v3 experiment results")
    parser.add_argument("results_dir", type=str,
                        help="Path to experiment_results_v3/")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <results_dir>/aggregated.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "aggregated.json"

    print(f"Loading results from {results_dir} ...")
    all_results = load_all_results(results_dir)
    print(f"  Loaded {len(all_results)} runs")

    if not all_results:
        print("No results found.")
        sys.exit(1)

    model_groups = group_by_model(all_results)
    model_seed_fold_groups = group_by_model_seed_fold(all_results)

    print("\nComputing cross-variant ensemble (soft-vote) ...")
    ens_runs = compute_variant_ensemble(all_results)
    if ens_runs:
        ens_name = ens_runs[0]["config_name"]
        model_groups[ens_name] = ens_runs
        ens_sf: Dict[int, Dict[int, dict]] = defaultdict(dict)
        for r in ens_runs:
            ens_sf[r["seed"]][r["fold"]] = r
        model_seed_fold_groups[ens_name] = {s: dict(f) for s, f in ens_sf.items()}
        print(f"  {len(ens_runs)} ensemble runs created ({ens_name})")

    print("Computing per-model statistics (all 30 runs) ...")
    model_stats = {}
    for model_name, runs in model_groups.items():
        model_stats[model_name] = compute_model_stats(runs)

    print("Computing per-seed statistics (mean over folds, then across seeds) ...")
    per_seed_stats = {}
    for model_name, seed_folds in model_seed_fold_groups.items():
        per_seed_stats[model_name] = compute_per_seed_stats(seed_folds)

    print("Running statistical tests ...")
    tests = pairwise_tests(model_groups)
    tests += pairwise_tests(model_groups, reference="ARA-Net Ensemble")

    print_table(model_stats, per_seed_stats)
    print_ci_table(model_stats, per_seed_stats)
    print_tests(tests)

    latex = generate_latex_table(model_stats, per_seed_stats)
    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex)

    output = {
        "individual": model_stats,
        "per_seed": per_seed_stats,
        "tests": tests,
        "latex": latex,
        "n_total_runs": len(all_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
