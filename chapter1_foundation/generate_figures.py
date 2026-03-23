#!/usr/bin/env python3
"""
Generate Nature-quality figures from experiment results.

Produces 10 publication figures:
  Fig 1: Training dynamics (loss + balanced acc curves with shading)
  Fig 2: Classification (confusion matrix + ROC + metrics table — combined panel)
  Fig 3: Ablation study (bar chart with error bars + significance tests)
  Fig 4: Baseline comparison (grouped bars: BAcc, F1, AUC)
  Fig 5: Comprehensive metrics table (all models, all metrics)
  Fig 6: t-SNE feature visualization (colored by class)
  Fig 7: Attention analysis (region importance by diagnosis)
  Fig 8: Convergence comparison across models
  Fig 9: Per-class performance breakdown (recall, precision, F1)
  Fig 10: SSL pretraining impact (with vs without pretraining)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon
import scienceplots

# ---------------------------------------------------------------------------
# Nature style constants
# ---------------------------------------------------------------------------

PALETTE = {
    "CN": "#3B82C4", "MCI": "#F39C34", "AD": "#D94F4F",
    "blue": "#3B82C4", "orange": "#F39C34", "red": "#D94F4F",
    "teal": "#2CA6A4", "green": "#5FA55A", "yellow": "#C9A227",
    "purple": "#8D6AB8", "pink": "#E58AAE", "grey": "#9AA0A6",
}

MODEL_COLORS = {
    "ARA-Net Ensemble": "#8B0000",
    "Ours (Atlas+AnatDist)": "#C73737",
    "Ours (Atlas only)": "#E38C2D",
    "Ours (no atlas)": "#C9A227",
    "3D ResNet-18": "#3B82C4",
    "3D ViT": "#2CA6A4",
    "Plain CNN": "#8D6AB8",
}

MODEL_DISPLAY = {
    "ARA-Net Ensemble": "ARA-Net Ens.",
    "Ours (Atlas+AnatDist)": "ARA-Net",
    "Ours (Atlas only)": "ARA-Net (−AD)",
    "Ours (no atlas)": "ARA-Net (−Atl)",
    "3D ResNet-18": "ResNet-18 3D",
    "3D ViT": "ViT 3D",
    "Plain CNN": "Plain CNN",
}

CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_COLORS = ["#3B82C4", "#F39C34", "#D94F4F"]

REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hipp", "L-Amyg",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hipp", "R-Amyg", "R-Acc",
]

AD_RELATED_REGIONS = {"L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"}


def set_nature_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> Dict:
    all_res = {}
    for seed_dir in sorted(results_dir.glob("seed_*")):
        rfile = seed_dir / "all_results.json"
        if rfile.exists():
            with open(rfile) as f:
                data = json.load(f)
                all_res.update(data)

    top_level = results_dir / "all_results.json"
    if top_level.exists():
        with open(top_level) as f:
            all_res.update(json.load(f))

    return all_res


def aggregate_by_model(results: Dict):
    by_model = defaultdict(lambda: {
        "test_acc": [], "test_balanced_acc": [], "val_acc": [],
        "test_y_true": [], "test_y_pred": [], "test_y_prob": [],
        "history": [], "test_features": [],
        "attention_maps": [], "attention_labels": [],
    })
    for key, r in results.items():
        name = r.get("config_name", r.get("model_name", "unknown"))
        by_model[name]["test_acc"].append(r.get("test_acc", 0))
        by_model[name]["test_balanced_acc"].append(
            r.get("test_balanced_acc", r.get("test_acc", 0)))
        by_model[name]["val_acc"].append(
            r.get("best_val_bacc", r.get("best_val_acc", 0)))
        by_model[name]["test_y_true"].append(r["test_y_true"])
        by_model[name]["test_y_pred"].append(r["test_y_pred"])
        by_model[name]["test_y_prob"].append(r["test_y_prob"])
        by_model[name]["history"].append(r.get("history", []))
        if r.get("test_features"):
            by_model[name]["test_features"].append(r["test_features"])
        if r.get("attention_maps"):
            by_model[name]["attention_maps"].extend(r["attention_maps"])
            by_model[name]["attention_labels"].extend(r["attention_labels"])
    return dict(by_model)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_auc(y_true, scores):
    order = np.argsort(-scores)
    ys = y_true[order]
    n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = auc = 0.
    tpr_prev = fpr_prev = 0.
    prev = -np.inf
    for i in range(len(ys)):
        if scores[order[i]] != prev:
            tpr, fpr = tp / n_pos, fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
            prev = scores[order[i]]
        if ys[i] == 1: tp += 1
        else: fp += 1
    tpr, fpr = tp / n_pos, fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
    return auc


def _roc_curve(y_true, scores, n_points=200):
    thresholds = np.linspace(scores.max() + 1e-8, scores.min() - 1e-8, n_points)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    fprs, tprs = [0.0], [0.0]
    for th in thresholds:
        pred = (scores >= th).astype(int)
        fprs.append(((pred == 1) & (y_true == 0)).sum() / max(n_neg, 1))
        tprs.append(((pred == 1) & (y_true == 1)).sum() / max(n_pos, 1))
    fprs.append(1.0); tprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def compute_all_metrics(y_true, y_pred, y_prob):
    acc = (y_true == y_pred).mean()
    n_classes = 3
    f1s, precs, recs, specs = [], [], [], []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        tn = ((y_pred != c) & (y_true != c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1); precs.append(prec); recs.append(rec); specs.append(spec)

    aucs = []
    for c in range(n_classes):
        binary = (y_true == c).astype(int)
        if y_prob.shape[1] > c:
            aucs.append(_binary_auc(binary, y_prob[:, c]))
        else:
            aucs.append(0.5)

    return {
        "acc": acc, "balanced_acc": np.mean(recs),
        "macro_f1": np.mean(f1s), "macro_auc": np.mean(aucs),
        "per_class_f1": dict(zip(CLASS_NAMES, f1s)),
        "per_class_auc": dict(zip(CLASS_NAMES, aucs)),
        "per_class_precision": dict(zip(CLASS_NAMES, precs)),
        "per_class_recall": dict(zip(CLASS_NAMES, recs)),
        "per_class_specificity": dict(zip(CLASS_NAMES, specs)),
    }


def _add_significance(ax, x1, x2, y, p_val, h=0.015):
    if p_val < 0.001: s = "***"
    elif p_val < 0.01: s = "**"
    elif p_val < 0.05: s = "*"
    else: s = "n.s."
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.7, c="k")
    ax.text((x1 + x2) / 2, y + h, s, ha="center", va="bottom", fontsize=6)


def _panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Figure 1: Training Dynamics
# ---------------------------------------------------------------------------

def _pad_to_length(arr_1d, target_len):
    """Pad a 1-D array to *target_len* by repeating its last value."""
    if len(arr_1d) >= target_len:
        return arr_1d[:target_len]
    return np.concatenate([arr_1d, np.full(target_len - len(arr_1d), arr_1d[-1])])


def fig1_training_dynamics(by_model, save_dir):
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6))

    model_order = ["Ours (Atlas+AnatDist)", "Ours (Atlas only)", "Ours (no atlas)",
                   "3D ResNet-18", "Plain CNN"]

    # First pass: find the global max epoch across all models so every
    # curve is drawn over the same x-axis range.
    global_max = 0
    for name in model_order:
        if name not in by_model:
            continue
        for h in by_model[name]["history"]:
            if h:
                global_max = max(global_max, len(h))
    if global_max == 0:
        return

    for name in model_order:
        if name not in by_model:
            continue
        histories = [h for h in by_model[name]["history"] if h]
        if not histories:
            continue

        col = MODEL_COLORS.get(name, "gray")
        short = MODEL_DISPLAY.get(name, name)
        va_key = "val_balanced_acc" if "val_balanced_acc" in histories[0][0] else "val_acc"

        tl = np.array([_pad_to_length(
            np.array([e["train_loss"] for e in h]), global_max) for h in histories])
        vl = np.array([_pad_to_length(
            np.array([e["val_loss"] for e in h]), global_max) for h in histories])
        va = np.array([_pad_to_length(
            np.array([e.get(va_key, e.get("val_acc", 0)) for e in h]), global_max)
            for h in histories])
        epochs = np.arange(1, global_max + 1)

        axes[0].plot(epochs, tl.mean(0), color=col, label=short)
        axes[0].fill_between(epochs, tl.mean(0) - tl.std(0),
                             tl.mean(0) + tl.std(0), alpha=0.1, color=col)
        axes[1].plot(epochs, vl.mean(0), color=col, label=short)
        axes[1].fill_between(epochs, vl.mean(0) - vl.std(0),
                             vl.mean(0) + vl.std(0), alpha=0.1, color=col)
        axes[2].plot(epochs, va.mean(0), color=col, label=short)
        axes[2].fill_between(epochs, va.mean(0) - va.std(0),
                             va.mean(0) + va.std(0), alpha=0.1, color=col)

    labels = ["a", "b", "c"]
    titles = ["Training Loss", "Validation Loss", "Val. Balanced Accuracy"]
    ylabels = ["Loss", "Loss", "Balanced Accuracy"]

    for i, ax in enumerate(axes):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabels[i])
        _panel_label(ax, labels[i])
        ax.set_title(titles[i], fontsize=9, pad=8)
        ax.legend(fontsize=5.5, loc="best", handlelength=1.5)
        ax.grid(True, axis="y")

    fig.tight_layout(w_pad=1.5)
    _save(fig, save_dir, "fig1_training_dynamics")
    print("  Fig 1: Training Dynamics")


# ---------------------------------------------------------------------------
# Figure 2: Classification Performance (combined panel)
# ---------------------------------------------------------------------------

def fig2_classification(by_model, save_dir):
    set_nature_style()

    best_name = "Ours (Atlas+AnatDist)"
    if best_name not in by_model:
        best_name = list(by_model.keys())[0]

    all_yt = np.concatenate([np.array(x) for x in by_model[best_name]["test_y_true"]])
    all_yp = np.concatenate([np.array(x) for x in by_model[best_name]["test_y_pred"]])
    all_yprob = np.concatenate([np.array(x) for x in by_model[best_name]["test_y_prob"]])

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1.0, 0.9],
        hspace=0.38,
        wspace=0.32
    )

    FONT_LABEL = 9.5
    FONT_TICK = 8.5
    FONT_LEGEND = 8

    # --- a: Confusion Matrix ---
    ax0 = fig.add_subplot(gs[0, 0])

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(all_yt, all_yp):
        cm[int(t), int(p)] += 1
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cm_cmap = LinearSegmentedColormap.from_list(
        "fig1_red", ["#FBEAE8", "#E8A9A0", "#C73737", "#8B1A1A"])
    ax0.imshow(cm_norm, cmap=cm_cmap, vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            txt = f"{cm_norm[i, j]*100:.1f}%"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax0.text(j, i, txt, ha="center", va="center",
                     fontsize=8.5, color=color,
                     fontweight="bold" if i == j else "normal")

    ax0.set_xticks([0, 1, 2])
    ax0.set_yticks([0, 1, 2])
    ax0.set_xticklabels(CLASS_NAMES, fontsize=FONT_TICK)
    ax0.set_yticklabels(CLASS_NAMES, fontsize=FONT_TICK)
    ax0.set_xlabel("Predicted", fontsize=FONT_LABEL)
    ax0.set_ylabel("True", fontsize=FONT_LABEL)
    _panel_label(ax0, "a")

    # --- b: ROC Curves ---
    ax1 = fig.add_subplot(gs[0, 1])

    for c in range(3):
        binary = (all_yt == c).astype(int)
        scores = all_yprob[:, c]
        fprs, tprs = _roc_curve(binary, scores)
        auc_val = _binary_auc(binary, scores)
        ax1.plot(fprs, tprs, lw=1.8,
                 label=f"{CLASS_NAMES[c]} (AUC={auc_val:.3f})")

    ax1.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax1.set_xlabel("False Positive Rate", fontsize=FONT_LABEL)
    ax1.set_ylabel("True Positive Rate", fontsize=FONT_LABEL)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.legend(fontsize=FONT_LEGEND, frameon=False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    _panel_label(ax1, "b")

    # --- c: Performance Comparison (bar plot) ---
    ax2 = fig.add_subplot(gs[1, :])

    model_names = list(by_model.keys())
    auc_means = []
    auc_stds = []

    for name in model_names:
        aucs = []
        for yt, yp, yprob in zip(
            by_model[name]["test_y_true"],
            by_model[name]["test_y_pred"],
            by_model[name]["test_y_prob"]
        ):
            yt = np.array(yt)
            yprob = np.array(yprob)
            per_class_auc = []
            for c in range(3):
                binary = (yt == c).astype(int)
                per_class_auc.append(_binary_auc(binary, yprob[:, c]))
            aucs.append(np.mean(per_class_auc))
        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs))

    x = np.arange(len(model_names))
    bars = ax2.bar(x, auc_means, yerr=auc_stds, capsize=3,
                   edgecolor="black", linewidth=0.6)

    for i, name in enumerate(model_names):
        bars[i].set_color(MODEL_COLORS.get(name, "#9AA0A6"))

    for i in range(len(model_names)):
        ax2.text(i, auc_means[i] + 0.01, f"{auc_means[i]:.3f}",
                 ha='center', fontsize=7)

    display_names = [MODEL_DISPLAY.get(n, n) for n in model_names]
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=20, ha="right",
                        fontsize=FONT_TICK)
    ax2.set_ylabel("Macro AUC", fontsize=FONT_LABEL)
    ax2.set_ylim(0.45, 1.0)
    ax2.tick_params(labelsize=FONT_TICK)
    _panel_label(ax2, "c")

    _save(fig, save_dir, "fig2_classification")
    print("  Fig 2: Classification Performance (Upgraded)")


# ---------------------------------------------------------------------------
# Figure 3: Ablation Study
# ---------------------------------------------------------------------------

def fig3_ablation(by_model, save_dir):
    set_nature_style()
    ablation_names = ["Ours (no atlas)", "Ours (Atlas only)", "Ours (Atlas+AnatDist)"]
    ablation_labels = ["ARA-Net\n(−Atlas)", "ARA-Net\n(−AnatDist)", "ARA-Net\n(Full)"]
    available = [n for n in ablation_names if n in by_model]
    if len(available) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))

    for metric_idx, (metric_name, get_val) in enumerate([
        ("Balanced Accuracy", lambda m: m["balanced_acc"]),
        ("Macro F1-Score", lambda m: m["macro_f1"]),
        ("Macro AUC", lambda m: m["macro_auc"]),
    ]):
        ax = axes[metric_idx]
        x = np.arange(len(available))
        means, stds, all_vals = [], [], []
        for name in available:
            vals = []
            for yt, yp, yprob in zip(by_model[name]["test_y_true"],
                                     by_model[name]["test_y_pred"],
                                     by_model[name]["test_y_prob"]):
                m = compute_all_metrics(np.array(yt), np.array(yp), np.array(yprob))
                vals.append(get_val(m))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            all_vals.append(vals)

        colors = [MODEL_COLORS.get(n, "gray") for n in available]
        short = [ablation_labels[ablation_names.index(n)] for n in available]

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
        for bar in bars:
            bar.set_edgecolor(bar.get_facecolor())
            bar.set_linewidth(0)

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.025, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

        if len(all_vals) >= 2:
            for i in range(len(all_vals) - 1):
                a, b = np.array(all_vals[i]), np.array(all_vals[i + 1])
                min_l = min(len(a), len(b))
                if min_l >= 5:
                    try:
                        _, p = wilcoxon(a[:min_l], b[:min_l])
                    except Exception:
                        p = 1.0
                    y_max = max(means[i] + stds[i], means[i + 1] + stds[i + 1]) + 0.08
                    _add_significance(ax, i, i + 1, y_max, p)

        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel(metric_name)
        _panel_label(ax, "abc"[metric_idx])
        ax.set_ylim(0, min(max(means) + max(stds) + 0.22, 1.12))
        ax.grid(True, axis="y")

    fig.tight_layout(w_pad=1.5)
    _save(fig, save_dir, "fig3_ablation")
    print("  Fig 3: Ablation Study")


# ---------------------------------------------------------------------------
# Figure 4: Baseline Comparison
# ---------------------------------------------------------------------------

def fig4_baselines(by_model, save_dir):
    set_nature_style()
    order = ["Plain CNN", "3D ResNet-18", "3D ViT",
             "Ours (no atlas)", "Ours (Atlas only)", "Ours (Atlas+AnatDist)"]
    available = [n for n in order if n in by_model]
    if len(available) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.4))
    x = np.arange(len(available))

    for metric_idx, (metric_name, get_val) in enumerate([
        ("Balanced Accuracy", lambda m: m["balanced_acc"]),
        ("Macro F1-Score", lambda m: m["macro_f1"]),
        ("Macro AUC", lambda m: m["macro_auc"]),
    ]):
        ax = axes[metric_idx]
        means, stds = [], []
        for name in available:
            vals = []
            for yt, yp, yprob in zip(by_model[name]["test_y_true"],
                                     by_model[name]["test_y_pred"],
                                     by_model[name]["test_y_prob"]):
                m = compute_all_metrics(np.array(yt), np.array(yp), np.array(yprob))
                vals.append(get_val(m))
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        colors = [MODEL_COLORS.get(n, "gray") for n in available]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors,
               edgecolor="white", linewidth=0, width=0.6, zorder=3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=6, fontweight="bold")

        short = [MODEL_DISPLAY.get(n, n) for n in available]
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=5.5, rotation=35, ha="right")
        ax.set_ylabel(metric_name)
        _panel_label(ax, "abc"[metric_idx])
        ax.set_ylim(0, min(max(means) + max(stds) + 0.15, 1.08))
        ax.grid(True, axis="y")

    fig.tight_layout(w_pad=1.5)
    _save(fig, save_dir, "fig4_baselines")
    print("  Fig 4: Baseline Comparison")


# ---------------------------------------------------------------------------
# Figure 5: Comprehensive Metrics Table
# ---------------------------------------------------------------------------

def fig5_metrics_table(by_model, save_dir):
    set_nature_style()
    order = ["Plain CNN", "3D ResNet-18", "3D ViT",
             "Ours (no atlas)", "Ours (Atlas only)", "Ours (Atlas+AnatDist)"]
    available = [n for n in order if n in by_model]

    fig, ax = plt.subplots(figsize=(7.8, 0.6 + 0.45 * len(available)))
    ax.axis("off")

    columns = ["Model", "Acc", "BAcc", "F1", "AUC",
               "CN-AUC", "MCI-AUC", "AD-AUC"]
    cell_data = []

    best_bacc = -1
    best_row = -1
    for row_i, name in enumerate(available):
        all_m = []
        for yt, yp, yprob in zip(by_model[name]["test_y_true"],
                                 by_model[name]["test_y_pred"],
                                 by_model[name]["test_y_prob"]):
            all_m.append(compute_all_metrics(np.array(yt), np.array(yp), np.array(yprob)))

        def fmt(key):
            vals = [m[key] for m in all_m]
            return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

        bacc_mean = np.mean([m["balanced_acc"] for m in all_m])
        if bacc_mean > best_bacc:
            best_bacc = bacc_mean
            best_row = row_i

        cell_data.append([
            MODEL_DISPLAY.get(name, name),
            fmt("acc"), fmt("balanced_acc"), fmt("macro_f1"), fmt("macro_auc"),
            f"{np.mean([m['per_class_auc']['CN'] for m in all_m]):.3f}",
            f"{np.mean([m['per_class_auc']['MCI'] for m in all_m]):.3f}",
            f"{np.mean([m['per_class_auc']['AD'] for m in all_m]):.3f}",
        ])

    tbl = ax.table(cellText=cell_data, colLabels=columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.6)
    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#cccccc")
        if key[0] == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif key[0] == best_row + 1:
            cell.set_facecolor("#ffe0e0")

    ax.set_title("Table 1: Classification Performance (mean ± std across folds and seeds)",
                 fontsize=9, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, save_dir, "fig5_metrics_table")
    print("  Fig 5: Metrics Table")


# ---------------------------------------------------------------------------
# Figure 6: t-SNE
# ---------------------------------------------------------------------------

def fig6_tsne(by_model, save_dir):
    set_nature_style()

    target_name = "Ours (Atlas+AnatDist)"
    if target_name not in by_model:
        target_name = list(by_model.keys())[0]

    feats_lists = by_model[target_name].get("test_features", [])
    if not feats_lists:
        print("  [Fig 6: t-SNE skipped — no features]")
        return

    best_idx = int(np.argmax(by_model[target_name]["test_balanced_acc"]))
    if best_idx < len(feats_lists):
        all_feats = np.array(feats_lists[best_idx])
        all_labels = np.array(by_model[target_name]["test_y_true"][best_idx])
    else:
        all_feats = np.array(feats_lists[0])
        all_labels = np.array(by_model[target_name]["test_y_true"][0])

    if len(all_feats) < 15:
        print("  [Fig 6: t-SNE skipped — too few samples]")
        return

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        feats_scaled = StandardScaler().fit_transform(all_feats)
        perp = min(30, len(feats_scaled) // 3, len(feats_scaled) - 1)
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=max(5, perp), max_iter=1500,
                    learning_rate="auto", init="pca")
        emb = tsne.fit_transform(feats_scaled)
    except ImportError:
        print("  [Fig 6: t-SNE skipped — sklearn not installed]")
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for c in range(3):
        mask = all_labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=CLASS_COLORS[c],
                   label=CLASS_NAMES[c], s=18, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.8, fontsize=7, loc="best")
    ax.set_title("Feature Space (t-SNE)", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.08)
    fig.tight_layout()
    _save(fig, save_dir, "fig6_tsne")
    print("  Fig 6: t-SNE")


# ---------------------------------------------------------------------------
# Figure 7: Attention Analysis
# ---------------------------------------------------------------------------

def fig7_attention(by_model, save_dir):
    set_nature_style()

    target_name = "Ours (Atlas+AnatDist)"
    if target_name not in by_model:
        return

    attn_maps = by_model[target_name].get("attention_maps", [])
    attn_labels = by_model[target_name].get("attention_labels", [])
    if not attn_maps:
        print("  [Fig 7: Attention skipped — no data]")
        return

    attn_by_group = {0: [], 1: [], 2: []}
    for attn, lbl in zip(attn_maps, attn_labels):
        attn_arr = np.array(attn)
        if attn_arr.ndim >= 3:
            avg_attn = attn_arr.mean(axis=0)
        else:
            avg_attn = attn_arr
        if avg_attn.ndim == 2:
            region_importance = avg_attn.mean(axis=0)
        else:
            region_importance = avg_attn
        n_regions = min(len(region_importance), len(REGION_NAMES))
        attn_by_group[lbl].append(region_importance[:n_regions])

    # Panel a: heatmap, Panel b-d: bar charts per group
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # Heatmap: regions × groups
    ax_heat = fig.add_subplot(gs[0, :])
    group_means = []
    for c in range(3):
        if attn_by_group[c]:
            group_means.append(np.mean(attn_by_group[c], axis=0))
        else:
            group_means.append(np.zeros(len(REGION_NAMES)))
    heat_data = np.array(group_means)  # (3, n_regions)
    n_r = heat_data.shape[1]
    names = REGION_NAMES[:n_r]

    cmap_heat = LinearSegmentedColormap.from_list("heat", ["#ffffff", "#E15759"])
    im = ax_heat.imshow(heat_data, cmap=cmap_heat, aspect="auto")
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_yticklabels(CLASS_NAMES)
    ax_heat.set_xticks(range(n_r))
    ax_heat.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="Attention Weight")
    _panel_label(ax_heat, "a", x=-0.05, y=1.12)
    ax_heat.set_title("Regional Attention Heatmap", fontsize=9, pad=8)

    for ci, c_name in enumerate(CLASS_NAMES):
        if not attn_by_group[ci]:
            continue
        ax = fig.add_subplot(gs[1, ci])
        data = np.array(attn_by_group[ci])
        n_r = data.shape[1]
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        sorted_idx = np.argsort(means)[::-1][:10]

        y = np.arange(len(sorted_idx))
        colors_bar = []
        for idx in sorted_idx:
            if idx < len(names) and names[idx] in AD_RELATED_REGIONS:
                colors_bar.append(CLASS_COLORS[ci])
            else:
                colors_bar.append("#cccccc")
        ax.barh(y, means[sorted_idx], xerr=stds[sorted_idx], capsize=2,
                color=colors_bar, alpha=0.85, height=0.65, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] if i < len(names) else f"R{i}"
                            for i in sorted_idx], fontsize=6)
        ax.set_xlabel("Weight", fontsize=7)
        ax.set_title(f"{c_name} (n={len(attn_by_group[ci])})", fontsize=8)
        ax.invert_yaxis()
        _panel_label(ax, "bcd"[ci])

    fig.tight_layout()
    _save(fig, save_dir, "fig7_attention")
    print("  Fig 7: Attention Analysis")


# ---------------------------------------------------------------------------
# Figure 8: Convergence Comparison
# ---------------------------------------------------------------------------

def fig8_convergence(by_model, save_dir):
    set_nature_style()
    order = ["Plain CNN", "3D ResNet-18", "3D ViT", "Ours (Atlas+AnatDist)"]
    available = [n for n in order if n in by_model]
    if not available:
        return

    global_max = 0
    for name in available:
        for h in by_model[name]["history"]:
            if h:
                global_max = max(global_max, len(h))
    if global_max == 0:
        return

    fig, ax = plt.subplots(figsize=(4.5, 3))
    for name in available:
        histories = [h for h in by_model[name]["history"] if h]
        if not histories:
            continue
        va_key = "val_balanced_acc" if "val_balanced_acc" in histories[0][0] else "val_acc"
        val_accs = np.array([_pad_to_length(
            np.array([e.get(va_key, e.get("val_acc", 0)) for e in h]), global_max)
            for h in histories])
        epochs = np.arange(1, global_max + 1)
        col = MODEL_COLORS.get(name, "gray")
        short = MODEL_DISPLAY.get(name, name)
        ax.plot(epochs, val_accs.mean(0), color=col, label=short)
        ax.fill_between(epochs, val_accs.mean(0) - val_accs.std(0),
                        val_accs.mean(0) + val_accs.std(0), alpha=0.1, color=col)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Convergence Comparison", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save(fig, save_dir, "fig8_convergence")
    print("  Fig 8: Convergence")


# ---------------------------------------------------------------------------
# Figure 9: Per-Class Performance
# ---------------------------------------------------------------------------

def fig9_per_class(by_model, save_dir):
    set_nature_style()
    order = ["Plain CNN", "3D ResNet-18", "Ours (Atlas+AnatDist)"]
    available = [n for n in order if n in by_model]
    if not available:
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.2))

    for metric_idx, (metric_name, key) in enumerate([
        ("Recall (Sensitivity)", "per_class_recall"),
        ("Precision", "per_class_precision"),
        ("F1-Score", "per_class_f1"),
    ]):
        ax = axes[metric_idx]
        width = 0.24
        x = np.arange(len(CLASS_NAMES))

        for model_idx, name in enumerate(available):
            vals_by_class = {c: [] for c in CLASS_NAMES}
            for yt, yp, yprob in zip(by_model[name]["test_y_true"],
                                     by_model[name]["test_y_pred"],
                                     by_model[name]["test_y_prob"]):
                m = compute_all_metrics(np.array(yt), np.array(yp), np.array(yprob))
                for c in CLASS_NAMES:
                    vals_by_class[c].append(m[key][c])

            means = [np.mean(vals_by_class[c]) for c in CLASS_NAMES]
            stds = [np.std(vals_by_class[c]) for c in CLASS_NAMES]
            offset = (model_idx - len(available) / 2 + 0.5) * width
            col = MODEL_COLORS.get(name, "gray")
            short = MODEL_DISPLAY.get(name, name)
            ax.bar(x + offset, means, width * 0.85, yerr=stds, capsize=2,
                   color=col, label=short, edgecolor="white", linewidth=0,
                   zorder=3, alpha=0.9)

        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_ylabel(metric_name)
        _panel_label(ax, "abc"[metric_idx])
        ax.set_ylim(0, 1.08)
        ax.grid(True, axis="y")
        if metric_idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout(w_pad=1.5)
    _save(fig, save_dir, "fig9_per_class")
    print("  Fig 9: Per-Class Performance")


# ---------------------------------------------------------------------------
# Figure 10: SSL Pretraining Impact
# ---------------------------------------------------------------------------

def fig10_ssl_impact(by_model, save_dir, v2_dir: Optional[Path] = None):
    """Compare SSL-pretrained vs random-init (v2) if v2 results exist."""
    set_nature_style()

    if v2_dir is None or not v2_dir.exists():
        print("  [Fig 10: SSL Impact skipped — no v2 results for comparison]")
        return

    v2_results = load_results(v2_dir)
    if not v2_results:
        print("  [Fig 10: SSL Impact skipped — v2 results empty]")
        return
    v2_by_model = aggregate_by_model(v2_results)

    target = "Ours (Atlas+AnatDist)"
    if target not in by_model or target not in v2_by_model:
        print("  [Fig 10: SSL Impact skipped — Ours not in both]")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    labels = ["Random Init", "SSL Pretrained"]
    colors = ["#BAB0AC", "#E15759"]

    for metric_idx, (metric_name, get_val) in enumerate([
        ("Balanced Accuracy", lambda m: m["balanced_acc"]),
        ("Macro F1-Score", lambda m: m["macro_f1"]),
        ("Macro AUC", lambda m: m["macro_auc"]),
    ]):
        ax = axes[metric_idx]
        datasets = [v2_by_model, by_model]
        means, stds = [], []
        for ds in datasets:
            vals = []
            for yt, yp, yprob in zip(ds[target]["test_y_true"],
                                     ds[target]["test_y_pred"],
                                     ds[target]["test_y_prob"]):
                m = compute_all_metrics(np.array(yt), np.array(yp), np.array(yprob))
                vals.append(get_val(m))
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(2)
        ax.bar(x, means, yerr=stds, capsize=5, color=colors,
               edgecolor="white", linewidth=0, width=0.5, zorder=3)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        improvement = means[1] - means[0]
        ax.annotate(f"+{improvement:.3f}", xy=(1, means[1]),
                    xytext=(1.3, means[1] - 0.05),
                    fontsize=7, color="#E15759", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#E15759", lw=1))

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric_name)
        _panel_label(ax, "abc"[metric_idx])
        ax.grid(True, axis="y")
        ax.set_ylim(0, min(max(means) + max(stds) + 0.12, 1.05))

    fig.suptitle("Impact of Self-Supervised Pretraining on ARA-Net",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=1.5)
    _save(fig, save_dir, "fig10_ssl_impact")
    print("  Fig 10: SSL Pretraining Impact")


# ---------------------------------------------------------------------------
# Figure 11: SOTA Comparison Bar Chart
# ---------------------------------------------------------------------------

SOTA_LITERATURE = [
    # (name, bacc, acc, color) — None means metric not reported
    # Rigorous evaluation (CV or proper split)
    ("3D-CNN (Korolev '17)", None, 59.7, "#BAB0AC"),
    ("THAN (Zhang '22)", None, 62.9, "#9AA0A6"),
    ("STNet (Jia '23)", None, 71.8, "#8D6AB8"),
    ("LSTM-Robust (Gao '23)", None, 76.0, "#2CA6A4"),
    ("ECAResNet269+FL (Alkhathami '25)", 74.0, None, "#5FA55A"),
    # Single-split (shown for context, marked with †)
    ("DEMNET† (Murugan '21)", None, 95.2, "#C9C9C9"),
    ("3D HCCT† (Majee '24)", None, 96.1, "#C9C9C9"),
]


def fig11_sota_comparison(by_model, save_dir, aggregated_path: Optional[Path] = None):
    set_nature_style()

    agg = None
    if aggregated_path and aggregated_path.exists():
        with open(aggregated_path) as f:
            agg = json.load(f)

    own_models = []
    for model_name in ["Ours (Atlas+AnatDist)", "Ours (no atlas)"]:
        if agg:
            src = agg.get("individual", {})
            if model_name in src and src[model_name]:
                bacc_info = src[model_name].get("BAcc", {})
                acc_info = src[model_name].get("Acc", {})
                display = MODEL_DISPLAY.get(model_name, model_name)
                own_models.append({
                    "name": display,
                    "bacc": bacc_info.get("mean", 0) * 100,
                    "bacc_std": bacc_info.get("std", 0) * 100,
                    "acc": acc_info.get("mean", 0) * 100,
                    "acc_std": acc_info.get("std", 0) * 100,
                })
        elif model_name in by_model:
            baccs = by_model[model_name]["test_balanced_acc"]
            display = MODEL_DISPLAY.get(model_name, model_name)
            own_models.append({
                "name": display,
                "bacc": np.mean(baccs) * 100,
                "bacc_std": np.std(baccs) * 100,
                "acc": np.mean(by_model[model_name]["test_acc"]) * 100,
                "acc_std": np.std(by_model[model_name]["test_acc"]) * 100,
            })

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))

    all_entries = []
    for name, bacc, acc, color in SOTA_LITERATURE:
        all_entries.append({"name": name, "bacc": bacc, "acc": acc, "color": color, "ours": False})
    for o in own_models:
        all_entries.append({
            "name": o["name"], "bacc": o["bacc"], "acc": o["acc"],
            "color": "#C73737", "ours": True, "bacc_std": o.get("bacc_std"),
        })

    y_pos = np.arange(len(all_entries))
    for i, entry in enumerate(all_entries):
        val = entry.get("bacc") or entry.get("acc") or 0
        err = entry.get("bacc_std", 0) if entry["ours"] else 0
        color = entry["color"]
        alpha = 1.0 if entry["ours"] else 0.7
        edgecolor = "#8B0000" if entry["ours"] else "none"
        lw = 1.5 if entry["ours"] else 0

        bar = ax.barh(i, val, xerr=err if err > 0 else None, capsize=3,
                       color=color, alpha=alpha, edgecolor=edgecolor, linewidth=lw,
                       height=0.6, zorder=3)
        metric_label = "BAcc" if entry.get("bacc") else "Acc"
        txt = f"{val:.1f}%" if err == 0 else f"{val:.1f}±{err:.1f}%"
        ax.text(val + 1.5, i, f"{txt} ({metric_label})", va="center", fontsize=7,
                fontweight="bold" if entry["ours"] else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["name"] for e in all_entries], fontsize=8)
    ax.set_xlabel("Performance (%)")
    ax.set_title("Comparison with Recent ADNI 3-Class Methods", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, save_dir, "fig11_sota_comparison")
    print("  Fig 11: SOTA Comparison")


# ---------------------------------------------------------------------------
# Figure 12: External Validation (IXI + OASIS)
# ---------------------------------------------------------------------------

def fig12_external_validation(save_dir, ext_results_path: Optional[Path] = None):
    set_nature_style()

    if ext_results_path is None or not ext_results_path.exists():
        print("  [Fig 12: External Validation skipped — no results file]")
        return

    with open(ext_results_path) as f:
        ext = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Panel A: IXI CN Specificity
    ax = axes[0]
    ixi = ext.get("ixi", {})
    specs = [r["cn_specificity"] for r in ixi.get("per_checkpoint", [])]
    if specs:
        mean_spec = np.mean(specs)
        std_spec = np.std(specs)

        ax.hist(specs, bins=15, color="#3B82C4", alpha=0.75, edgecolor="white", zorder=3)
        ax.axvline(mean_spec, color="#C73737", linewidth=2, linestyle="--",
                   label=f"Mean: {mean_spec:.3f}±{std_spec:.3f}", zorder=4)
        if ixi.get("ensemble"):
            ens_spec = ixi["ensemble"]["cn_specificity"]
            ax.axvline(ens_spec, color="#F39C34", linewidth=2, linestyle="-.",
                       label=f"Ensemble: {ens_spec:.3f}", zorder=4)
        ax.set_xlabel("CN Specificity")
        ax.set_ylabel("Count (checkpoints)")
        ax.set_title("a  IXI External Validation (CN)", fontsize=9, fontweight="bold", loc="left")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "No IXI results", ha="center", va="center", transform=ax.transAxes)

    # Panel B: OASIS 3-class
    ax = axes[1]
    oasis = ext.get("oasis", {})
    baccs = [r["bacc"] for r in oasis.get("per_checkpoint", [])]
    aucs = [r["macro_auc"] for r in oasis.get("per_checkpoint", [])]
    if baccs:
        x = np.arange(2)
        width = 0.35

        mean_bacc = np.mean(baccs)
        std_bacc = np.std(baccs)
        mean_auc = np.mean(aucs) if aucs else 0
        std_auc = np.std(aucs) if aucs else 0

        bars1 = ax.bar(x[0], mean_bacc, width, yerr=std_bacc, capsize=4,
                       color="#C73737", alpha=0.85, zorder=3, label=f"BAcc: {mean_bacc:.3f}")
        bars2 = ax.bar(x[1], mean_auc, width, yerr=std_auc, capsize=4,
                       color="#3B82C4", alpha=0.85, zorder=3, label=f"AUC: {mean_auc:.3f}")

        if oasis.get("ensemble"):
            ens = oasis["ensemble"]
            ax.bar(x[0] + width, ens["bacc"], width, color="#C73737", alpha=0.5,
                   edgecolor="#C73737", linewidth=1.5, zorder=3, hatch="//",
                   label=f"Ens BAcc: {ens['bacc']:.3f}")
            ax.bar(x[1] + width, ens["macro_auc"], width, color="#3B82C4", alpha=0.5,
                   edgecolor="#3B82C4", linewidth=1.5, zorder=3, hatch="//",
                   label=f"Ens AUC: {ens['macro_auc']:.3f}")

        ax.set_xticks([0.175, 1.175])
        ax.set_xticklabels(["BAcc", "Macro AUC"])
        ax.set_ylabel("Score")
        ax.set_title("b  OASIS External Validation", fontsize=9, fontweight="bold", loc="left")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "No OASIS results", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(w_pad=2)
    _save(fig, save_dir, "fig12_external_validation")
    print("  Fig 12: External Validation")


# ---------------------------------------------------------------------------
# Figure 13: Bootstrap CI Forest Plot
# ---------------------------------------------------------------------------

def _ci_forest_single(save_dir, agg, metric_name, metric_key, panel_label, filename):
    """Draw one CI forest plot for a single metric and save it."""
    set_nature_style()

    entries = []
    for src_key in ["individual"]:
        src = agg.get(src_key, {})
        model_order = [
            "Ours (Atlas+AnatDist)",
            "Ours (Atlas only)",
            "Ours (no atlas)",
            "3D ResNet-18",
            "3D ViT",
            "Plain CNN",
        ]
        for model_name in model_order:
            if model_name not in src or not src[model_name]:
                continue
            stats = src[model_name].get(metric_key, {})
            if not stats:
                continue
            display = MODEL_DISPLAY.get(model_name, model_name)
            scale = 100 if metric_key in ("BAcc", "Acc") else 1
            entries.append({
                "name": display,
                "mean": stats["mean"] * scale,
                "ci_lo": stats["ci_lo"] * scale,
                "ci_hi": stats["ci_hi"] * scale,
                "color": MODEL_COLORS.get(model_name, "#9AA0A6"),
                "is_ours": "Ours" in model_name,
            })

    if not entries:
        return

    n = len(entries)
    fig, ax = plt.subplots(figsize=(5.5, 0.45 * n + 0.8))

    y_pos = np.arange(n)
    ci_max = max(e["ci_hi"] for e in entries)
    x_pad = (ci_max - min(e["ci_lo"] for e in entries)) * 0.12

    for i, e in enumerate(entries):
        marker = "D" if e["is_ours"] else "o"
        ms = 7 if e["is_ours"] else 5
        lw = 2.2 if e["is_ours"] else 1.2
        ax.plot(e["mean"], i, marker, color=e["color"], markersize=ms, zorder=5)
        ax.hlines(i, e["ci_lo"], e["ci_hi"], color=e["color"], linewidth=lw, zorder=4)
        unit = "%" if metric_key in ("BAcc", "Acc") else ""
        label = f"{e['mean']:.1f}{unit}  [{e['ci_lo']:.1f}, {e['ci_hi']:.1f}]"
        ax.text(ci_max + x_pad, i, label, va="center", fontsize=7,
                color=e["color"], fontweight="bold" if e["is_ours"] else "normal",
                clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e["name"] for e in entries], fontsize=8)
    xlabel = f"{metric_name} (%)" if metric_key in ("BAcc", "Acc") else metric_name
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"{panel_label}  {metric_name} — Bootstrap 95% CI",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(left=None, right=ci_max + x_pad * 5)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, save_dir, filename)


def fig13_ci_forest(save_dir, aggregated_path: Optional[Path] = None):
    if aggregated_path is None or not aggregated_path.exists():
        print("  [Fig 13: CI Forest Plot skipped — no aggregated results]")
        return
    with open(aggregated_path) as f:
        agg = json.load(f)

    _ci_forest_single(save_dir, agg, "Balanced Accuracy", "BAcc", "a", "fig13a_ci_bacc")
    _ci_forest_single(save_dir, agg, "Macro AUC", "AUC", "b", "fig13b_ci_auc")
    print("  Fig 13a: CI Forest (BAcc)")
    print("  Fig 13b: CI Forest (AUC)")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, save_dir, name):
    fig.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_dir / f"{name}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "chapter1_foundation/experiment_results_ssl")
    save_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "chapter1_foundation/figures_ssl")
    save_dir.mkdir(parents=True, exist_ok=True)

    v2_dir = Path("chapter1_foundation/archive/legacy_results/experiment_results_v2")
    agg_path = results_dir / "aggregated.json"
    if not agg_path.exists():
        agg_path = Path("chapter1_foundation/experiment_results_v3/aggregated.json")
    ext_path = Path("chapter1_foundation/external_validation_results.json")

    print("=" * 60)
    print("Generating Nature-Quality Figures")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {save_dir}")

    results = load_results(results_dir)
    if not results:
        print("ERROR: No results found!")
        return

    by_model = aggregate_by_model(results)

    # --- inject ensemble (soft-vote across 3 ARA-Net variants per seed×fold) ---
    import re as _re
    _variants = ["Ours (Atlas+AnatDist)", "Ours (Atlas only)", "Ours (no atlas)"]
    _sf: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for key, val in results.items():
        parts = key.split("__")
        model = parts[0]
        if model not in _variants:
            continue
        seed = fold = None
        for p in parts:
            ms = _re.match(r"seed(\d+)", p)
            mf = _re.match(r"fold(\d+)", p)
            if ms: seed = int(ms.group(1))
            if mf: fold = int(mf.group(1))
        if seed is not None and fold is not None:
            _sf[(seed, fold)][model] = val

    ens_key = "ARA-Net Ensemble"
    by_model[ens_key] = {
        "test_acc": [], "test_balanced_acc": [], "val_acc": [],
        "test_y_true": [], "test_y_pred": [], "test_y_prob": [],
        "history": [], "test_features": [],
        "attention_maps": [], "attention_labels": [],
    }
    for (seed, fold), models in sorted(_sf.items()):
        runs = [models[v] for v in _variants if v in models]
        if len(runs) < len(_variants):
            continue
        y_true = np.array(runs[0]["test_y_true"])
        avg_prob = np.mean([np.array(r["test_y_prob"]) for r in runs], axis=0)
        y_pred = avg_prob.argmax(axis=1)
        recalls = [float((y_pred[y_true == c] == c).mean())
                   for c in range(3) if (y_true == c).sum() > 0]
        bacc = float(np.mean(recalls))
        acc = float((y_pred == y_true).mean())
        by_model[ens_key]["test_acc"].append(acc)
        by_model[ens_key]["test_balanced_acc"].append(bacc)
        by_model[ens_key]["val_acc"].append(bacc)
        by_model[ens_key]["test_y_true"].append(y_true.tolist())
        by_model[ens_key]["test_y_pred"].append(y_pred.tolist())
        by_model[ens_key]["test_y_prob"].append(avg_prob.tolist())
        by_model[ens_key]["history"].append([])
    print(f"  Ensemble injected: {len(by_model[ens_key]['test_acc'])} runs")
    # --- end ensemble injection ---

    n_runs = len(results)
    n_models = len(by_model)
    print(f"Loaded {n_runs} runs + ensemble across {n_models} model configs")

    for name in by_model:
        n = len(by_model[name]["test_acc"])
        bacc = np.mean(by_model[name]["test_balanced_acc"])
        print(f"  {name}: {n} runs, mean test_bacc={bacc:.4f}")
    print()

    fig1_training_dynamics(by_model, save_dir)
    fig2_classification(by_model, save_dir)
    fig3_ablation(by_model, save_dir)
    fig4_baselines(by_model, save_dir)
    fig5_metrics_table(by_model, save_dir)
    fig6_tsne(by_model, save_dir)
    fig7_attention(by_model, save_dir)
    fig8_convergence(by_model, save_dir)
    fig9_per_class(by_model, save_dir)
    fig10_ssl_impact(by_model, save_dir, v2_dir)
    fig11_sota_comparison(by_model, save_dir, agg_path if agg_path.exists() else None)
    fig12_external_validation(save_dir, ext_path if ext_path.exists() else None)
    fig13_ci_forest(save_dir, agg_path if agg_path.exists() else None)

    print(f"\nAll figures saved to: {save_dir}/")
    print("Formats: PNG (300 dpi) + PDF (vector)")


if __name__ == "__main__":
    main()
