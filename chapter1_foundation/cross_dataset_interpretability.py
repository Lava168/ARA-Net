#!/usr/bin/env python3
"""
Cross-Dataset Interpretability Generalization Analysis.

Tests whether the attention patterns learned on ADNI generalize to
external datasets (IXI, OASIS). This is a novel contribution —
most papers only validate classification accuracy, not interpretability.

Key analyses:
1. IXI (all CN): attention pattern should match ADNI-CN pattern
2. OASIS (CN/MCI/AD): attention group differences should replicate ADNI findings
3. Cross-dataset attention consistency (cosine similarity, rank correlation)

Usage:
    python -m chapter1_foundation.cross_dataset_interpretability \
        --results_dir chapter1_foundation/experiment_results_v3 \
        --data_root sample_data \
        --output chapter1_foundation/cross_dataset_interpretability_results.json \
        --figures chapter1_foundation/figures_cross_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapter1_foundation.models.atlas_guided_model import create_model
from chapter1_foundation.data.foundation_loader import remap_segmentation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

CLASS_NAMES = ["CN", "MCI", "AD"]
NUM_MAPPED_REGIONS = 21

REGION_NAMES = [
    "L-WM", "L-Ctx", "L-Vent", "L-Thal", "L-Caud",
    "L-Put", "L-Pall", "BStem", "L-Hipp", "L-Amyg",
    "L-Acc", "R-WM", "R-Ctx", "R-Vent", "R-Thal",
    "R-Caud", "R-Put", "R-Pall", "R-Hipp", "R-Amyg", "R-Acc",
]

AD_KEY_REGIONS = {"L-Hipp", "R-Hipp", "L-Amyg", "R-Amyg", "L-Vent", "R-Vent"}


def set_nature_style():
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── dataset ──────────────────────────────────────────────────────────────────

class ExternalDataset(Dataset):
    def __init__(self, npz_paths, labels=None):
        self.paths = npz_paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        image = data["image"].astype(np.float32)
        seg = remap_segmentation(data["seg"].astype(np.int64))
        image = torch.from_numpy(image).unsqueeze(0)
        seg = torch.from_numpy(seg)
        label = self.labels[idx] if self.labels is not None else -1
        return image, seg, label, str(self.paths[idx].stem)


# ── model ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    model = create_model(
        in_channels=1, base_channels=32, feature_dim=128, num_heads=4,
        num_classes=3, num_regions=NUM_MAPPED_REGIONS,
        use_atlas_conditioning=True, dropout=0.3, num_attn_layers=2,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_with_attention(model, dataset, device, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_attn, all_probs, all_preds, all_labels = [], [], [], []

    for images, segs, labels, ids in loader:
        images = images.to(device)
        segs = segs.to(device)
        outputs = model(images, segmentation=segs, return_attention=True)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        attn = outputs.get("attention")
        if attn is not None:
            all_attn.append(attn.cpu().numpy())

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.extend(labels.numpy().tolist())

    result = {
        "probs": np.concatenate(all_probs),
        "preds": np.concatenate(all_preds),
        "labels": np.array(all_labels),
    }
    if all_attn:
        result["attention"] = np.concatenate(all_attn)
    return result


# ── attention analysis ───────────────────────────────────────────────────────

def compute_region_attention(attn):
    """attn: (N, heads, R, R) -> (N, R) received attention."""
    attn_mean = attn.mean(axis=1)
    return attn_mean.sum(axis=1)


def attention_profile(attn, labels=None):
    """Compute per-class mean attention profile."""
    received = compute_region_attention(attn)
    if labels is None:
        return {"all": received.mean(axis=0)}
    profiles = {}
    for c, cname in enumerate(CLASS_NAMES):
        mask = labels == c
        if mask.sum() > 0:
            profiles[cname] = received[mask].mean(axis=0)
    return profiles


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def rank_correlation(a, b):
    from scipy.stats import spearmanr
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


# ── load ADNI attention from results ─────────────────────────────────────────

def load_adni_attention_profiles(results_dir):
    """Load ADNI attention profiles from biomarker results."""
    bio_path = Path("chapter1_foundation/attention_biomarker_results.json")
    if bio_path.exists():
        with open(bio_path) as f:
            bio = json.load(f)
        return {k: np.array(v) for k, v in bio.get("per_class_mean_attention", {}).items()}

    from chapter1_foundation.attention_biomarker_analysis import load_attention_data, compute_region_attention_received
    attn, labels = load_attention_data(results_dir, "Ours (Atlas+AnatDist)")
    if attn is None:
        return {}
    received = compute_region_attention_received(attn)
    return {CLASS_NAMES[c]: received[labels == c].mean(axis=0) for c in range(3) if (labels == c).sum() > 0}


# ── find checkpoints ─────────────────────────────────────────────────────────

def find_best_checkpoints(results_dir, model_name="Ours (Atlas+AnatDist)", max_ckpts=6):
    """Find best model checkpoints (one per seed)."""
    ckpts = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        for pth in sorted(seed_dir.glob("best_model_*.pth")):
            if model_name.replace(" ", "_").replace("(", "").replace(")", "") in pth.stem or \
               "Atlas_AnatDist" in pth.stem or "atlas_anatdist" in pth.stem.lower():
                ckpts.append(pth)
                break
        else:
            for pth in sorted(seed_dir.glob("best_model_*.pth")):
                ckpts.append(pth)
                break
    return ckpts[:max_ckpts]


# ── figures ──────────────────────────────────────────────────────────────────

def fig_cross_dataset_attention(adni_profiles, ext_profiles, ext_name, save_dir):
    """Compare attention profiles between ADNI and external dataset."""
    set_nature_style()

    common_classes = [c for c in CLASS_NAMES if c in adni_profiles and c in ext_profiles]
    if not common_classes and "all" in ext_profiles:
        common_classes = ["CN"]
        if "CN" not in ext_profiles:
            ext_profiles["CN"] = ext_profiles["all"]

    n_classes = max(len(common_classes), 1)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for idx, cname in enumerate(common_classes):
        ax = axes[idx]
        adni_vals = adni_profiles.get(cname, adni_profiles.get("CN"))
        ext_vals = ext_profiles.get(cname, ext_profiles.get("all"))

        if adni_vals is None or ext_vals is None:
            continue

        x = np.arange(len(REGION_NAMES))
        width = 0.35
        ax.bar(x - width / 2, adni_vals, width, label=f"ADNI {cname}", color="#3B82C4", alpha=0.8)
        ax.bar(x + width / 2, ext_vals, width, label=f"{ext_name} {cname}", color="#C73737", alpha=0.8)

        cos_sim = cosine_similarity(adni_vals, ext_vals)
        rho, p = rank_correlation(adni_vals, ext_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        ax.set_xticks(x)
        ax.set_xticklabels(REGION_NAMES, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("Mean Attention Received")
        ax.set_title(f"{cname}: cos={cos_sim:.3f}, ρ={rho:.3f}{sig}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Attention Profile: ADNI vs {ext_name}", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, f"cross_dataset_{ext_name.lower()}")
    print(f"  Cross-dataset figure ({ext_name})")


def fig_consistency_summary(consistency_results, save_dir):
    """Summary of cross-dataset attention consistency."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    datasets = list(consistency_results.keys())
    if not datasets:
        plt.close(fig)
        return

    # Panel A: Cosine similarity
    ax = axes[0]
    for ds_name in datasets:
        cr = consistency_results[ds_name]
        classes = list(cr.get("cosine_similarity", {}).keys())
        cos_vals = [cr["cosine_similarity"][c] for c in classes]
        x = np.arange(len(classes))
        ax.bar(x, cos_vals, 0.6, label=ds_name, alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("a  Attention Profile Similarity", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="High consistency")
    ax.legend(fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: Spearman correlation
    ax = axes[1]
    for ds_name in datasets:
        cr = consistency_results[ds_name]
        classes = list(cr.get("spearman_rho", {}).keys())
        rho_vals = [cr["spearman_rho"][c] for c in classes]
        x = np.arange(len(classes))
        ax.bar(x, rho_vals, 0.6, label=ds_name, alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("b  Attention Rank Consistency", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.5, label="Strong correlation")
    ax.legend(fontsize=6)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Dataset Interpretability Generalization",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "consistency_summary")
    print("  Consistency summary figure")


def _save(fig, save_dir, name):
    fig.savefig(save_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_dir / f"{name}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="chapter1_foundation/experiment_results_v3")
    parser.add_argument("--data_root", type=str, default="sample_data")
    parser.add_argument("--output", type=str, default="chapter1_foundation/cross_dataset_interpretability_results.json")
    parser.add_argument("--figures", type=str, default="chapter1_foundation/figures_cross_dataset")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.data_root) / "cache_real"
    save_dir = Path(args.figures)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("CROSS-DATASET INTERPRETABILITY GENERALIZATION")
    print("=" * 70)

    # Load ADNI attention profiles
    print("\nLoading ADNI attention profiles ...")
    adni_profiles = load_adni_attention_profiles(results_dir)
    if not adni_profiles:
        print("No ADNI attention profiles found. Run attention_biomarker_analysis first.")
        sys.exit(1)
    print(f"  ADNI classes: {list(adni_profiles.keys())}")

    # Find checkpoints
    ckpts = find_best_checkpoints(results_dir)
    if not ckpts:
        print("No checkpoints found.")
        sys.exit(1)
    print(f"  Using {len(ckpts)} checkpoints")

    # Load first checkpoint for inference
    model = load_model(ckpts[0], device)
    print(f"  Loaded model from {ckpts[0].name}")

    consistency_results = {}

    # ── IXI Analysis ─────────────────────────────────────────────────────
    ixi_files = sorted(cache_dir.glob("IXI*.npz"))
    print(f"\n--- IXI Analysis ({len(ixi_files)} files) ---")

    if ixi_files:
        ixi_dataset = ExternalDataset(ixi_files, labels=[0] * len(ixi_files))
        ixi_result = predict_with_attention(model, ixi_dataset, device, args.batch_size)

        if "attention" in ixi_result:
            ixi_profiles = attention_profile(ixi_result["attention"])
            ixi_profiles["CN"] = ixi_profiles.pop("all", ixi_profiles.get("CN"))

            cos_cn = cosine_similarity(adni_profiles["CN"], ixi_profiles["CN"])
            rho_cn, p_cn = rank_correlation(adni_profiles["CN"], ixi_profiles["CN"])

            print(f"  IXI CN vs ADNI CN: cosine={cos_cn:.3f}, Spearman ρ={rho_cn:.3f} (p={p_cn:.4f})")

            consistency_results["IXI"] = {
                "n_subjects": len(ixi_files),
                "cosine_similarity": {"CN": cos_cn},
                "spearman_rho": {"CN": rho_cn},
                "spearman_p": {"CN": p_cn},
                "ixi_profile": ixi_profiles["CN"].tolist(),
                "adni_cn_profile": adni_profiles["CN"].tolist(),
            }

            fig_cross_dataset_attention(adni_profiles, ixi_profiles, "IXI", save_dir)

            # CN specificity
            cn_correct = (ixi_result["preds"] == 0).sum()
            print(f"  CN specificity: {cn_correct}/{len(ixi_files)} = {cn_correct/len(ixi_files):.3f}")
        else:
            print("  No attention maps returned (model may not support return_attention)")

    # ── OASIS Analysis ───────────────────────────────────────────────────
    oasis_files = sorted(cache_dir.glob("OASIS*.npz"))
    print(f"\n--- OASIS Analysis ({len(oasis_files)} files) ---")

    if len(oasis_files) > 5:
        oasis_labels = []
        oasis_paths = []
        for f in oasis_files:
            data = np.load(f, allow_pickle=True)
            oasis_labels.append(int(data["label"]))
            oasis_paths.append(f)

        label_dist = {CLASS_NAMES[i]: sum(1 for l in oasis_labels if l == i) for i in range(3)}
        print(f"  Label distribution: {label_dist}")

        oasis_dataset = ExternalDataset(oasis_paths, labels=oasis_labels)
        oasis_result = predict_with_attention(model, oasis_dataset, device, args.batch_size)

        if "attention" in oasis_result:
            oasis_labels_arr = np.array(oasis_labels)
            oasis_profiles = attention_profile(oasis_result["attention"], oasis_labels_arr)

            cos_by_class = {}
            rho_by_class = {}
            p_by_class = {}
            for cname in CLASS_NAMES:
                if cname in oasis_profiles and cname in adni_profiles:
                    cos_by_class[cname] = cosine_similarity(adni_profiles[cname], oasis_profiles[cname])
                    r, p = rank_correlation(adni_profiles[cname], oasis_profiles[cname])
                    rho_by_class[cname] = r
                    p_by_class[cname] = p
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  OASIS {cname} vs ADNI {cname}: cos={cos_by_class[cname]:.3f}, ρ={r:.3f} {sig}")

            consistency_results["OASIS"] = {
                "n_subjects": len(oasis_files),
                "label_distribution": label_dist,
                "cosine_similarity": cos_by_class,
                "spearman_rho": rho_by_class,
                "spearman_p": p_by_class,
                "oasis_profiles": {k: v.tolist() for k, v in oasis_profiles.items()},
                "adni_profiles": {k: v.tolist() for k, v in adni_profiles.items()},
            }

            fig_cross_dataset_attention(adni_profiles, oasis_profiles, "OASIS", save_dir)

            # Classification metrics
            valid = oasis_labels_arr >= 0
            preds = oasis_result["preds"][valid]
            labels_v = oasis_labels_arr[valid]
            acc = float((preds == labels_v).mean())
            recalls = [float((preds[labels_v == c] == c).mean()) for c in range(3) if (labels_v == c).sum() > 0]
            bacc = float(np.mean(recalls))
            print(f"  OASIS classification: Acc={acc:.3f}, BAcc={bacc:.3f}")
    else:
        print("  Not enough OASIS files for analysis (need preprocessing)")

    # Summary figure
    if consistency_results:
        fig_consistency_summary(consistency_results, save_dir)

    # Save results
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(consistency_results, f, indent=2, cls=_Encoder)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
