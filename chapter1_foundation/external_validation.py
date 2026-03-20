#!/usr/bin/env python3
"""
External validation: evaluate trained ARA-Net on IXI (CN specificity)
and OASIS (independent 3-class test).

Usage:
    python -m chapter1_foundation.external_validation \
        --results_dir chapter1_foundation/experiment_results_v3 \
        --data_root sample_data \
        --output chapter1_foundation/external_validation_results.json
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

CLASS_NAMES = ["CN", "MCI", "AD"]
NUM_MAPPED_REGIONS = 21


# ── dataset for external validation ──────────────────────────────────────────

class ExternalDataset(Dataset):
    """Load .npz files for external validation (IXI or OASIS)."""

    def __init__(self, npz_paths: List[Path], labels: Optional[List[int]] = None):
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


# ── model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
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


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_dataset(model: torch.nn.Module, dataset: Dataset,
                    device: torch.device, batch_size: int = 4) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_probs, all_preds, all_labels, all_ids = [], [], [], []

    for images, segs, labels, ids in loader:
        images = images.to(device)
        segs = segs.to(device)
        outputs = model(images, segmentation=segs, return_attention=False)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.extend(labels.numpy().tolist())
        all_ids.extend(ids)

    return {
        "probs": np.concatenate(all_probs),
        "preds": np.concatenate(all_preds),
        "labels": np.array(all_labels),
        "ids": all_ids,
    }


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_ixi_metrics(results: dict) -> dict:
    """IXI: all subjects are CN (label=0). Measure CN specificity."""
    preds = results["preds"]
    probs = results["probs"]
    n = len(preds)
    cn_correct = (preds == 0).sum()
    cn_specificity = float(cn_correct / n)

    cn_prob_mean = float(probs[:, 0].mean())
    cn_prob_std = float(probs[:, 0].std())

    pred_dist = {CLASS_NAMES[i]: int((preds == i).sum()) for i in range(3)}

    return {
        "n_subjects": n,
        "cn_specificity": cn_specificity,
        "cn_prob_mean": cn_prob_mean,
        "cn_prob_std": cn_prob_std,
        "prediction_distribution": pred_dist,
    }


def compute_oasis_metrics(results: dict) -> dict:
    """OASIS: 3-class (CN/MCI/AD). Compute BAcc, Acc, per-class recall, AUC."""
    preds = results["preds"]
    labels = results["labels"]
    probs = results["probs"]

    valid = labels >= 0
    preds = preds[valid]
    labels = labels[valid]
    probs = probs[valid]
    n = len(labels)

    acc = float((preds == labels).mean())
    recalls = []
    per_class = {}
    for i, c in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            rec = float((preds[mask] == i).mean())
        else:
            rec = 0.0
        recalls.append(rec)
        per_class[c] = {"recall": rec, "n": int(mask.sum())}
    bacc = float(np.mean(recalls))

    aucs = []
    for i in range(3):
        binary = (labels == i).astype(int)
        scores = probs[:, i]
        if binary.sum() == 0 or binary.sum() == len(binary):
            aucs.append(0.5)
            continue
        order = np.argsort(-scores)
        ys = binary[order]
        n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
        tp = fp = auc = 0.0
        tpr_prev = fpr_prev = 0.0
        prev = -np.inf
        for j in range(len(ys)):
            if scores[order[j]] != prev:
                tpr, fpr = tp / n_pos, fp / n_neg
                auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
                tpr_prev, fpr_prev = tpr, fpr
                prev = scores[order[j]]
        tpr, fpr = tp / n_pos, fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        aucs.append(auc)
        per_class[CLASS_NAMES[i]]["auc"] = auc

    macro_auc = float(np.mean(aucs))

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    return {
        "n_subjects": n,
        "acc": acc,
        "bacc": bacc,
        "macro_auc": macro_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ── ensemble prediction across seeds ─────────────────────────────────────────

def ensemble_predict(models: List[torch.nn.Module], dataset: Dataset,
                     device: torch.device, batch_size: int = 4) -> dict:
    """Average predictions across multiple model checkpoints."""
    all_results = []
    for model in models:
        res = predict_dataset(model, dataset, device, batch_size)
        all_results.append(res)

    avg_probs = np.mean([r["probs"] for r in all_results], axis=0)
    avg_preds = avg_probs.argmax(axis=1)

    return {
        "probs": avg_probs,
        "preds": avg_preds,
        "labels": all_results[0]["labels"],
        "ids": all_results[0]["ids"],
    }


# ── data discovery ───────────────────────────────────────────────────────────

def find_ixi_files(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.glob("IXI*.npz"))


def find_oasis_files(cache_dir: Path, exclude_stems: set = None) -> Tuple[List[Path], List[int]]:
    """Find OASIS .npz files and their labels. Excludes training set stems."""
    exclude_stems = exclude_stems or set()
    paths, labels = [], []
    for f in sorted(cache_dir.glob("OASIS_*.npz")):
        if f.stem in exclude_stems:
            continue
        data = np.load(f, allow_pickle=True)
        paths.append(f)
        labels.append(int(data["label"]))
    return paths, labels


def find_checkpoints(results_dir: Path) -> Dict[str, List[Path]]:
    """Find best_model checkpoints grouped by seed_fold."""
    ckpts = defaultdict(list)
    for seed_dir in sorted(results_dir.glob("seed_*")):
        for pth in sorted(seed_dir.glob("best_model_*.pth")):
            ckpts[seed_dir.name].append(pth)
    return dict(ckpts)


def get_training_oasis_stems(results_dir: Path) -> set:
    """Get OASIS stems used in training (to exclude from external validation)."""
    stems = set()
    for seed_dir in results_dir.glob("seed_*"):
        for fname in ["all_results.json", "all_results_partial.json"]:
            fpath = seed_dir / fname
            if fpath.exists():
                break
    cache_dir = Path("sample_data/cache_real")
    for f in cache_dir.glob("OASIS_*.npz"):
        stems.add(f.stem)
    return stems


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="External validation (IXI + OASIS)")
    parser.add_argument("--results_dir", type=str,
                        default="chapter1_foundation/experiment_results_v3")
    parser.add_argument("--data_root", type=str, default="sample_data")
    parser.add_argument("--output", type=str,
                        default="chapter1_foundation/external_validation_results.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.data_root) / "cache_real"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Results dir: {results_dir}")
    print(f"Cache dir: {cache_dir}")

    ckpt_groups = find_checkpoints(results_dir)
    print(f"\nFound checkpoints in {len(ckpt_groups)} seeds:")
    all_ckpts = []
    for seed_name, ckpts in ckpt_groups.items():
        print(f"  {seed_name}: {len(ckpts)} checkpoints")
        all_ckpts.extend(ckpts)

    if not all_ckpts:
        print("No checkpoints found. Run experiments first.")
        sys.exit(1)

    # ── IXI validation ───────────────────────────────────────────────────
    ixi_files = find_ixi_files(cache_dir)
    print(f"\nIXI files: {len(ixi_files)}")

    ixi_results = {"per_checkpoint": [], "ensemble": None}

    if ixi_files:
        ixi_dataset = ExternalDataset(ixi_files, labels=[0] * len(ixi_files))

        models = []
        for ckpt_path in all_ckpts:
            print(f"  Loading {ckpt_path.name} ...")
            model = load_model(ckpt_path, device)
            models.append(model)

            res = predict_dataset(model, ixi_dataset, device, args.batch_size)
            metrics = compute_ixi_metrics(res)
            metrics["checkpoint"] = str(ckpt_path)
            ixi_results["per_checkpoint"].append(metrics)
            print(f"    CN specificity: {metrics['cn_specificity']:.3f}  "
                  f"dist: {metrics['prediction_distribution']}")

        if len(models) > 1:
            print(f"\n  Ensemble ({len(models)} models) ...")
            ens_res = ensemble_predict(models, ixi_dataset, device, args.batch_size)
            ens_metrics = compute_ixi_metrics(ens_res)
            ixi_results["ensemble"] = ens_metrics
            print(f"    CN specificity (ensemble): {ens_metrics['cn_specificity']:.3f}")

        specs = [r["cn_specificity"] for r in ixi_results["per_checkpoint"]]
        ixi_results["summary"] = {
            "mean_specificity": float(np.mean(specs)),
            "std_specificity": float(np.std(specs)),
            "n_checkpoints": len(specs),
            "n_subjects": len(ixi_files),
        }
        print(f"\n  IXI Summary: {ixi_results['summary']['mean_specificity']:.3f} "
              f"± {ixi_results['summary']['std_specificity']:.3f}")

        del models
        torch.cuda.empty_cache()

    # ── OASIS validation ─────────────────────────────────────────────────
    training_oasis = get_training_oasis_stems(results_dir)
    oasis_files, oasis_labels = find_oasis_files(cache_dir, exclude_stems=training_oasis)
    print(f"\nOASIS files: {len(oasis_files)} (excluded {len(training_oasis)} training)")

    oasis_results = {"per_checkpoint": [], "ensemble": None}

    if oasis_files:
        oasis_dataset = ExternalDataset(oasis_files, labels=oasis_labels)
        label_dist = {CLASS_NAMES[i]: sum(1 for l in oasis_labels if l == i)
                      for i in range(3)}
        print(f"  Label distribution: {label_dist}")

        models = []
        for ckpt_path in all_ckpts:
            print(f"  Loading {ckpt_path.name} ...")
            model = load_model(ckpt_path, device)
            models.append(model)

            res = predict_dataset(model, oasis_dataset, device, args.batch_size)
            metrics = compute_oasis_metrics(res)
            metrics["checkpoint"] = str(ckpt_path)
            oasis_results["per_checkpoint"].append(metrics)
            print(f"    BAcc: {metrics['bacc']:.3f}  Acc: {metrics['acc']:.3f}  "
                  f"AUC: {metrics['macro_auc']:.3f}")

        if len(models) > 1:
            print(f"\n  Ensemble ({len(models)} models) ...")
            ens_res = ensemble_predict(models, oasis_dataset, device, args.batch_size)
            ens_metrics = compute_oasis_metrics(ens_res)
            oasis_results["ensemble"] = ens_metrics
            print(f"    BAcc (ensemble): {ens_metrics['bacc']:.3f}  "
                  f"AUC: {ens_metrics['macro_auc']:.3f}")

        baccs = [r["bacc"] for r in oasis_results["per_checkpoint"]]
        oasis_results["summary"] = {
            "mean_bacc": float(np.mean(baccs)),
            "std_bacc": float(np.std(baccs)),
            "n_checkpoints": len(baccs),
            "n_subjects": len(oasis_files),
        }
        print(f"\n  OASIS Summary: BAcc {oasis_results['summary']['mean_bacc']:.3f} "
              f"± {oasis_results['summary']['std_bacc']:.3f}")

    # ── save ─────────────────────────────────────────────────────────────
    output = {
        "ixi": ixi_results,
        "oasis": oasis_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
