#!/usr/bin/env python3
"""
Recover 'Ours (Atlas+AnatDist)' results from saved .pth checkpoints.

Loads best_model_seed{S}_fold{F}.pth, rebuilds the exact same fold splits,
runs inference on the test set, and writes all_results_partial.json so that
the main run_experiment.py can skip already-completed configs on restart.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chapter1_foundation.data.foundation_loader import (
    RealCachedDataset, kfold_split,
)
from chapter1_foundation.models import create_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob, all_feat = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        seg = batch.get("segmentation")
        seg = seg.to(device) if seg is not None else None

        outputs = model(image, segmentation=seg, return_features=True)
        logits = outputs["logits"]
        total_loss += F.cross_entropy(logits, label).item()
        probs = F.softmax(logits, dim=-1)
        all_true.extend(label.cpu().tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_prob.append(probs.cpu().numpy())
        all_feat.append(outputs["pooled"].cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.concatenate(all_prob) if all_prob else np.zeros((0, 3))
    features = np.concatenate(all_feat) if all_feat else np.zeros((0,))
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0

    per_class_acc = {}
    for c in range(3):
        mask = y_true == c
        per_class_acc[c] = float((y_pred[mask] == c).mean()) if mask.sum() > 0 else 0.0

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": acc,
        "balanced_acc": np.mean(list(per_class_acc.values())),
        "per_class_acc": per_class_acc,
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "features": features,
    }


@torch.no_grad()
def collect_attention(model, loader, device, max_samples=50):
    model.eval()
    attn_list, label_list, feat_list = [], [], []
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
        image = batch["image"].to(device)
        seg = batch.get("segmentation")
        seg = seg.to(device) if seg is not None else None
        outputs = model(image, segmentation=seg, return_attention=True, return_features=True)
        if outputs.get("attention") is not None:
            for b in range(image.shape[0]):
                if count >= max_samples:
                    break
                attn_list.append(outputs["attention"][b].cpu().numpy())
                label_list.append(batch["label"][b].item())
                feat_list.append(outputs["pooled"][b].cpu().numpy())
                count += 1
    return attn_list, label_list, feat_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="sample_data")
    parser.add_argument("--results_base", type=str,
                        default="chapter1_foundation/experiment_results_ssl")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 153, 264, 375, 486, 597])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.data_root) / "cache_real"
    results_base = Path(args.results_base)

    npz_files = sorted(cache_dir.glob("*.npz"))
    all_labels_full = np.array([int(np.load(f, allow_pickle=True)["label"]) for f in npz_files])
    all_stems = np.array([f.stem for f in npz_files])
    all_sources = np.array([
        "ADNI" if s.startswith("ADNI_") else
        "SYNTH" if s.startswith("SYNTH_") else
        "OASIS" if s.startswith("OASIS_") else
        "IXI" for s in all_stems
    ])

    adni_mask = all_sources == "ADNI"
    adni_indices = np.where(adni_mask)[0]
    adni_labels = all_labels_full[adni_mask]

    print(f"ADNI samples: {len(adni_indices)}", flush=True)
    print(f"Device: {device}", flush=True)

    for seed in args.seeds:
        seed_dir = results_base / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"\n=== seed_{seed}: directory not found, skipping ===", flush=True)
            continue

        ckpts = sorted(seed_dir.glob(f"best_model_seed{seed}_fold*.pth"))
        if len(ckpts) < args.n_folds:
            print(f"\n=== seed_{seed}: only {len(ckpts)}/{args.n_folds} checkpoints, skipping ===",
                  flush=True)
            continue

        partial_path = seed_dir / "all_results_partial.json"
        all_results = {}
        if partial_path.exists():
            with open(partial_path) as f:
                all_results = json.load(f)
            print(f"\n=== seed_{seed}: loaded {len(all_results)} existing results ===", flush=True)

        folds = kfold_split(adni_labels, n_folds=args.n_folds, seed=seed)

        for fold_i, (train_rel, val_rel, test_rel) in enumerate(folds):
            key = f"Ours (Atlas+AnatDist)__seed{seed}__fold{fold_i}"
            if key in all_results and "error" not in all_results[key]:
                print(f"  fold {fold_i}: already recovered, skipping", flush=True)
                continue

            ckpt_path = seed_dir / f"best_model_seed{seed}_fold{fold_i}.pth"
            if not ckpt_path.exists():
                print(f"  fold {fold_i}: checkpoint not found", flush=True)
                continue

            test_abs = adni_indices[test_rel]
            val_abs = adni_indices[val_rel]

            test_ds = RealCachedDataset(cache_dir, indices=test_abs, seed=seed)
            val_ds = RealCachedDataset(cache_dir, indices=val_abs, seed=seed)
            test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                                     num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                    num_workers=2, pin_memory=True)

            model = create_model(
                num_classes=3, in_channels=1, base_channels=32,
                num_regions=21, dropout=args.dropout,
            ).to(device)

            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            n_params = sum(p.numel() for p in model.parameters())

            t0 = time.time()
            test_res = evaluate(model, test_loader, device)
            val_res = evaluate(model, val_loader, device)
            attn_data, attn_labels, _ = collect_attention(model, test_loader, device)
            dt = time.time() - t0

            pca = test_res["per_class_acc"]
            print(f"  fold {fold_i} ({dt:.1f}s): test_acc={test_res['acc']:.4f} "
                  f"test_bacc={test_res['balanced_acc']:.4f} "
                  f"val_bacc={val_res['balanced_acc']:.4f} "
                  f"[CN={pca[0]:.2f} MCI={pca[1]:.2f} AD={pca[2]:.2f}]", flush=True)

            all_results[key] = {
                "model_name": "ours",
                "use_atlas": True,
                "use_anat_dist": True,
                "n_params": n_params,
                "best_val_bacc": val_res["balanced_acc"],
                "test_acc": test_res["acc"],
                "test_balanced_acc": test_res["balanced_acc"],
                "test_per_class_acc": test_res["per_class_acc"],
                "test_y_true": test_res["y_true"].tolist(),
                "test_y_pred": test_res["y_pred"].tolist(),
                "test_y_prob": test_res["y_prob"].tolist(),
                "test_features": test_res["features"].tolist(),
                "history": [],
                "seed": seed,
                "fold": fold_i,
                "config_name": "Ours (Atlas+AnatDist)",
                "attention_maps": [a.tolist() for a in attn_data] if attn_data else [],
                "attention_labels": attn_labels,
                "recovered_from_checkpoint": True,
            }

            del model
            torch.cuda.empty_cache()

        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  => Saved {len(all_results)} results to {partial_path}", flush=True)

    print("\nDone! Partial results written. Re-run run_experiment.py to skip recovered configs.",
          flush=True)


if __name__ == "__main__":
    main()
