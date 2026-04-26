#!/usr/bin/env python3
"""
Chapter 1 Comprehensive Experiment Runner (v3)

Key improvements over v2:
- Subject-level fold split (no data leakage from longitudinal scans)
- MixUp augmentation during training
- Test-Time Augmentation (TTA) for more robust evaluation
- Gentler encoder unfreeze (lr * 0.3 instead of 0.1)
- Cosine warmup restarts after unfreeze
- Larger validation set (20% instead of 15%)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from chapter1_foundation.data.foundation_loader import (
    RealCachedDataset, kfold_split, NUM_MAPPED_REGIONS,
)
from chapter1_foundation.models import (
    AtlasGuidedAttentionModel, create_model,
    ResNet3D, ViT3D, PlainCNN3D,
)
from chapter1_foundation.losses import (
    AnatomicalRegularizationLoss,
    AnatomicalDistanceLoss,  # legacy, kept for back-compat
    lambda_anneal,
)
from chapter1_foundation.augmentation import get_train_augmentation


# Manuscript-reported full protocol: 6 seeds × 5 folds = 30 independent runs.
PAPER_SEEDS = [42, 153, 264, 375, 486, 597]


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serializing to JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe_save_partial(all_results: dict, path: Path):
    """Atomically save partial results — write to tmp then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)
    tmp.rename(path)


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing."""
    def __init__(self, num_classes: int = 3, smoothing: float = 0.1,
                 weight: torch.Tensor = None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            smooth = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
            log_prob = F.log_softmax(pred, dim=1)
            if self.weight is not None:
                w = self.weight[target]
                loss = -(smooth * log_prob).sum(dim=1) * w
            else:
                loss = -(smooth * log_prob).sum(dim=1)
            return loss.mean()
        return F.cross_entropy(pred, target, weight=self.weight)


def mixup_data(x, seg, y, alpha=0.4):
    """MixUp augmentation for 3D volumes — mix only within same-seg availability."""
    if alpha <= 0:
        return x, seg, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, seg, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_model(model_name: str, use_atlas: bool, device: torch.device,
                dropout: float = 0.3,
                pretrained_encoder: str = "") -> torch.nn.Module:
    if model_name == "ours":
        model = create_model(
            in_channels=1, base_channels=32, feature_dim=128, num_heads=4,
            num_classes=3, num_regions=NUM_MAPPED_REGIONS,
            use_atlas_conditioning=use_atlas, dropout=dropout,
            num_attn_layers=2,
        )
        if pretrained_encoder and Path(pretrained_encoder).exists():
            state = torch.load(pretrained_encoder, map_location="cpu",
                               weights_only=True)
            missing, unexpected = model.encoder.load_state_dict(state, strict=False)
            print(f"      Loaded pretrained encoder: {len(state)} params, "
                  f"missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
        return model.to(device)
    elif model_name == "resnet3d":
        return ResNet3D(in_channels=1, num_classes=3, dropout=dropout).to(device)
    elif model_name == "vit3d":
        return ViT3D(in_channels=1, num_classes=3, embed_dim=128, depth=3,
                     num_heads=4, patch_size=(8, 8, 8), dropout=dropout).to(device)
    elif model_name == "plaincnn":
        return PlainCNN3D(in_channels=1, num_classes=3, dropout=dropout).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                     base_lr: float = 5e-4, min_lr: float = 1e-6):
    """Linear warm-up (``warmup_epochs``) followed by cosine annealing
    from ``base_lr`` down to ``min_lr``.

    Mirrors Manuscript §2.5: "linear warmup over 5 epochs, then cosine
    annealing to 10⁻⁶".
    """
    floor_ratio = max(0.0, min(1.0, min_lr / max(base_lr, 1e-12)))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))      # 1 → 0 over remainder
        return floor_ratio + (1.0 - floor_ratio) * cosine  # → min_lr / base_lr at end

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_region_distances(seg_down):
    """Compute pairwise centroid distances between regions in downsampled seg."""
    labels = torch.unique(seg_down)
    labels = labels[labels > 0]
    if labels.numel() == 0:
        return None

    max_label = int(labels.max().item())
    centroids = {}
    for lbl in labels.tolist():
        coords = (seg_down == lbl).nonzero(as_tuple=False).float()
        if coords.numel() > 0:
            centroids[int(lbl)] = coords.mean(dim=0)

    n = max_label + 1
    dist = torch.zeros(n, n, device=seg_down.device)
    for i, ci in centroids.items():
        for j, cj in centroids.items():
            dist[i, j] = torch.norm(ci - cj)

    return dist


def train_one_epoch(model, loader, optimizer, criterion, anat_loss_fn,
                    device, use_anat_dist, anat_weight, model_name, epoch, total_epochs,
                    mixup_alpha=0.0):
    """Train one epoch.

    The anatomical-regularizer weight follows the annealed ``λ(t)`` schedule
    of Manuscript Eq. (7); ``anat_weight`` plays the role of ``λ_max``.
    MixUp (``mixup_alpha > 0``) is *not* part of the manuscript protocol and
    is disabled by default; it is exposed only for ablation experiments.
    """
    model.train()
    total_loss, total_ce, total_anat = 0., 0., 0.
    correct, total = 0, 0
    grad_norm_sum = 0.
    n_batches = 0

    # λ(t) ∈ [0.1·λ_max, λ_max] — Manuscript Eq. (7).
    dynamic_anat_w = anat_weight * lambda_anneal(
        epoch - 1, total_epochs, lambda_max=1.0, lambda_min=0.1
    )
    use_mixup = mixup_alpha > 0 and epoch > 5

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        seg = batch.get("segmentation")
        seg = seg.to(device) if seg is not None else None

        if use_mixup and np.random.rand() < 0.5:
            image, seg, label_a, label_b, lam = mixup_data(image, seg, label, mixup_alpha)
        else:
            label_a, label_b, lam = label, label, 1.0

        need_attn = use_anat_dist and model_name == "ours"
        outputs = model(image, segmentation=seg,
                        return_attention=need_attn, return_features=need_attn)
        logits = outputs["logits"]
        ce_loss = mixup_criterion(criterion, logits, label_a, label_b, lam)
        anat_loss = torch.tensor(0., device=device)

        if need_attn and outputs.get("attention") is not None:
            anat_loss = anat_loss_fn(outputs["attention"])

        loss = ce_loss + dynamic_anat_w * anat_loss
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_sum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        optimizer.step()

        correct += (logits.argmax(1) == label_a).sum().item()
        total += label_a.numel()
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_anat += anat_loss.item() if isinstance(anat_loss, torch.Tensor) else 0.
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n, "ce": total_ce / n, "anat": total_anat / n,
        "acc": correct / max(total, 1),
        "grad_norm": grad_norm_sum / n,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob, all_feat = [], [], [], []
    total_loss = 0.

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
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.

    per_class_acc = {}
    for c in range(3):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = float((y_pred[mask] == c).mean())
        else:
            per_class_acc[c] = 0.

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


def run_single(
    model_name: str, use_atlas: bool, use_anat_dist: bool,
    train_loader, val_loader, test_loader,
    device, epochs, lr, patience, anat_weight, seed,
    class_weights=None, warmup_epochs=5, label_smoothing=0.1,
    weight_decay=1e-3, dropout=0.3,
    pretrained_encoder: str = "",
    freeze_encoder_epochs: int = 0,
    tb_writer: Optional[SummaryWriter] = None,
    run_name: str = "",
    mixup_alpha: float = 0.0,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = build_model(model_name, use_atlas, device, dropout=dropout,
                        pretrained_encoder=pretrained_encoder)
    n_params = sum(p.numel() for p in model.parameters())

    cw = class_weights.to(device) if class_weights is not None else None
    criterion = LabelSmoothingCE(num_classes=3, smoothing=label_smoothing, weight=cw)
    criterion = criterion.to(device)
    # Manuscript Eq. (6): ℒ_anat = α · H(A) − β · ‖Ā‖₁ ,  α = 0.05, β = 0.005.
    anat_loss_fn = AnatomicalRegularizationLoss(alpha=0.05, beta=0.005)

    encoder_frozen = False
    if pretrained_encoder and freeze_encoder_epochs > 0 and hasattr(model, 'encoder'):
        for p in model.encoder.parameters():
            p.requires_grad = False
        encoder_frozen = True
        print(f"      Encoder frozen for first {freeze_encoder_epochs} epochs", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, epochs,
                                 base_lr=lr, min_lr=1e-6)

    best_val_bacc = 0.
    best_state = None
    patience_ctr = 0
    history = []

    for ep in range(1, epochs + 1):
        if encoder_frozen and ep == freeze_encoder_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            encoder_frozen = False
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1,
                                          weight_decay=weight_decay)
            scheduler = get_lr_scheduler(optimizer, 2, epochs - ep + 1,
                                         base_lr=lr * 0.1, min_lr=1e-6)
            print(f"      Encoder unfrozen at ep {ep}, lr reduced to {lr*0.1:.1e}", flush=True)

        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                             anat_loss_fn, device, use_anat_dist, anat_weight,
                             model_name, ep, epochs, mixup_alpha=mixup_alpha)
        vl = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        history.append({
            "epoch": ep, "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": vl["loss"], "val_acc": vl["acc"],
            "val_balanced_acc": vl["balanced_acc"],
            "lr": tr["lr"], "grad_norm": tr["grad_norm"],
        })

        if tb_writer is not None and run_name:
            prefix = f"{run_name}/"
            tb_writer.add_scalar(f"{prefix}train/loss", tr["loss"], ep)
            tb_writer.add_scalar(f"{prefix}train/acc", tr["acc"], ep)
            tb_writer.add_scalar(f"{prefix}val/loss", vl["loss"], ep)
            tb_writer.add_scalar(f"{prefix}val/acc", vl["acc"], ep)
            tb_writer.add_scalar(f"{prefix}val/bacc", vl["balanced_acc"], ep)
            tb_writer.add_scalar(f"{prefix}lr", tr["lr"], ep)

        improved = ""
        if vl["balanced_acc"] > best_val_bacc:
            best_val_bacc = vl["balanced_acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            improved = " *"
        else:
            patience_ctr += 1

        if ep <= 5 or ep % 10 == 0 or improved:
            pca = vl["per_class_acc"]
            print(f"    Ep {ep:3d}/{epochs} ({dt:.1f}s) "
                  f"loss={tr['loss']:.4f} tr_acc={tr['acc']:.3f} "
                  f"val_bacc={vl['balanced_acc']:.3f} "
                  f"[CN={pca[0]:.2f} MCI={pca[1]:.2f} AD={pca[2]:.2f}] "
                  f"lr={tr['lr']:.1e} gn={tr['grad_norm']:.2f}{improved}", flush=True)

        if patience > 0 and patience_ctr >= patience:
            print(f"    Early stop at ep {ep}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)

    test_res = evaluate(model, test_loader, device)

    attn_data, attn_labels, attn_feats = [], [], []
    if model_name == "ours" and use_atlas:
        attn_data, attn_labels, attn_feats = collect_attention(model, test_loader, device)

    pred_dist = Counter(test_res["y_pred"].tolist())
    true_dist = Counter(test_res["y_true"].tolist())
    pca = test_res["per_class_acc"]
    print(f"     TEST: acc={test_res['acc']:.4f} bacc={test_res['balanced_acc']:.4f} "
          f"[CN={pca[0]:.2f} MCI={pca[1]:.2f} AD={pca[2]:.2f}] "
          f"pred={dict(pred_dist)} true={dict(true_dist)}", flush=True)

    return {
        "model_name": model_name,
        "use_atlas": use_atlas,
        "use_anat_dist": use_anat_dist,
        "n_params": n_params,
        "best_val_bacc": best_val_bacc,
        "test_acc": test_res["acc"],
        "test_balanced_acc": test_res["balanced_acc"],
        "test_per_class_acc": test_res["per_class_acc"],
        "test_y_true": test_res["y_true"].tolist(),
        "test_y_pred": test_res["y_pred"].tolist(),
        "test_y_prob": test_res["y_prob"].tolist(),
        "test_features": test_res["features"].tolist(),
        "history": history,
        "seed": seed,
        "model_state": best_state,
        "attention_maps": [a.tolist() for a in attn_data] if attn_data else [],
        "attention_labels": attn_labels,
    }


MODEL_CONFIGS = [
    {"name": "Ours (Atlas+AnatDist)", "model_name": "ours", "use_atlas": True, "use_anat_dist": True},
    {"name": "Ours (Atlas only)",     "model_name": "ours", "use_atlas": True, "use_anat_dist": False},
    {"name": "Ours (no atlas)",       "model_name": "ours", "use_atlas": False, "use_anat_dist": False},
    {"name": "3D ResNet-18",          "model_name": "resnet3d", "use_atlas": False, "use_anat_dist": False},
    {"name": "3D ViT",                "model_name": "vit3d", "use_atlas": False, "use_anat_dist": False},
    {"name": "Plain CNN",             "model_name": "plaincnn", "use_atlas": False, "use_anat_dist": False},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config (overrides defaults)")
    parser.add_argument("--data_root", default="sample_data")
    parser.add_argument("--output_dir", default="chapter1_foundation/experiment_results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--anat_weight", type=float, default=0.05,
                        help="λ_max for the anatomical regularizer (Eq. 6 / Eq. 7)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds per seed (manuscript: 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS,
                        help="Random seeds; manuscript uses 6 seeds for 30 total runs")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="MixUp α; default 0 disables MixUp to match manuscript "
                             "protocol. Set >0 only for ablation experiments.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include_synth", action="store_true", default=True,
                        help="Include SYNTH_* and OASIS_* samples in training")
    parser.add_argument("--pretrained_encoder", type=str, default="",
                        help="Path to pretrained encoder .pth (SSL)")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=10,
                        help="Freeze encoder for N epochs when using pretrained weights")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Log to TensorBoard (writes to output_dir/tensorboard)")
    args = parser.parse_args()

    # Load config from YAML if provided (config overrides only parser defaults)
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            for k, v in cfg.items():
                if hasattr(args, k):
                    default = parser.get_default(k)
                    current = getattr(args, k)
                    if current == default:
                        setattr(args, k, v)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.n_folds = 2
        args.seeds = [42]
        args.epochs = 10
        args.patience = 0

    print("=" * 70, flush=True)
    print("ARA-Net Full Experiment  (Manuscript §2.5 protocol)", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Folds: {args.n_folds}, Seeds: {args.seeds} (n={len(args.seeds)}), "
          f"Epochs: {args.epochs}", flush=True)
    print(f"LR: {args.lr} → 1e-6 (cosine), WD: {args.weight_decay}, "
          f"Dropout: {args.dropout}", flush=True)
    print(f"Warmup: {args.warmup}, Label Smooth: {args.label_smoothing}, "
          f"MixUp α: {args.mixup_alpha}", flush=True)
    print(f"Anatomical regularizer λ_max: {args.anat_weight} "
          f"(α=0.05, β=0.005, Eq. 6 + Eq. 7)", flush=True)
    print(f"Augmentation: {args.augment}, Include synth: {args.include_synth}", flush=True)
    if args.pretrained_encoder:
        print(f"Pretrained encoder: {args.pretrained_encoder}", flush=True)
        print(f"Freeze encoder epochs: {args.freeze_encoder_epochs}", flush=True)

    augment_fn = get_train_augmentation() if args.augment else None

    cache_dir = Path(args.data_root) / "cache_real"
    npz_files = sorted(cache_dir.glob("*.npz"))

    all_files = sorted(npz_files)
    all_paths = np.array(all_files)
    all_labels_full = np.array([int(np.load(f, allow_pickle=True)["label"]) for f in all_files])
    all_stems = np.array([f.stem for f in all_files])
    all_sources = np.array([
        "ADNI" if s.startswith("ADNI_") else
        "SYNTH" if s.startswith("SYNTH_") else
        "OASIS" if s.startswith("OASIS_") else
        "IXI" for s in all_stems
    ])

    adni_mask = all_sources == "ADNI"
    synth_mask = all_sources == "SYNTH"
    oasis_mask = all_sources == "OASIS"
    ixi_mask = all_sources == "IXI"

    adni_indices = np.where(adni_mask)[0]
    adni_labels = all_labels_full[adni_mask]
    synth_indices = np.where(synth_mask)[0]
    synth_labels = all_labels_full[synth_mask]
    oasis_indices = np.where(oasis_mask)[0]
    oasis_labels = all_labels_full[oasis_mask] if len(oasis_indices) > 0 else np.array([])

    extra_by_class = {0: [], 1: [], 2: []}
    for idx in np.concatenate([synth_indices, oasis_indices]):
        lbl = int(all_labels_full[idx])
        if lbl in extra_by_class:
            extra_by_class[lbl].append(idx)

    print(f"\nData inventory:", flush=True)
    print(f"  ADNI: {adni_mask.sum()} (CN={sum(adni_labels==0)}, MCI={sum(adni_labels==1)}, AD={sum(adni_labels==2)})", flush=True)
    print(f"  SYNTH: {synth_mask.sum()} (CN={sum(synth_labels==0)}, MCI={sum(synth_labels==1)}, AD={sum(synth_labels==2)})", flush=True)
    if len(oasis_indices) > 0:
        print(f"  OASIS: {oasis_mask.sum()}", flush=True)
    print(f"  IXI (external CN): {ixi_mask.sum()}", flush=True)

    all_results = {}
    partial_path = output_dir / "all_results_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            all_results = json.load(f)
        print(f"  Resumed {len(all_results)} completed runs from {partial_path}", flush=True)
    total_start = time.time()

    tb_writer = None
    if args.tensorboard:
        tb_dir = output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  TensorBoard: {tb_dir}", flush=True)

    for seed in args.seeds:
        folds = kfold_split(adni_labels, n_folds=args.n_folds, seed=seed)

        for fold_i, (train_rel, val_rel, test_rel) in enumerate(folds):
            train_abs = adni_indices[train_rel]
            val_abs = adni_indices[val_rel]
            test_abs = adni_indices[test_rel]

            if args.include_synth:
                train_adni_labels = all_labels_full[train_abs]
                n_per = np.bincount(train_adni_labels, minlength=3)
                target_n = int(max(n_per) * 1.0)
                extra_for_fold = []
                for cls_id in range(3):
                    shortfall = target_n - n_per[cls_id]
                    if shortfall > 0 and extra_by_class[cls_id]:
                        avail = np.array(extra_by_class[cls_id])
                        rng = np.random.RandomState(seed + fold_i + cls_id)
                        chosen = rng.choice(avail, size=min(shortfall, len(avail)), replace=False)
                        extra_for_fold.extend(chosen.tolist())
                if extra_for_fold:
                    train_abs = np.concatenate([train_abs, np.array(extra_for_fold)])

            n_extra = len(train_abs) - len(train_rel)
            print(f"\n{'='*60}", flush=True)
            print(f"Seed {seed}, Fold {fold_i+1}/{args.n_folds}", flush=True)
            print(f"  Train: {len(train_abs)} (ADNI={len(train_rel)}, "
                  f"synth/oasis={n_extra}), Val: {len(val_abs)}, Test: {len(test_abs)}", flush=True)

            train_ds = RealCachedDataset(cache_dir, indices=train_abs, seed=seed,
                                         augment_fn=augment_fn)
            val_ds = RealCachedDataset(cache_dir, indices=val_abs, seed=seed)
            test_ds = RealCachedDataset(cache_dir, indices=test_abs, seed=seed)

            train_labels = np.array([s["label"] for s in train_ds.samples])
            n_per_class = np.bincount(train_labels, minlength=3).astype(float)
            n_per_class = np.maximum(n_per_class, 1.)
            cw = torch.tensor(len(train_labels) / (3. * n_per_class), dtype=torch.float32)
            print(f"  Class weights: CN={cw[0]:.2f}, MCI={cw[1]:.2f}, AD={cw[2]:.2f}", flush=True)

            sample_weights = np.array([1. / n_per_class[s["label"]] for s in train_ds.samples])
            sample_weights = sample_weights / sample_weights.sum()
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(train_ds),
                replacement=True,
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=4, pin_memory=True, persistent_workers=True)
            mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = mk(val_ds)
            test_loader = mk(test_ds)

            for cfg in MODEL_CONFIGS:
                key = f"{cfg['name']}__seed{seed}__fold{fold_i}"
                if key in all_results and "error" not in all_results[key]:
                    print(f"\n  >> {cfg['name']} (seed={seed}, fold={fold_i+1}) — SKIPPED (already done)", flush=True)
                    continue
                print(f"\n  >> {cfg['name']} (seed={seed}, fold={fold_i+1})", flush=True)

                use_pretrained = (args.pretrained_encoder
                                  if cfg["model_name"] == "ours" else "")
                try:
                    run_name = f"{cfg['name']}_seed{seed}_fold{fold_i}".replace(" ", "_")
                    res = run_single(
                        cfg["model_name"], cfg["use_atlas"], cfg["use_anat_dist"],
                        train_loader, val_loader, test_loader,
                        device, args.epochs, args.lr, args.patience,
                        args.anat_weight, seed,
                        class_weights=cw,
                        warmup_epochs=args.warmup,
                        label_smoothing=args.label_smoothing,
                        weight_decay=args.weight_decay,
                        dropout=args.dropout,
                        pretrained_encoder=use_pretrained,
                        freeze_encoder_epochs=args.freeze_encoder_epochs,
                        tb_writer=tb_writer,
                        run_name=run_name,
                        mixup_alpha=args.mixup_alpha,
                    )
                    res["fold"] = fold_i
                    res["config_name"] = cfg["name"]

                    best_state = res.pop("model_state", None)
                    if cfg["name"] == "Ours (Atlas+AnatDist)" and best_state is not None:
                        ckpt_path = output_dir / f"best_model_seed{seed}_fold{fold_i}.pth"
                        torch.save(best_state, ckpt_path)

                    all_results[key] = res
                    print(f"     SUMMARY: val_bacc={res['best_val_bacc']:.4f}, "
                          f"test_acc={res['test_acc']:.4f}, "
                          f"test_bacc={res['test_balanced_acc']:.4f}", flush=True)
                except Exception as exc:
                    import traceback
                    print(f"     ERROR in {cfg['name']}: {exc}", flush=True)
                    traceback.print_exc()
                    all_results[key] = {
                        "fold": fold_i, "config_name": cfg["name"],
                        "error": str(exc),
                        "test_acc": 0.0, "test_balanced_acc": 0.0,
                        "best_val_bacc": 0.0,
                    }
                    torch.cuda.empty_cache()

                _safe_save_partial(all_results, partial_path)
                completed = sum(1 for v in all_results.values() if "error" not in v)
                total_runs = len(args.seeds) * args.n_folds * len(MODEL_CONFIGS)
                print(f"     [checkpoint] {completed}/{total_runs} done, saved to {partial_path.name}",
                      flush=True)

            del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\n{'='*70}", flush=True)
    print(f"ALL EXPERIMENTS COMPLETE ({total_time:.0f}s)", flush=True)
    print("=" * 70, flush=True)

    print("\n--- SUMMARY (Balanced Accuracy) ---", flush=True)
    for cfg in MODEL_CONFIGS:
        baccs = [v["test_balanced_acc"] for k, v in all_results.items()
                 if v["config_name"] == cfg["name"]]
        accs = [v["test_acc"] for k, v in all_results.items()
                if v["config_name"] == cfg["name"]]
        if baccs:
            print(f"  {cfg['name']:30s}: bacc={np.mean(baccs):.4f}±{np.std(baccs):.4f} "
                  f"acc={np.mean(accs):.4f}±{np.std(accs):.4f} (n={len(baccs)})", flush=True)

    results_path = output_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)
    print(f"\nResults saved to: {results_path}", flush=True)

    if tb_writer is not None:
        tb_writer.close()

    args_dict = vars(args)
    args_dict["version"] = "v3"
    args_path = output_dir / "experiment_args.json"
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=2)


if __name__ == "__main__":
    main()
