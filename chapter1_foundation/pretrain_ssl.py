#!/usr/bin/env python3
"""
Self-Supervised Pretraining for FeatureEncoder3D (Models Genesis style).

Pretext tasks (applied simultaneously):
  1. Non-linear voxel intensity transform — forces learning intensity invariance
  2. Local pixel shuffling — forces learning local spatial coherence
  3. Random 3D inpainting (block masking) — forces learning global context
  4. Random out-painting (outer crop) — forces learning boundary completion

The encoder-decoder pair is trained to reconstruct the original image from
these corrupted versions. After pretraining, only the encoder weights are
saved and loaded into ARA-Net for supervised fine-tuning.

Data: All 887 cached MRIs (306 ADNI + 581 IXI) — no labels needed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from chapter1_foundation.models.atlas_guided_model import (
    FeatureEncoder3D, ConvBlock3D,
)


# ---------------------------------------------------------------------------
# Self-Supervised Corruption Transforms
# ---------------------------------------------------------------------------

class ModelsGenesisTransform:
    """Apply Models Genesis-style corruptions to a 3D volume.

    Returns (corrupted, original) pair for reconstruction.
    """
    def __init__(
        self,
        nonlinear_prob: float = 0.9,
        local_shuffle_prob: float = 0.5,
        inpaint_prob: float = 0.9,
        outpaint_prob: float = 0.5,
    ):
        self.nonlinear_prob = nonlinear_prob
        self.local_shuffle_prob = local_shuffle_prob
        self.inpaint_prob = inpaint_prob
        self.outpaint_prob = outpaint_prob

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original = image.copy()
        corrupted = image.copy()

        if np.random.rand() < self.nonlinear_prob:
            corrupted = self._nonlinear_transform(corrupted)

        if np.random.rand() < self.local_shuffle_prob:
            corrupted = self._local_pixel_shuffle(corrupted)

        if np.random.rand() < self.inpaint_prob:
            corrupted = self._inpainting(corrupted)
        elif np.random.rand() < self.outpaint_prob:
            corrupted = self._outpainting(corrupted, original)

        return corrupted.astype(np.float32), original.astype(np.float32)

    def _nonlinear_transform(self, img: np.ndarray) -> np.ndarray:
        points = np.sort(np.random.uniform(0, 1, 4))
        values = np.sort(np.random.uniform(0, 1, 4))
        points = np.concatenate([[0], points, [1]])
        values = np.concatenate([[0], values, [1]])
        return np.interp(img, points, values).astype(np.float32)

    def _local_pixel_shuffle(self, img: np.ndarray, window: int = 5) -> np.ndarray:
        D, H, W = img.shape
        n_swaps = int(D * H * W * 0.01)
        for _ in range(n_swaps):
            d = np.random.randint(0, max(D - window, 1))
            h = np.random.randint(0, max(H - window, 1))
            w = np.random.randint(0, max(W - window, 1))
            patch = img[d:d+window, h:h+window, w:w+window].copy()
            flat = patch.flatten()
            np.random.shuffle(flat)
            img[d:d+window, h:h+window, w:w+window] = flat.reshape(patch.shape)
        return img

    def _inpainting(self, img: np.ndarray) -> np.ndarray:
        D, H, W = img.shape
        n_blocks = np.random.randint(3, 8)
        for _ in range(n_blocks):
            bd = np.random.randint(D // 8, D // 3)
            bh = np.random.randint(H // 8, H // 3)
            bw = np.random.randint(W // 8, W // 3)
            d0 = np.random.randint(0, max(D - bd, 1))
            h0 = np.random.randint(0, max(H - bh, 1))
            w0 = np.random.randint(0, max(W - bw, 1))
            img[d0:d0+bd, h0:h0+bh, w0:w0+bw] = np.random.uniform(0, 1)
        return img

    def _outpainting(self, img: np.ndarray, original: np.ndarray) -> np.ndarray:
        D, H, W = img.shape
        margin_d = np.random.randint(D // 8, D // 4)
        margin_h = np.random.randint(H // 8, H // 4)
        margin_w = np.random.randint(W // 8, W // 4)
        result = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
        result[margin_d:D-margin_d, margin_h:H-margin_h, margin_w:W-margin_w] = \
            img[margin_d:D-margin_d, margin_h:H-margin_h, margin_w:W-margin_w]
        return result


# ---------------------------------------------------------------------------
# Pretraining Dataset
# ---------------------------------------------------------------------------

class SSLDataset(Dataset):
    """Load all cached MRI volumes (ADNI + IXI) for self-supervised pretraining."""

    def __init__(self, cache_dir: str, transform: ModelsGenesisTransform = None):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(
            list(self.cache_dir.glob("ADNI_*.npz")) +
            list(self.cache_dir.glob("IXI*.npz"))
        )
        if not self.files:
            raise FileNotFoundError(f"No ADNI/IXI .npz files in {cache_dir}")
        self.transform = transform or ModelsGenesisTransform()
        print(f"  SSLDataset: {len(self.files)} volumes for pretraining")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = data["image"].astype(np.float32)

        lo, hi = np.percentile(image, [1, 99])
        image = np.clip(image, lo, hi)
        image = (image - lo) / max(hi - lo, 1e-8)

        corrupted, original = self.transform(image)

        corrupted_t = torch.from_numpy(corrupted).unsqueeze(0)  # (1, D, H, W)
        original_t = torch.from_numpy(original).unsqueeze(0)
        return corrupted_t, original_t


# ---------------------------------------------------------------------------
# Decoder (symmetric to FeatureEncoder3D)
# ---------------------------------------------------------------------------

class FeatureDecoder3D(nn.Module):
    """Symmetric decoder for reconstruction.

    Mirrors the encoder:
      256 @ 6×7×6 -> 128 @ 12×14×12 -> 64 @ 24×28×24 -> 32 @ 48×56×48 -> 32 @ 96×112×96 -> 1 @ 96×112×96
    """
    def __init__(self, base_channels: int = 32, num_stages: int = 4):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(num_stages)]
        channels = channels[::-1]  # [256, 128, 64, 32]

        self.up_stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.up_stages.append(nn.Sequential(
                nn.ConvTranspose3d(channels[i], channels[i+1], 2, stride=2),
                nn.BatchNorm3d(channels[i+1]),
                nn.GELU(),
                nn.Conv3d(channels[i+1], channels[i+1], 3, padding=1),
                nn.BatchNorm3d(channels[i+1]),
                nn.GELU(),
            ))

        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(channels[-1], channels[-1], 2, stride=2),
            nn.BatchNorm3d(channels[-1]),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv3d(channels[-1], channels[-1], 3, padding=1),
            nn.BatchNorm3d(channels[-1]),
            nn.GELU(),
            nn.Conv3d(channels[-1], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.up_stages:
            x = stage(x)
        x = self.final_up(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Combined Encoder-Decoder for pretraining
# ---------------------------------------------------------------------------

class SSLModel(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.encoder = FeatureEncoder3D(
            in_channels=1, base_channels=base_channels,
            num_stages=4, dropout=0.0,
        )
        self.decoder = FeatureDecoder3D(
            base_channels=base_channels, num_stages=4,
        )
        self.use_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            feats = torch.utils.checkpoint.checkpoint(
                self.encoder, x, use_reentrant=False)
            recon = torch.utils.checkpoint.checkpoint(
                self.decoder, feats, use_reentrant=False)
        else:
            feats = self.encoder(x)
            recon = self.decoder(feats)
        return recon


# ---------------------------------------------------------------------------
# Loss: MSE + perceptual (multi-scale feature) + SSIM-like
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.5):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)

        if self.ssim_weight > 0:
            mu_p = F.avg_pool3d(pred, 3, stride=1, padding=1)
            mu_t = F.avg_pool3d(target, 3, stride=1, padding=1)
            sigma_p = F.avg_pool3d(pred ** 2, 3, stride=1, padding=1) - mu_p ** 2
            sigma_t = F.avg_pool3d(target ** 2, 3, stride=1, padding=1) - mu_t ** 2
            sigma_pt = F.avg_pool3d(pred * target, 3, stride=1, padding=1) - mu_p * mu_t

            C1, C2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
            ssim_loss = 1 - ssim.mean()
            return mse + self.ssim_weight * ssim_loss

        return mse


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_ssl(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Self-Supervised Pretraining (Models Genesis style)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    transform = ModelsGenesisTransform()
    dataset = SSLDataset(args.cache_dir, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        drop_last=True,
    )

    model = SSLModel(base_channels=args.base_channels).to(device)
    model.use_checkpointing = True
    accum_steps = args.accum_steps
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    n_params_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Encoder params: {n_params_enc:,}, Decoder params: {n_params_dec:,}")
    print(f"Grad accumulation: {accum_steps} steps (effective batch={args.batch_size * accum_steps})")

    criterion = ReconstructionLoss(ssim_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = min(5, args.epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    history = []

    print(f"\nStarting pretraining on {len(dataset)} volumes...")
    print(f"Batches per epoch: {len(loader)}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.
        t0 = time.time()

        optimizer.zero_grad()
        for batch_idx, (corrupted, original) in enumerate(loader):
            corrupted = corrupted.to(device)
            original = original.to(device)

            recon = model(corrupted)

            target_size = recon.shape[2:]
            if original.shape[2:] != target_size:
                original = F.interpolate(original, size=target_size, mode='trilinear',
                                         align_corners=False)

            loss = criterion(recon, original) / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        improved = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(),
                       output_dir / "pretrained_encoder.pth")
            torch.save(model.state_dict(),
                       output_dir / "pretrained_full.pth")
            improved = " * (saved)"

        history.append({"epoch": epoch, "loss": avg_loss, "lr": lr})

        if epoch <= 5 or epoch % 5 == 0 or improved:
            print(f"  Ep {epoch:3d}/{args.epochs} ({dt:.1f}s) "
                  f"loss={avg_loss:.6f} lr={lr:.1e}{improved}", flush=True)

    print(f"\nPretraining complete. Best loss: {best_loss:.6f}")
    print(f"Encoder weights saved to: {output_dir / 'pretrained_encoder.pth'}")

    import json
    with open(output_dir / "pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return str(output_dir / "pretrained_encoder.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="sample_data/cache_real")
    parser.add_argument("--output_dir", default="chapter1_foundation/pretrained")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--accum_steps", type=int, default=4)
    args = parser.parse_args()

    train_ssl(args)


if __name__ == "__main__":
    main()
