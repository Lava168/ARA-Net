"""Smoke tests for ARA-Net's manuscript shape contract and loss formulation.

Run with:

    python -m pytest tests/test_shapes.py -q

Each test exercises a single invariant from Manuscript §2.3 / §2.5 so that a
failure points reviewers directly to the broken assumption.
"""
from __future__ import annotations

import math

import torch
import pytest

from chapter1_foundation.models import (
    create_model,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_FEATURE_SHAPE,
    EXPECTED_NUM_REGION_TOKENS,
    EXPECTED_FEATURE_DIM,
    EXPECTED_NUM_HEADS,
    EXPECTED_HEAD_DIM,
    EXPECTED_NUM_ATTN_LAYERS,
    EXPECTED_NUM_CLASSES,
)
from chapter1_foundation.losses import (
    AnatomicalRegularizationLoss,
    lambda_anneal,
)


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fake_batch():
    """A 2-sample synthetic batch matching the manuscript I/O contract."""
    torch.manual_seed(0)
    B = 2
    image = torch.randn(B, *EXPECTED_INPUT_SHAPE)                   # (2, 1, 96, 112, 96)
    # Synthetic segmentation with all 21 regions present.
    seg = torch.zeros(B, *EXPECTED_INPUT_SHAPE[1:], dtype=torch.long)
    for r in range(1, EXPECTED_NUM_REGION_TOKENS + 1):
        # Carve out a small cube per region so every region has a few voxels.
        z = (r * 5) % EXPECTED_INPUT_SHAPE[1]
        y = (r * 5) % EXPECTED_INPUT_SHAPE[2]
        x = (r * 5) % EXPECTED_INPUT_SHAPE[3]
        seg[:, z:z + 3, y:y + 3, x:x + 3] = r
    return image, seg


def test_model_default_hyperparameters():
    model = create_model()
    cfg_layers = sum(1 for _ in model.attn_layers)
    assert cfg_layers == EXPECTED_NUM_ATTN_LAYERS
    assert model.attn_layers[0].num_heads == EXPECTED_NUM_HEADS
    assert model.attn_layers[0].head_dim == EXPECTED_HEAD_DIM
    assert model.num_regions == EXPECTED_NUM_REGION_TOKENS
    # Final classifier emits 3 logits (CN / MCI / AD).
    last_linear = list(model.classifier.children())[-1]
    assert last_linear.out_features == EXPECTED_NUM_CLASSES


def test_encoder_feature_shape(fake_batch):
    image, _ = fake_batch
    model = create_model().eval()
    with torch.no_grad():
        feats = model.encoder(image)
        feats = model.proj(feats)
    expected = (image.shape[0], *EXPECTED_FEATURE_SHAPE)
    assert tuple(feats.shape) == expected, (
        f"Encoder feature shape {tuple(feats.shape)} != expected {expected}"
    )


def test_forward_returns_3_class_logits(fake_batch):
    image, seg = fake_batch
    model = create_model().eval()
    with torch.no_grad():
        out = model(image, segmentation=seg, return_attention=True,
                    return_features=True)
    logits = out["logits"]
    assert logits.shape == (image.shape[0], EXPECTED_NUM_CLASSES)
    region_features = out["region_features"]
    assert region_features.shape == (
        image.shape[0],
        EXPECTED_NUM_REGION_TOKENS,
        EXPECTED_FEATURE_DIM,
    )
    attention = out["attention"]
    # (B, H, N, N)
    assert attention.shape == (
        image.shape[0],
        EXPECTED_NUM_HEADS,
        EXPECTED_NUM_REGION_TOKENS,
        EXPECTED_NUM_REGION_TOKENS,
    )


def test_attention_rows_are_probability_distributions(fake_batch):
    image, seg = fake_batch
    model = create_model().eval()
    with torch.no_grad():
        out = model(image, segmentation=seg, return_attention=True)
    attn = out["attention"]
    # Each query row should sum to ~1 across keys for valid (non-masked) entries.
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


# ---------------------------------------------------------------------------
# Loss contract  —  Manuscript Eq. 6
# ---------------------------------------------------------------------------
def test_anat_loss_matches_eq6_formula():
    torch.manual_seed(1)
    B, H, N = 2, 4, 21
    raw = torch.randn(B, H, N, N)
    attn = torch.softmax(raw, dim=-1)

    alpha, beta = 0.05, 0.005
    loss_fn = AnatomicalRegularizationLoss(alpha=alpha, beta=beta)
    loss = loss_fn(attn).item()

    # Reference computation that follows Eq. (6) verbatim.
    eps = 1e-8
    entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean().item()
    mean_attn = attn.mean(dim=(0, 1, 2))
    l1 = mean_attn.abs().sum().item()
    expected = alpha * entropy - beta * l1

    assert math.isclose(loss, expected, rel_tol=1e-5, abs_tol=1e-7), (
        f"AnatomicalRegularizationLoss does not match Eq. (6): "
        f"got {loss}, expected {expected}"
    )


def test_lambda_anneal_endpoints():
    # λ(0) == λ_max; λ(T-1) == λ_min; monotonic decrease in between.
    T = 50
    lam0 = lambda_anneal(0, T, lambda_max=1.0, lambda_min=0.1)
    lamT = lambda_anneal(T - 1, T, lambda_max=1.0, lambda_min=0.1)
    assert math.isclose(lam0, 1.0, abs_tol=1e-6)
    assert math.isclose(lamT, 0.1, abs_tol=1e-6)
    seq = [lambda_anneal(t, T, lambda_max=1.0, lambda_min=0.1) for t in range(T)]
    assert all(seq[i] >= seq[i + 1] for i in range(T - 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
