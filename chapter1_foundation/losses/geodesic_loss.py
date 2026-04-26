"""Anatomical regularization losses for ARA-Net.

This file implements the anatomical regularization term used by ARA-Net,
following Manuscript Section 2.3, Eq. (5)–(7):

    ℒ_total = ℒ_CE + λ(t) · ℒ_anat                                       (Eq. 5)
    ℒ_anat  = α · H(A) − β · ‖Ā‖₁                                         (Eq. 6)
    λ(t)    = anneals from λ_max to λ_min across training epochs          (Eq. 7)

with α = 0.05 and β = 0.005.

* `H(A)`     — entropy of the per-token attention distribution. Minimizing
               H(A) pushes the model toward decisive region selection.
* `‖Ā‖₁`     — L1 norm of the cross-batch / cross-head **mean** attention
               vector (length = num_regions). The negative sign in Eq. 6
               keeps a small reward for non-vanishing total mass on each
               region; combined with the entropy term it shapes attention
               toward a sparse-but-non-degenerate distribution.

Two classes are exported:

* `AnatomicalRegularizationLoss` — preferred, manuscript-aligned (Eq. 6).
* `AnatomicalDistanceLoss`      — legacy, kept for backward compatibility
                                  with earlier checkpoints / scripts that
                                  used a centroid-distance penalty.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Manuscript-aligned anatomical regularizer (Eq. 6)
# ---------------------------------------------------------------------------
class AnatomicalRegularizationLoss(nn.Module):
    r"""Anatomical regularization term ``ℒ_anat`` from Manuscript Eq. (6).

    .. math::

        \mathcal{L}_{\mathrm{anat}}
            = \alpha \cdot H(A) \; - \; \beta \cdot \lVert \bar{A} \rVert_1

    Args:
        alpha: weight ``α`` on the entropy term (default 0.05, manuscript value).
        beta:  weight ``β`` on the L1 of the mean attention (default 0.005).

    Forward signature:
        attention: tensor of softmax attention weights with shape
            ``(B, H, N_q, N_k)`` produced by the multi-head attention block.
            ``B`` = batch, ``H`` = number of heads, ``N_q``/``N_k`` = number
            of region tokens (= 21 in ARA-Net).

    Returns:
        Scalar loss tensor.

    Notes:
        * Entropy is computed along the *key* axis (axis = -1) and averaged
          over batch, heads, and queries — i.e. it measures how peaked the
          attention from each query is over its keys.
        * The "mean attention vector"  Ā ∈ ℝ^{N_k}  is obtained by averaging
          the attention map across batch, heads, and queries; its L1 norm is
          summed over regions.
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.005) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        if attention.dim() not in (3, 4):
            raise ValueError(
                f"attention must be 3-D (B, N_q, N_k) or 4-D (B, H, N_q, N_k); "
                f"got shape {tuple(attention.shape)}"
            )
        if attention.dim() == 3:
            # (B, N_q, N_k) → add a singleton head axis
            attention = attention.unsqueeze(1)

        eps = 1e-8

        # ------- Entropy term: α · H(A) ----------------------------------
        # Per-query entropy along the key axis, then averaged over (B, H, N_q).
        ent = -(attention * (attention + eps).log()).sum(dim=-1)  # (B, H, N_q)
        entropy = ent.mean()

        # ------- L1 of the mean attention vector: β · ‖Ā‖₁ --------------
        # Mean across batch, heads, and queries → 1-D vector of length N_k.
        mean_attn = attention.mean(dim=(0, 1, 2))                  # (N_k,)
        l1 = mean_attn.abs().sum()

        return self.alpha * entropy - self.beta * l1


# ---------------------------------------------------------------------------
# Legacy centroid-distance loss (kept for backward compatibility)
# ---------------------------------------------------------------------------
class AnatomicalDistanceLoss(nn.Module):
    """Multi-component anatomical regularizer used in early development.

    .. deprecated::
       Prefer :class:`AnatomicalRegularizationLoss`, which matches the
       manuscript exactly. This class is retained so that experiment
       scripts produced before the manuscript-aligned formulation are
       still runnable.

    Components (legacy):
      1. ``distance_weight``: penalizes attention paid to centroid-distant
         regions (centroid-based Euclidean distance, *not* surface geodesic).
      2. ``entropy_weight``:   weight on entropy (effective sign in the
         total: ``− entropy_weight · entropy``).
      3. ``sparsity_weight``:  L1 on the per-batch mean attention; for
         softmaxed attention this term is approximately constant, so it
         contributes little gradient in practice.
    """

    def __init__(
        self,
        distance_weight: float = 1.0,
        entropy_weight: float = 0.1,
        sparsity_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.distance_weight = distance_weight
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight

    def forward(
        self,
        attention: torch.Tensor,
        anatomical_distances: Optional[torch.Tensor] = None,
        atlas_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=attention.device, dtype=attention.dtype)

        if anatomical_distances is not None and self.distance_weight > 0:
            attn_mean = attention.mean(dim=1)
            if anatomical_distances.dim() == 2:
                dist = anatomical_distances.unsqueeze(0).expand(attn_mean.shape[0], -1, -1)
            else:
                dist = anatomical_distances
            dist_penalty = (attn_mean * dist).sum(dim=-1).mean()
            loss = loss + self.distance_weight * dist_penalty

        if self.entropy_weight > 0:
            entropy = -(attention * (attention + 1e-8).log()).sum(dim=-1).mean()
            loss = loss - self.entropy_weight * entropy

        if self.sparsity_weight > 0:
            sparsity = attention.mean(dim=1).norm(p=1, dim=-1).mean()
            loss = loss + self.sparsity_weight * sparsity

        return loss


# Backward-compatible alias used by some early scripts.
GeodesicAttentionLoss = AnatomicalDistanceLoss


# ---------------------------------------------------------------------------
# λ(t) annealing schedule (Eq. 7)
# ---------------------------------------------------------------------------
def lambda_anneal(
    epoch: int,
    total_epochs: int,
    lambda_max: float = 1.0,
    lambda_min: float = 0.1,
) -> float:
    """Linear annealing of the anatomical-regularizer weight ``λ(t)``.

    Mirrors Manuscript Eq. (7): strong anatomical guidance during early
    fine-tuning, gradually reduced so that classification accuracy can
    dominate in late training.

    Args:
        epoch: current 0-indexed epoch.
        total_epochs: total number of fine-tuning epochs.
        lambda_max: λ at epoch 0.
        lambda_min: λ at the final epoch (lower bound).

    Returns:
        ``λ(t) ∈ [lambda_min, lambda_max]``.
    """
    if total_epochs <= 1:
        return lambda_max
    progress = max(0.0, min(1.0, epoch / float(total_epochs - 1)))
    return float(lambda_max + (lambda_min - lambda_max) * progress)
