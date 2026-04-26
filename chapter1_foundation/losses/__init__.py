"""Losses for Chapter 1 / ARA-Net.

Preferred entry point is :class:`AnatomicalRegularizationLoss`, which
implements Manuscript Eq. (6) exactly. The remaining symbols are kept
for backward compatibility with earlier experiment scripts.
"""
from .geodesic_loss import (
    AnatomicalDistanceLoss,
    AnatomicalRegularizationLoss,
    GeodesicAttentionLoss,
    lambda_anneal,
)

__all__ = [
    "AnatomicalRegularizationLoss",
    "lambda_anneal",
    "AnatomicalDistanceLoss",
    "GeodesicAttentionLoss",
]
