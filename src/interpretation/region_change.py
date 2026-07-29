"""Aggregate brain-region change summaries and AD-key concentration helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

from src.atlas.regions import AD_KEY_REGIONS


def region_change_summary(row: Mapping[str, object]) -> str:
    region = str(row["region"])
    direction = str(row.get("observed_direction", "unknown"))
    delta = row.get("ad_minus_cn_percent", "NA")
    effect = row.get("cohens_d", "NA")
    return f"{region}: AD-CN direction={direction}, delta={delta}%, Cohen d={effect}"


def ad_key_concentration_score(
    region_scores: Mapping[str, float],
    *,
    ad_key_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Share of |score| mass falling in a priori AD-key regions.

    Manuscript locked AIBL held-out enrichment was 0.510 vs uniform null 6/21=0.286.
    This helper is for public aggregate summaries, not subject-level MRI redistribution.
    """
    if ad_key_names is None:
        ad_key_names = tuple(
            name for spec in AD_KEY_REGIONS.values() for name in spec.get("names", [])
        )
    total = sum(abs(float(v)) for v in region_scores.values())
    if total <= 0:
        return {"score": float("nan"), "null_uniform": 6.0 / 21.0, "n_regions": float(len(region_scores))}
    key_mass = sum(abs(float(region_scores[k])) for k in ad_key_names if k in region_scores)
    return {
        "score": float(key_mass / total),
        "null_uniform": 6.0 / 21.0,
        "n_regions": float(len(region_scores)),
        "n_ad_key": float(len(ad_key_names)),
    }
