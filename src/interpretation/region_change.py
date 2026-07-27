"""Aggregate brain-region change summaries."""

from __future__ import annotations

from typing import Mapping


def region_change_summary(row: Mapping[str, object]) -> str:
    region = str(row["region"])
    direction = str(row.get("observed_direction", "unknown"))
    delta = row.get("ad_minus_cn_percent", "NA")
    effect = row.get("cohens_d", "NA")
    return f"{region}: AD-CN direction={direction}, delta={delta}%, Cohen d={effect}"
