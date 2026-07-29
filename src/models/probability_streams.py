"""Locked public base-model stream names and manuscript-facing metadata."""

from __future__ import annotations

from typing import Mapping

LOCKED_BASE_MODEL_STREAMS = [
    "aibl_adapted_atlas_biomarker_enhanced__hgb",
    "aibl_adapted_atlas_core_clinical__hgb",
    "aibl_adapted_clinical_biomarker_only__rf_balanced",
    "aibl_adapted_clinical_core_only__hgb",
    "aibl_adapted_clinical_core_only__rf_balanced",
    "rf__logreg",
]

STREAM_METADATA: Mapping[str, Mapping[str, str]] = {
    "aibl_adapted_atlas_biomarker_enhanced__hgb": {
        "family": "atlas+clinical",
        "learner": "hist_gradient_boosting",
        "role": "atlas biomarker enhanced multimodal stream",
    },
    "aibl_adapted_atlas_core_clinical__hgb": {
        "family": "atlas+clinical",
        "learner": "hist_gradient_boosting",
        "role": "atlas + core clinical stream",
    },
    "aibl_adapted_clinical_biomarker_only__rf_balanced": {
        "family": "clinical",
        "learner": "random_forest_balanced",
        "role": "clinical biomarker-only stream",
    },
    "aibl_adapted_clinical_core_only__hgb": {
        "family": "clinical",
        "learner": "hist_gradient_boosting",
        "role": "core clinical stream",
    },
    "aibl_adapted_clinical_core_only__rf_balanced": {
        "family": "clinical",
        "learner": "random_forest_balanced",
        "role": "core clinical stream (balanced RF)",
    },
    "rf__logreg": {
        "family": "linear",
        "learner": "logistic_regression",
        "role": "linear comparator stream",
    },
}


def n_locked_streams() -> int:
    return len(LOCKED_BASE_MODEL_STREAMS)


def stream_summary() -> list[dict[str, str]]:
    return [
        {"name": name, **dict(STREAM_METADATA[name])}
        for name in LOCKED_BASE_MODEL_STREAMS
    ]
