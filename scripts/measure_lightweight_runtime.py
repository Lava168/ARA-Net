#!/usr/bin/env python3
"""Measure deployment-side ARA-Net RC-SPE inference cost."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))

from research_inference import (  # noqa: E402
    aggregate_subjects,
    ensemble_scan_probabilities,
    load_config,
    validate_rows,
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def benchmark(rows: list[dict], config: dict, warmup: int, iterations: int) -> dict:
    validate_rows(rows, config)
    for _ in range(warmup):
        probs = ensemble_scan_probabilities(rows, config)
        aggregate_subjects(rows, probs, config)

    timings_ms: list[float] = []
    n_subjects = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        probs = ensemble_scan_probabilities(rows, config)
        subjects = aggregate_subjects(rows, probs, config)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        timings_ms.append(elapsed_ms)
        n_subjects = len(subjects)

    mean_ms = statistics.fmean(timings_ms)
    return {
        "iterations": iterations,
        "warmup_iterations": warmup,
        "scan_rows_per_batch": len(rows),
        "subject_units_per_batch": n_subjects,
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": statistics.median(timings_ms),
        "p95_ms_per_batch": percentile(timings_ms, 95),
        "mean_ms_per_scan_row": mean_ms / max(len(rows), 1),
        "mean_ms_per_subject_unit": mean_ms / max(n_subjects, 1),
        "throughput_scan_rows_per_second": 1000.0 * len(rows) / mean_ms,
        "throughput_subject_units_per_second": 1000.0 * n_subjects / mean_ms,
    }


def model_size_payload(config_path: Path, config: dict) -> dict:
    weights = config["weights"]
    offsets = config["offsets"]
    parameter_count = len(weights) + len(offsets) + 1
    return {
        "base_probability_streams": len(config["base_models"]),
        "scalar_parameters": parameter_count,
        "parameter_composition": {
            "base_stream_weights": len(weights),
            "class_offsets": len(offsets),
            "temperature": 1,
        },
        "raw_float64_parameter_bytes": parameter_count * 8,
        "json_config_bytes": len(config_path.read_bytes()),
        "gpu_inference": "not_applicable_numpy_probability_head",
        "upstream_pipeline_note": (
            "Metrics cover the public RC-SPE probability head only; raw MRI preprocessing, "
            "atlas extraction, and base-model artifact size are excluded."
        ),
    }


def write_markdown(path: Path, payload: dict) -> None:
    size = payload["model_size"]
    timing = payload["timing"]
    primary = payload["primary_evaluation"]
    lines = [
        "# ARA-Net Lightweight Runtime Metrics",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Base probability streams | {size['base_probability_streams']} |",
        f"| RC-SPE scalar parameters | {size['scalar_parameters']} |",
        f"| Raw float64 parameter storage | {size['raw_float64_parameter_bytes']} bytes |",
        f"| JSON config size | {size['json_config_bytes']} bytes |",
        f"| CPU mean runtime / batch | {timing['mean_ms_per_batch']:.3f} ms |",
        f"| CPU median runtime / batch | {timing['median_ms_per_batch']:.3f} ms |",
        f"| CPU p95 runtime / batch | {timing['p95_ms_per_batch']:.3f} ms |",
        f"| CPU mean runtime / scan row | {timing['mean_ms_per_scan_row']:.3f} ms |",
        f"| CPU mean runtime / subject unit | {timing['mean_ms_per_subject_unit']:.3f} ms |",
        f"| CPU throughput | {timing['throughput_scan_rows_per_second']:.0f} scan rows/s |",
        "| GPU inference | Not applicable for the NumPy RC-SPE probability head |",
        "",
        "## Evaluation Context",
        "",
        f"- Primary cohort: {primary.get('cohort', 'AIBL locked heldout')}.",
        f"- AIBL heldout accuracy: {primary.get('accuracy', 0):.3f}.",
        f"- AIBL heldout balanced accuracy: {primary.get('balanced_accuracy', 0):.3f}.",
        f"- AIBL heldout macro AUC: {primary.get('macro_auc_ovr', 0):.3f}.",
        "",
        "## Claim Boundary",
        "",
        size["upstream_pipeline_note"],
        "Do not describe these measurements as full raw-MRI end-to-end runtime.",
        "",
        "## Reproducibility",
        "",
        f"- Python: {platform.python_version()}.",
        f"- Platform: {platform.platform()}.",
        f"- NumPy: {np.__version__}.",
        f"- Benchmark iterations: {timing['iterations']}.",
        f"- Workload: {timing['scan_rows_per_batch']} scan rows aggregated into {timing['subject_units_per_batch']} subject units.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("deployment/final_ensemble_config.json"))
    parser.add_argument("--input-csv", type=Path, default=Path("examples/demo_probability_input.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/v6_final_model/tables/lightweight_runtime_metrics.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/v6_final_model/tables/lightweight_runtime_metrics.md"))
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_rows(args.input_csv)
    payload = {
        "model": "ARA-Net",
        "component": "RC-SPE probability ensemble head",
        "model_size": model_size_payload(args.config, config),
        "timing": benchmark(rows, config, args.warmup, args.iterations),
        "primary_evaluation": config.get("primary_evaluation", {}),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[saved] {args.output_json}")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
