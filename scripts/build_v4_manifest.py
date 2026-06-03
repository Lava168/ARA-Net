#!/usr/bin/env python3
"""Build a subject-level v4 manifest for external-generalization experiments.

The v4 manifest is intentionally explicit: each scan has a dataset, subject,
label, path, and split. ADNI is split at subject level for internal training;
AIBL is split at subject level into adaptation train/val/heldout while also
remaining an external cohort; OASIS and IXI are locked as external tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]
ADNI_RE = re.compile(r"^ADNI_([0-9]{3}_S_[0-9]{4})")
AIBL_RE = re.compile(r"^AIBL_([0-9]+)_")
OASIS_RE = re.compile(r"^(OASIS_)?(OAS[0-9]_[0-9]+)")


def read_label(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        if "label" not in data.files:
            return 0
        return int(np.asarray(data["label"]).item())


def parse_subject(dataset: str, stem: str) -> str:
    if dataset == "ADNI":
        match = ADNI_RE.match(stem)
        return match.group(1) if match else stem.replace("ADNI_", "", 1)
    if dataset == "AIBL":
        match = AIBL_RE.match(stem)
        return f"AIBL_{match.group(1)}" if match else stem
    if dataset == "OASIS":
        match = OASIS_RE.match(stem)
        return match.group(2) if match else stem
    if dataset == "IXI":
        return stem.split("_", 1)[0]
    return stem


def scan_cache(cache_dir: Path, dataset: str, prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(cache_dir.glob("*.npz")):
        stem = path.stem
        if prefix and not stem.startswith(prefix):
            continue
        if stem.startswith("SYNTH"):
            continue
        label = read_label(path)
        if label < 0 or label >= len(CLASS_NAMES):
            continue
        rows.append(
            {
                "dataset": dataset,
                "subject_id": parse_subject(dataset, stem),
                "scan_id": stem,
                "label": label,
                "label_name": CLASS_NAMES[label],
                "path": str(path),
                "cache_space": "96x112x96",
                "has_seg": "1",
            }
        )
    return rows


def subject_labels(rows: Sequence[dict]) -> Dict[str, int]:
    by_subject: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(int(row["label"]))
    labels = {}
    for subject, values in by_subject.items():
        counts = Counter(values)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]
        labels[subject] = max(candidates)
    return labels


def split_subjects(
    labels_by_subject: Dict[str, int],
    ratios: Tuple[float, float, float],
    names: Tuple[str, str, str],
    seed: int,
) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    split_by_subject: Dict[str, str] = {}
    for label in sorted(set(labels_by_subject.values())):
        subjects = [s for s, y in labels_by_subject.items() if y == label]
        subjects = list(rng.permutation(subjects))
        n = len(subjects)
        n0 = int(round(n * ratios[0]))
        n1 = int(round(n * ratios[1]))
        if n0 + n1 > n:
            n1 = max(0, n - n0)
        blocks = (
            subjects[:n0],
            subjects[n0 : n0 + n1],
            subjects[n0 + n1 :],
        )
        for split_name, block in zip(names, blocks):
            for subject in block:
                split_by_subject[subject] = split_name
    return split_by_subject


def add_split(rows: Iterable[dict], split_by_subject: Dict[str, str], default: str) -> List[dict]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["split"] = split_by_subject.get(row["subject_id"], default)
        out.append(copied)
    return out


def summarize(rows: Sequence[dict]) -> dict:
    summary: dict = {"n_scans": len(rows)}
    for key in ["dataset", "split"]:
        counts = defaultdict(Counter)
        subjects = defaultdict(set)
        for row in rows:
            bucket = row[key]
            counts[bucket][row["label_name"]] += 1
            subjects[bucket].add(row["subject_id"])
        summary[f"{key}_counts"] = {
            bucket: {
                "scans": int(sum(counter.values())),
                "subjects": int(len(subjects[bucket])),
                "labels": {name: int(counter.get(name, 0)) for name in CLASS_NAMES},
            }
            for bucket, counter in sorted(counts.items())
        }

    overlap_checks = {}
    split_subjects_map = defaultdict(set)
    for row in rows:
        split_subjects_map[row["split"]].add(row["subject_id"])
    splits = sorted(split_subjects_map)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            # Only check leakage inside the same dataset family. ADNI and AIBL
            # subject IDs are disjoint by construction.
            overlap = split_subjects_map[left] & split_subjects_map[right]
            if overlap:
                overlap_checks[f"{left}__{right}"] = sorted(overlap)[:50]
    summary["subject_overlap_checks"] = overlap_checks
    return summary


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "subject_id",
        "scan_id",
        "label",
        "label_name",
        "path",
        "cache_space",
        "has_seg",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter1-cache", type=Path, required=True)
    parser.add_argument("--aibl-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--adni-train", type=float, default=0.70)
    parser.add_argument("--adni-val", type=float, default=0.15)
    parser.add_argument("--aibl-adapt-train", type=float, default=0.55)
    parser.add_argument("--aibl-adapt-val", type=float, default=0.15)
    args = parser.parse_args()

    if not args.chapter1_cache.is_dir():
        raise FileNotFoundError(args.chapter1_cache)
    if not args.aibl_cache.is_dir():
        raise FileNotFoundError(args.aibl_cache)

    adni = scan_cache(args.chapter1_cache, "ADNI", "ADNI_")
    oasis = scan_cache(args.chapter1_cache, "OASIS", "OASIS_")
    ixi = scan_cache(args.chapter1_cache, "IXI", "IXI")
    aibl = scan_cache(args.aibl_cache, "AIBL", "AIBL_")

    adni_subject_split = split_subjects(
        subject_labels(adni),
        (args.adni_train, args.adni_val, 1.0 - args.adni_train - args.adni_val),
        ("train", "val", "internal_test"),
        args.seed,
    )
    aibl_subject_split = split_subjects(
        subject_labels(aibl),
        (
            args.aibl_adapt_train,
            args.aibl_adapt_val,
            1.0 - args.aibl_adapt_train - args.aibl_adapt_val,
        ),
        ("aibl_adapt_train", "aibl_adapt_val", "aibl_heldout"),
        args.seed + 17,
    )

    rows = []
    rows.extend(add_split(adni, adni_subject_split, "train"))
    rows.extend(add_split(aibl, aibl_subject_split, "aibl_heldout"))
    rows.extend(add_split(oasis, {}, "oasis_external"))
    rows.extend(add_split(ixi, {}, "ixi_external"))
    rows = sorted(rows, key=lambda r: (r["dataset"], r["split"], r["subject_id"], r["scan_id"]))

    summary = summarize(rows)
    summary["inputs"] = {
        "chapter1_cache": str(args.chapter1_cache),
        "aibl_cache": str(args.aibl_cache),
        "seed": args.seed,
        "adni_split_ratios": [args.adni_train, args.adni_val, 1.0 - args.adni_train - args.adni_val],
        "aibl_adapt_split_ratios": [
            args.aibl_adapt_train,
            args.aibl_adapt_val,
            1.0 - args.aibl_adapt_train - args.aibl_adapt_val,
        ],
    }

    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.output_csv}")
    print(f"[saved] {args.output_json}")
    print(json.dumps(summary["split_counts"], indent=2))


if __name__ == "__main__":
    main()
