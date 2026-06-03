#!/usr/bin/env python3
"""Split OASIS external rows into adaptation and locked heldout subsets.

The v4 manifest keeps all OASIS scans as one external stress-test split. That
is useful for zero-shot evaluation, but it cannot tell us whether OASIS failure
is due to irreducible label/domain mismatch or simply lack of adaptation. This
script creates an alternate manifest where OASIS is split at subject level into:

* oasis_adapt_train
* oasis_adapt_val
* oasis_heldout

All non-OASIS rows are preserved unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def subject_labels(rows: Sequence[dict]) -> Dict[str, int]:
    by_subject: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(int(row["label"]))
    labels = {}
    for subject, values in by_subject.items():
        counts = Counter(values)
        max_count = max(counts.values())
        labels[subject] = max(label for label, count in counts.items() if count == max_count)
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
        blocks = (subjects[:n0], subjects[n0 : n0 + n1], subjects[n0 + n1 :])
        for split_name, block in zip(names, blocks):
            for subject in block:
                split_by_subject[subject] = split_name
    return split_by_subject


def summarize(rows: Sequence[dict]) -> dict:
    counts = defaultdict(Counter)
    subjects = defaultdict(set)
    for row in rows:
        split = row["split"]
        counts[split][row["label_name"]] += 1
        subjects[split].add(row["subject_id"])
    return {
        split: {
            "scans": int(sum(counter.values())),
            "subjects": int(len(subjects[split])),
            "labels": {name: int(counter.get(name, 0)) for name in CLASS_NAMES},
        }
        for split, counter in sorted(counts.items())
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--adapt-train", type=float, default=0.50)
    parser.add_argument("--adapt-val", type=float, default=0.20)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    oasis_rows = [row for row in rows if row["dataset"] == "OASIS"]
    other_rows = [row for row in rows if row["dataset"] != "OASIS"]
    ratios = (args.adapt_train, args.adapt_val, 1.0 - args.adapt_train - args.adapt_val)
    split_map = split_subjects(
        subject_labels(oasis_rows),
        ratios,
        ("oasis_adapt_train", "oasis_adapt_val", "oasis_heldout"),
        args.seed,
    )
    updated_oasis = []
    for row in oasis_rows:
        copied = dict(row)
        copied["split"] = split_map[row["subject_id"]]
        updated_oasis.append(copied)
    out_rows = sorted(
        other_rows + updated_oasis,
        key=lambda row: (row["dataset"], row["split"], row["subject_id"], row["scan_id"]),
    )
    summary = {
        "source_manifest": str(args.manifest),
        "seed": args.seed,
        "oasis_ratios": list(ratios),
        "split_counts": summarize(out_rows),
    }
    write_csv(args.output_csv, out_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {args.output_csv}")
    print(f"[saved] {args.output_json}")
    print(json.dumps({k: v for k, v in summary["split_counts"].items() if k.startswith("oasis")}, indent=2))


if __name__ == "__main__":
    main()
