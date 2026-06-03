#!/usr/bin/env python3
"""Generate a public-release manifest for the ARA-Net repository.

The manifest is intentionally built from `git ls-files`, so it describes what
will actually be visible in the public repository rather than local ignored
outputs. It also checks for restricted row-level prediction artifacts.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_JSON = Path("reports/v6_final_model/public_release_manifest.json")
DEFAULT_MD = Path("reports/v6_final_model/public_release_manifest.md")

FORBIDDEN_PATTERNS = [
    "reports/**/final_subject_predictions_enriched.csv",
    "reports/**/*top_confident_errors.csv",
    "reports/**/oasis_locked*_predictions.csv",
    "reports/**/manifest_v4_oasis_locked_summary.json",
    "reports/v6_final_model/final_rescue_model_summary.json",
    "outputs/**",
    "data/**",
    "sample_data/**",
    "cache/**",
    "cache_real/**",
    "**/*.nii",
    "**/*.nii.gz",
    "**/*.mgz",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.pyc",
]

REPRODUCTION_COMMANDS = [
    {
        "name": "Research CLI smoke test",
        "command": (
            "python deployment/research_inference.py "
            "--input-csv examples/probability_input_example.csv "
            "--output examples/predictions_subject.csv --unit subject"
        ),
        "scope": "Checks the public probability-ensemble wrapper.",
    },
    {
        "name": "Core reviewer evidence matrix",
        "command": "python scripts/generate_core_reviewer_evidence_matrix.py",
        "scope": "Regenerates the external classification / CAS replacement / Braak-alternative evidence matrix.",
    },
    {
        "name": "Claim boundary audit",
        "command": "python scripts/audit_claim_boundaries.py",
        "scope": "Scans public Git-tracked files for unsupported Braak/CAS/OASIS/zero-shot/clinical-deployment overclaims.",
    },
    {
        "name": "Public release manifest",
        "command": "python scripts/generate_public_release_manifest.py",
        "scope": "Regenerates the public-file manifest and restricted-artifact check.",
    },
    {
        "name": "Final figures",
        "command": (
            "python scripts/generate_v6_final_figures.py "
            "--summary reports/v6_final_model/final_rescue_model_summary_public.json "
            "--table2 reports/v4/tables/table2_classification.csv "
            "--table-dir reports/v6_final_model/tables "
            "--out-dir reports/v6_final_model/figures"
        ),
        "scope": "Regenerates v6 aggregate manuscript figures after upstream result files exist.",
    },
]


def git_ls_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(path: Path) -> str:
    text = str(path)
    suffix = path.suffix.lower()
    if text.startswith("deployment/") or path.name == "Dockerfile":
        return "deployment"
    if text.startswith("frontend/"):
        return "frontend"
    if text.startswith("docs/") or path.name in {"README.md", "LICENSE"}:
        return "documentation"
    if text.startswith("scripts/"):
        return "analysis_code"
    if text.startswith("reports/") and suffix in {".png", ".pdf"}:
        return "figures"
    if text.startswith("reports/"):
        return "aggregate_reports"
    if text.startswith("examples/"):
        return "examples"
    if path.name.startswith("requirements"):
        return "environment"
    return "other"


def role(path: Path) -> str:
    text = str(path)
    if text == "reports/v6_final_model/core_reviewer_evidence_matrix.md":
        return "Three-core-issue reviewer evidence matrix."
    if text == "reports/v6_final_model/claim_boundary_audit.md":
        return "Public claim-boundary audit for reviewer-safe wording."
    if text == "reports/v6_final_model/final_rescue_model_summary_public.json":
        return "Public aggregate final-model metrics and bootstrap evidence."
    if text == "scripts/audit_claim_boundaries.py":
        return "Public overclaim audit script."
    if text == "deployment/final_ensemble_config.json":
        return "Locked deployable ensemble configuration."
    if text.startswith("frontend/"):
        return "Browser research console asset."
    if text.startswith("deployment/"):
        return "Research inference/deployment wrapper."
    if text.startswith("reports/v6_final_model/figures/"):
        return "Final v6 manuscript figure."
    if text.startswith("reports/v6_final_model/tables/"):
        return "Aggregate final v6 table."
    if text.startswith("reports/v4/tables/table"):
        return "Aggregate v4 rebuild table used by final evidence reports."
    if text.startswith("scripts/"):
        return "Reproducible analysis or report-generation script."
    if text.startswith("docs/"):
        return "Public documentation and claim-boundary document."
    return "Public repository file."


def forbidden_matches(paths: list[Path]) -> list[str]:
    matches = []
    for path in paths:
        text = str(path)
        for pattern in FORBIDDEN_PATTERNS:
            if fnmatch.fnmatch(text, pattern):
                matches.append(text)
                break
    return sorted(matches)


def build_manifest(paths: list[Path], excluded_outputs: set[str]) -> dict:
    files = []
    for path in paths:
        if str(path) in excluded_outputs:
            continue
        files.append(
            {
                "path": str(path),
                "category": category(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role(path),
            }
        )
    counts = Counter(item["category"] for item in files)
    restricted = forbidden_matches(paths)
    return {
        "file_count": len(files),
        "category_counts": dict(sorted(counts.items())),
        "restricted_artifact_check": {
            "status": "pass" if not restricted else "fail",
            "forbidden_patterns": FORBIDDEN_PATTERNS,
            "tracked_matches": restricted,
        },
        "data_boundary": {
            "raw_data_redistributed": False,
            "raw_data_note": "Raw ADNI, AIBL, OASIS, and IXI data are governed by source data-use agreements and are not redistributed.",
            "public_scope": "Code, deployment wrapper, aggregate reports, final figures, documentation, and toy probability examples.",
        },
        "manifest_note": "The manifest output files are excluded from their own file list to avoid self-referential hashes.",
        "reproduction_commands": REPRODUCTION_COMMANDS,
        "files": files,
    }


def write_markdown(manifest: dict, path: Path) -> None:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in manifest["files"]:
        by_category[item["category"]].append(item)

    lines = [
        "# Public Release Manifest",
        "",
        f"- Public tracked file count: {manifest['file_count']}",
        f"- Restricted artifact check: **{manifest['restricted_artifact_check']['status']}**",
        "- Public scope: code, deployment wrapper, aggregate reports, final figures, documentation, and toy probability examples.",
        "- Not redistributed: raw ADNI/AIBL/OASIS/IXI data, private clinical spreadsheets, row-level subject/scan predictions, MRI volumes, and model checkpoints.",
        f"- Manifest note: {manifest['manifest_note']}",
        "",
        "## Category Counts",
        "",
        "| category | files |",
        "|---|---:|",
    ]
    for name, count in manifest["category_counts"].items():
        lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "## Reproduction Commands",
        "",
        "| name | command | scope |",
        "|---|---|---|",
    ]
    for command in manifest["reproduction_commands"]:
        lines.append(
            f"| {command['name']} | `{command['command']}` | {command['scope']} |"
        )

    lines += [
        "",
        "## Restricted Artifact Check",
        "",
    ]
    matches = manifest["restricted_artifact_check"]["tracked_matches"]
    if matches:
        lines += ["Tracked restricted files:", ""]
        lines.extend(f"- `{match}`" for match in matches)
    else:
        lines.append("No tracked files matched the restricted row-level/data/model-artifact patterns.")

    lines += [
        "",
        "## Public Files",
        "",
    ]
    for name in sorted(by_category):
        lines += [f"### {name}", "", "| path | bytes | sha256 | role |", "|---|---:|---|---|"]
        for item in sorted(by_category[name], key=lambda value: value["path"]):
            lines.append(
                f"| `{item['path']}` | {item['bytes']} | `{item['sha256'][:12]}` | {item['role']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    paths = git_ls_files()
    excluded_outputs = {str(args.json), str(args.markdown)}
    manifest = build_manifest(paths, excluded_outputs)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(manifest, args.markdown)
    status = manifest["restricted_artifact_check"]["status"]
    print(f"[saved] {args.json}")
    print(f"[saved] {args.markdown}")
    print(f"[restricted_artifact_check] {status}")
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
