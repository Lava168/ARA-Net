#!/usr/bin/env python3
"""Audit manuscript-facing claim boundaries for the revised ARA-Net package.

The goal is not to ban terms such as "Braak" or "clinical deployment"; the
revised paper must discuss them. The audit instead flags unsupported positive
claims, while allowing explicit negation and limitation language.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT = Path("reports/v6_final_model/claim_boundary_audit.md")
SCAN_ROOTS = [Path("README.md"), Path("docs"), Path("reports"), Path("scripts"), Path("deployment"), Path("frontend")]
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".html", ".js", ".css"}
SKIP_PATHS = {
    "scripts/audit_claim_boundaries.py",
    "scripts/audit_word_manuscript_claims.py",
    "reports/v6_final_model/claim_boundary_audit.md",
    "reports/v6_final_model/public_release_manifest.json",
    "reports/v6_final_model/public_release_manifest.md",
}

SAFE_CUES = [
    "not",
    "no longer",
    "do not",
    "does not",
    "did not",
    "cannot",
    "avoid",
    "removed",
    "remove",
    "replace",
    "replaced",
    "rather than",
    "instead",
    "limitation",
    "stress test",
    "stress-test",
    "unresolved",
    "weak",
    "not solved",
    "not successful",
    "not direct",
    "requires",
    "required",
    "research prototype",
    "not a medical device",
    "not cleared",
    "not approved",
    "not intended",
    "not marketed",
    "not presented",
    "not reported",
    "not used",
    "not claim",
    "without",
    "unsupported",
    "insufficient",
    "insufficiently",
    "non-significant",
    "invalid",
    "below",
    "failed",
    "failure",
    "concern",
    "problem",
    "reviewer-safe",
    "avoid:",
    "not allowed",
    "no ",
    "no neuropathological",
    "avoid",
    "do not claim",
    "do not frame",
    "must not",
    "should not",
    "quotes to avoid",
    "old wording to remove",
    "replacement rules",
    "target statement",
    "safest target",
    "requires formal",
    "may require",
    "depending on",
    "premarket review",
    "regulatory boundary",
    "regulatory assessment",
    "fda",
    "https://",
    "overview",
]


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    pattern: re.Pattern[str]
    unsafe_if_no_safe_cue: bool = True
    description: str = ""


RULES = [
    Rule(
        name="direct_braak_positive_claim",
        severity="blocker",
        pattern=re.compile(r"\b(direct\s+)?braak[- ]?(stage|staging)?\s*(validation|validated|proof|correlation|claim)", re.I),
        description="Direct Braak validation/proof must not be claimed without neuropathology labels.",
    ),
    Rule(
        name="attention_cas_biomarker_claim",
        severity="blocker",
        pattern=re.compile(r"\b(CAS|attention)[^.\\n]{0,80}\b(validated|validates|biomarker|clinical alignment)", re.I),
        description="The old attention-only CAS must not be presented as a validated biomarker.",
    ),
    Rule(
        name="oasis_success_claim",
        severity="blocker",
        pattern=re.compile(r"\bOASIS[^.\\n]{0,80}\b(solved|successful|validated|strong|generaliz(?:e|ation))", re.I),
        description="OASIS must remain a stress-test limitation, not a success claim.",
    ),
    Rule(
        name="zero_shot_solved_claim",
        severity="blocker",
        pattern=re.compile(r"\b(pure\s+)?zero[- ]shot[^.\\n]{0,80}\b(solved|strong|validated|generaliz(?:e|ation))", re.I),
        description="Pure ADNI-to-AIBL zero-shot staging is not solved.",
    ),
    Rule(
        name="clinical_deployment_ready_claim",
        severity="blocker",
        pattern=re.compile(r"\b(deployment[- ]ready|clinically deployable|clinical deployment|clinical device|diagnostic device|medical device)", re.I),
        description="The package is a research prototype, not a clinical device.",
    ),
    Rule(
        name="diagnosis_or_patient_care_claim",
        severity="warning",
        pattern=re.compile(r"\b(standalone diagnosis|patient care|diagnostic recommendations|routine care|direct diagnostic use)", re.I),
        description="Clinical-use language should remain explicitly negated or prospective.",
    ),
]


def git_tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def is_under_roots(path: Path, roots: list[Path]) -> bool:
    path_text = path.as_posix()
    for root in roots:
        root_text = root.as_posix().rstrip("/")
        if path_text == root_text or path_text.startswith(f"{root_text}/"):
            return True
    return False


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in git_tracked_files():
        if not is_under_roots(path, paths):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.as_posix() in SKIP_PATHS:
            continue
        if not path.exists():
            continue
        files.append(path)
    return sorted(files)


def context_is_safe(lines: list[str], index: int) -> bool:
    start = max(0, index - 8)
    end = min(len(lines), index + 4)
    text = " ".join(lines[start:end]).lower()
    return any(cue in text for cue in SAFE_CUES)


def audit_file(path: Path) -> list[dict]:
    findings = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        lineno = index + 1
        for rule in RULES:
            if rule.pattern.search(line):
                safe = context_is_safe(lines, index)
                status = "allowed_safe_context" if safe and rule.unsafe_if_no_safe_cue else "flagged"
                findings.append(
                    {
                        "path": str(path),
                        "line": lineno,
                        "rule": rule.name,
                        "severity": rule.severity,
                        "status": status,
                        "text": line.strip(),
                        "description": rule.description,
                    }
                )
    return findings


def build_report(findings: list[dict], scanned_count: int) -> str:
    blockers = [item for item in findings if item["status"] == "flagged" and item["severity"] == "blocker"]
    warnings = [item for item in findings if item["status"] == "flagged" and item["severity"] == "warning"]
    allowed = [item for item in findings if item["status"] == "allowed_safe_context"]
    status = "pass" if not blockers else "fail"

    lines = [
        "# Claim Boundary Audit",
        "",
        f"- Status: **{status}**",
        f"- Files scanned: {scanned_count}",
        f"- Blocker findings: {len(blockers)}",
        f"- Warning findings: {len(warnings)}",
        f"- Allowed safe-context mentions: {len(allowed)}",
        "",
        "## Interpretation",
        "",
        "This audit scans public Git-tracked text files and flags unsupported positive claims around direct Braak validation, attention-only CAS biomarkers, OASIS success, pure zero-shot transfer, and clinical deployment readiness. Mentions are allowed when they appear in explicit negation, limitation, or claim-boundary language.",
        "",
    ]

    if blockers:
        lines += ["## Blockers", "", "| file | line | rule | text |", "|---|---:|---|---|"]
        for item in blockers:
            lines.append(f"| `{item['path']}` | {item['line']} | {item['rule']} | {item['text']} |")
        lines.append("")
    else:
        lines += ["## Blockers", "", "No blocker claim-boundary violations were detected.", ""]

    if warnings:
        lines += ["## Warnings", "", "| file | line | rule | text |", "|---|---:|---|---|"]
        for item in warnings:
            lines.append(f"| `{item['path']}` | {item['line']} | {item['rule']} | {item['text']} |")
        lines.append("")
    else:
        lines += ["## Warnings", "", "No warning-level positive clinical-use claims were detected.", ""]

    lines += [
        "## Allowed Safe-Context Examples",
        "",
        "| file | line | rule | text |",
        "|---|---:|---|---|",
    ]
    for item in allowed[:80]:
        lines.append(f"| `{item['path']}` | {item['line']} | {item['rule']} | {item['text']} |")
    if len(allowed) > 80:
        lines.append(f"| ... | ... | ... | {len(allowed) - 80} additional safe-context mentions omitted. |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roots", nargs="*", type=Path, default=SCAN_ROOTS)
    args = parser.parse_args()

    files = iter_files(args.roots)
    findings: list[dict] = []
    for path in files:
        findings.extend(audit_file(path))
    report = build_report(findings, len(files))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[saved] {args.output}")
    blockers = [item for item in findings if item["status"] == "flagged" and item["severity"] == "blocker"]
    print(f"[blockers] {len(blockers)}")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
