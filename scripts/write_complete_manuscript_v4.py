#!/usr/bin/env python3
"""Assemble a complete v4 manuscript draft with tables, figures, and captions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(read_text(path))


def md_image(path: str, label: str) -> str:
    return f"![{label}]({path})"


def figure_block(fig_dir: Path, stem: str, caption: str, label: str) -> str:
    png = fig_dir / f"{stem}.png"
    pdf = fig_dir / f"{stem}.pdf"
    return "\n".join(
        [
            f"### {label}",
            "",
            md_image(str(png), label),
            "",
            f"**Caption:** {caption}",
            "",
            f"PDF version: `{pdf}`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    reports = args.reports_dir
    skeleton = read_text(reports / "full_manuscript_skeleton.md").rstrip()
    table_dir = reports / "tables"
    fig_dir = reports / "figures"
    captions = load_json(fig_dir / "figure_captions.json")

    tables = {
        "Table 1. Cohort and split summary": read_text(table_dir / "table1_cohort_splits.md").rstrip(),
        "Table 2. Main classification results": read_text(table_dir / "table2_classification.md").rstrip(),
        "Table 3. Replicate stability for key hybrid candidates": read_text(table_dir / "table3_replicates.md").rstrip(),
        "Table 4. Atlas neurodegeneration consistency validation": read_text(table_dir / "table4_neurodegeneration.md").rstrip(),
    }

    figure_stems = [
        ("figure1_revised_study_design", "Figure 1. Revised study design"),
        ("figure2_external_classification_improvement", "Figure 2. External classification improvement"),
        ("figure3_aibl_confusion_matrices", "Figure 3. AIBL heldout confusion matrices"),
        ("figure4_neurodegeneration_consistency", "Figure 4. Neurodegeneration consistency score"),
        ("figure5_oasis_stress_test", "Figure 5. Honest OASIS stress-test limitation"),
    ]

    lines = [
        "# Complete V4 Manuscript Draft",
        "",
        "This draft is assembled from the v4 rebuild package. It is intended as the working source for the Word manuscript rewrite, not as the final formatted journal file.",
        "",
        "## Claim Boundaries For The Rewrite",
        "",
        "Allowed claims:",
        "",
        "- The revised atlas-guided multimodal model achieved strong domain-adapted performance on a locked AIBL heldout split.",
        "- IXI was used as a healthy external negative-control cohort to estimate false impairment.",
        "- The original attention-only CAS was replaced by a structural atlas neurodegeneration consistency score.",
        "- The biological validation supports an MRI neurodegeneration proxy, not direct Braak staging.",
        "",
        "Claims to avoid:",
        "",
        "- Do not claim pure ADNI-to-AIBL zero-shot staging is solved.",
        "- Do not claim direct Braak-stage validation.",
        "- Do not claim attention mass alone is a validated biomarker.",
        "- Do not hide or soften the weak OASIS transfer result.",
        "- Do not frame the model as ready for clinical deployment.",
        "",
        "## Manuscript Text",
        "",
        skeleton,
        "",
        "## Manuscript Tables",
        "",
    ]

    for title, table in tables.items():
        lines += [f"### {title}", "", table, ""]

    lines += ["## Manuscript Figures", ""]
    for stem, label in figure_stems:
        lines += [figure_block(fig_dir, stem, captions[stem], label), ""]

    lines += [
        "## Figure Caption Text For Manuscript",
        "",
        "**Figure 1. Revised study design.** "
        + captions["figure1_revised_study_design"],
        "",
        "**Figure 2. External classification improvement.** "
        + captions["figure2_external_classification_improvement"],
        "",
        "**Figure 3. AIBL heldout confusion matrices.** "
        + captions["figure3_aibl_confusion_matrices"],
        "",
        "**Figure 4. Neurodegeneration consistency score.** "
        + captions["figure4_neurodegeneration_consistency"],
        "",
        "**Figure 5. OASIS stress-test limitation.** "
        + captions["figure5_oasis_stress_test"],
        "",
        "## Submission Integration Checklist",
        "",
        "1. Replace the old ARA-Net title/abstract with the v4 atlas-guided multimodal framing.",
        "2. Insert Table 1 in Methods after cohort/split description.",
        "3. Insert Table 2 in Results after the external classification paragraph.",
        "4. Insert Table 3 near the stability/sensitivity analysis.",
        "5. Insert Table 4 near the neurodegeneration consistency section.",
        "6. Insert Figures 1-4 in the main manuscript if space allows.",
        "7. Keep Figure 5 in the main manuscript or supplement, but do not omit the OASIS limitation from the text.",
        "8. Remove direct Braak-staging wording and all claims that CAS validates attention as a biomarker.",
        "9. Fix all old equation boxes, broken figure references, and `Error! Reference source not found` artifacts before submission.",
        "10. Add reviewer-accessible code/data manifest instructions before resubmission.",
        "",
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {args.output_md}")


if __name__ == "__main__":
    main()
