# Word Manuscript Rewrite Map

## Purpose

The uploaded Word manuscript (`/Users/mac/Downloads/ARA-Net_MedIA_Paper（完整版）.docx`) is still structurally an older attention-as-biomarker ARA-Net paper. The V6 repository package has moved to a locked subject-level atlas-guided multimodal rescue ensemble with external AIBL validation, IXI healthy specificity, OASIS stress-test limitation, and atlas structural neurodegeneration consistency. This map turns the Word audit into a section-by-section rewrite plan.

## Audit Result

Current Word audit: `word_manuscript_claim_audit.md`

- Paragraph/table units scanned: 346
- Blocker findings: 70
- Warning findings: 6
- Conclusion: the current Word manuscript cannot be safely resubmitted as a light edit. It needs a section-level rewrite.

## Rewrite Strategy

| Word area | current problem | action | source to use |
|---|---|---|---|
| Title | Frames paper as "Atlas-Guided Region Attention". | Replace with atlas-guided multimodal subject-level staging title. | `manuscript_v6_rewrite_package.md` |
| Abstract | Methods/results/conclusion are old attention-as-biomarker claims and old ADNI CV metrics. | Replace entire abstract. | V6 abstract in `manuscript_v6_full_draft.md` |
| Keywords/abbreviations | Includes CAS, RDI, attention biomarker discovery as central terms. | Remove CAS/RDI unless only historical; add subject-level, external heldout, atlas structural neurodegeneration consistency. | `final_submission_closure_packet.md` |
| Introduction | Research questions ask whether attention aligns with Braak. | Rewrite around three reviewer failures: external classification, invalid CAS, non-significant Braak. | V6 Introduction in `manuscript_v6_full_draft.md` |
| Contributions | C1/C2 are attention model and Attention-as-Biomarker framework. | Replace with leakage-free subject-level splits, locked AIBL endpoint, IXI specificity, comparators, bootstrap, error analysis, structural proxy. | V6 contributions in `manuscript_v6_full_draft.md` |
| Cohorts | Old Table 1 says three datasets and OASIS interpretability generalization. | Replace with ADNI/AIBL/OASIS/IXI subject-level split table. | `docs/DATA_CARD.md`, V6 Methods |
| Model methods | Old sections detail CNN attention architecture and attention losses. | Shorten as legacy baseline or remove; primary method is final rescue probability ensemble. | `deployment/final_ensemble_config.json`, V6 Methods |
| Attention-as-Biomarker section | Central invalid CAS/RDI/disease-gradient framework. | Remove as primary method. Replace with atlas structural neurodegeneration consistency analysis. | V6 Biological Consistency section |
| Results classification | Old ADNI 30-run CV BAcc 0.671 is no longer the main result. | Lead with AIBL heldout subject-level BAcc 0.833 and IXI CN retention 1.000. | `final_rescue_model_lock_report.md` |
| Attention figures/tables | Figures 2-6 and Tables 4-5 focus attention/CAS. | Replace with final figure blueprint and error/stability/biological consistency figures. | `final_figure_blueprint.md` |
| OASIS | Old text frames OASIS as cross-dataset interpretability generalization. | Keep only as stress-test limitation: BAcc 0.334, AD recall 0.000. | `core_reviewer_evidence_matrix.md` |
| Discussion | Old discussion defends attention as clinically verifiable and clinically desirable. | Rewrite around domain-adapted external evidence, MCI/AD error boundary, OASIS limitation, MRI proxy not direct Braak, research-use boundary. | V6 Discussion |
| Clinical translation | Old text leans toward clinical adoption/deployment. | Replace with research prototype and prospective-validation requirements. | `deployment_code_clinical_response.md`, `clinical_translation_roadmap.md` |
| Code availability | Old code statement promises attention module and Attention-as-Biomarker. | Replace with public V6 code, deployment wrapper, aggregate reports, no raw data/checkpoints. | `README.md`, `public_release_manifest.md` |

## High-Risk Word Units From Audit

These unit numbers should be treated as replacement anchors, not as sentences to lightly polish.

| unit range | main issue | rewrite action |
|---|---|---|
| 1, 13-17 | Title, abstract, keywords, abbreviations still define ARA-Net as attention/CAS biomarker work. | Replace front matter. |
| 21-31 | Introduction and contributions frame attention weights as Braak/neuropathology validation target. | Replace introduction and contribution list. |
| 35-46 | Cohort/model framework still uses old three-dataset/attention model protocol. | Replace Table 1 and Figure 1 text. |
| 55-89 | Attention layer/loss and Attention-as-Biomarker methods dominate Methods. | Remove or demote to historical baseline; insert V6 ensemble and structural proxy methods. |
| 107-148 | Old figures/tables are attention/CAS/error-conditioned interpretability artifacts. | Replace with final figure blueprint and final classification/error/stability/biological consistency tables. |
| 162-197 | Discussion claims attention is biologically/clinically meaningful and discusses Braak as a failed alignment. | Replace Discussion with V6 claim-boundary version. |
| 201 | Code statement describes attention module and Attention-as-Biomarker release. | Replace with public V6 research package statement. |
| 300-302 | Supplementary old dataset table frames OASIS as cross-dataset interpretability. | Replace with OASIS stress-test limitation. |

## Replacement Front-Matter Text

### Title

Atlas-guided multimodal Alzheimer's disease staging with locked external subject-level validation and structural neurodegeneration consistency

### Keywords

Alzheimer's disease; structural MRI; atlas-guided multimodal learning; external validation; subject-level staging; neurodegeneration consistency; open-source research prototype

### Abbreviation Changes

Remove CAS and RDI from the main abbreviation list unless the manuscript includes them only as historical terms being explicitly removed. Add:

- AIBL, Australian Imaging, Biomarkers and Lifestyle study
- IXI, Information eXtraction from Images dataset
- OASIS, Open Access Series of Imaging Studies
- CN, cognitively normal
- MCI, mild cognitive impairment
- AD, Alzheimer's disease
- BAcc, balanced accuracy

## Replacement Main Results

Use this as the Results lead:

> The locked final subject-level rescue ensemble achieved AIBL heldout accuracy of 0.903, balanced accuracy of 0.833, macro AUC of 0.937, and AD-vs-CN AUC of 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95% confidence intervals were 0.759-0.899 for balanced accuracy, 0.531-0.839 for MCI recall, and 0.710-0.966 for AD recall. The same model retained all IXI healthy controls as CN. OASIS transfer remained weak and is reported as a stress-test limitation rather than successful external validation.

## Replacement Biological Claim

Use this as the biological-validation wording:

> The original attention-only CAS was removed because it did not provide valid evidence that attention weights were biomarkers. We instead evaluated atlas structural neurodegeneration consistency in a priori AD-relevant regions. In AIBL heldout, the AD-key volume consistency score was 0.510 compared with a uniform regional null of 0.286, with bootstrap CI 0.479-0.526 and permutation p=0.026. This supports an MRI neurodegeneration proxy, not attention as a biomarker and not direct Braak-stage validation.

## Replacement OASIS Wording

Use this wording:

> OASIS was retained only as an external stress test and was not used for final tuning. The final subject-level model achieved OASIS accuracy 0.586, balanced accuracy 0.334, macro AUC 0.554, AD-vs-CN AUC 0.371, and CN/MCI/AD recall 0.966/0.034/0.000. This result demonstrates an unresolved transfer limitation and should not be interpreted as successful OASIS validation.

## Replacement Clinical-Use Wording

Use this wording:

> The software is released as an open-source research prototype for retrospective evaluation and future prospective validation. It is not a medical device, is not cleared or approved for clinical use, and is not intended for standalone diagnosis or patient-care decisions. Clinical use would require prospective multi-center testing, scanner/protocol robustness assessment, local calibration, workflow integration, uncertainty reporting, cybersecurity review, and regulatory assessment.

## Word Rewrite Acceptance Gate

After the Word manuscript is rewritten, rerun:

```bash
/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/audit_word_manuscript_claims.py \
  --docx /path/to/rewritten.docx \
  --output reports/v6_final_model/word_manuscript_claim_audit_rewritten.md
```

The rewritten Word manuscript should have:

- 0 blocker findings.
- No attention/CAS biomarker claims except in explicitly historical removal language.
- No direct Braak-stage validation claim.
- No OASIS success claim.
- No clinical-deployment readiness claim.
- The final subject-level rescue ensemble named as the locked primary model.
