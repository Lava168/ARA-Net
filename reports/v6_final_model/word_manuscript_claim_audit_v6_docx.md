# Word Manuscript Claim Audit

- Source DOCX: `reports/v6_final_model/ARA-Net_V6_full_manuscript_draft.docx`
- Paragraph/table-text units scanned: 122
- Status: **pass**
- Blocker findings: 0
- Warning findings: 0
- Allowed safe-context mentions: 12

## Interpretation

This report audits the Word manuscript, not only the public repository Markdown. Blockers are stale or unsupported manuscript claims that should be removed before resubmission.

## Blockers

No blockers were detected.

## Warnings

No warnings were detected.

## Allowed Safe-Context Examples

| unit | rule | text |
|---:|---|---|
| 7 | oasis_success_claim | Conclusion: The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance, preserved IXI healthy specificity, and atlas-region structural neurodegeneration consistency. It... |
| 7 | zero_shot_success_claim | Conclusion: The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance, preserved IXI healthy specificity, and atlas-region structural neurodegeneration consistency. It... |
| 7 | clinical_ready_claim | Conclusion: The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance, preserved IXI healthy specificity, and atlas-region structural neurodegeneration consistency. It... |
| 29 | oasis_success_claim | The primary endpoint was AIBL locked heldout subject-level CN/MCI/AD staging. The specificity endpoint was IXI healthy CN retention. OASIS was retained to test the boundary of cross-cohort transfer, not to support a success claim. |
| 58 | clinical_ready_claim | The software is released as an open-source research prototype for retrospective evaluation and future prospective validation. It is not a medical device, is not cleared or approved for clinical use, and is not intended for standalone diagnosis or patient-ca... |
| 84 | zero_shot_success_claim | The final model is strongest when interpreted as a domain-adapted external AD staging framework. AIBL adaptation data were used for model fitting and calibration, but AIBL heldout endpoint units remained locked and were not used for final endpoint evaluatio... |
| 87 | clinical_ready_claim | Several limitations remain. OASIS transfer is not solved and should be treated as a stress-test failure requiring larger and cleaner external cohorts. The internal subject-level calibration pattern remains modest, especially for CN specificity within the in... |
| 89 | oasis_success_claim | This revised work is a substantive rebuild of the original ARA-Net study. It replaces unsupported interpretability-centered claims with locked external subject-level classification, healthy negative-control specificity, bootstrap uncertainty, MCI/AD error a... |
| 89 | zero_shot_success_claim | This revised work is a substantive rebuild of the original ARA-Net study. It replaces unsupported interpretability-centered claims with locked external subject-level classification, healthy negative-control specificity, bootstrap uncertainty, MCI/AD error a... |
| 89 | clinical_ready_claim | This revised work is a substantive rebuild of the original ARA-Net study. It replaces unsupported interpretability-centered claims with locked external subject-level classification, healthy negative-control specificity, bootstrap uncertainty, MCI/AD error a... |
| 91 | oasis_success_claim | Figure 1. Revised ARA-Net V6 study framework. The revised workflow uses subject-level cohort construction, atlas-guided multimodal feature extraction, a locked rescue probability ensemble, and subject-level probability averaging. The primary endpoint is loc... |
| 98 | clinical_ready_claim | The software is a research prototype. It is not a medical device and is not cleared or approved for clinical use. |
