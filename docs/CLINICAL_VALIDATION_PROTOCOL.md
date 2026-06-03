# Prospective Clinical Validation Protocol Draft

## Objective

Evaluate whether the ARA-Net V6 research prototype can support CN/MCI/AD decision-support workflows under prospective, multi-center conditions.

## Study Type

Prospective observational validation study. The model is run silently or as a non-interventional research output until safety, calibration, workflow, and regulatory requirements are satisfied.

## Intended Clinical Question

Can an atlas-guided multimodal AD staging system provide calibrated subject-level CN/MCI/AD risk estimates that support specialist review without replacing clinician judgment?

## Inclusion Criteria

- Adults undergoing structural brain MRI for cognitive evaluation or research enrollment.
- Availability of required clinical variables for the selected model configuration.
- Consent and local ethics approval for retrospective/prospective model evaluation.

## Exclusion Criteria

- Major non-AD neurological pathology that invalidates the intended staging task.
- MRI quality insufficient for segmentation or atlas-feature extraction.
- Missing data patterns outside the validated model envelope.

## Primary Endpoint

Subject-level balanced accuracy for CN/MCI/AD staging against adjudicated clinical diagnosis or expert consensus.

## Secondary Endpoints

- Per-class recall, especially MCI and AD.
- AD-vs-CN AUC.
- False impairment rate among clinically normal controls.
- Calibration error and decision-curve analysis.
- Site/scanner subgroup performance.
- Human-AI team performance, if outputs are shown to clinicians.

## Safety Monitoring

- Track AD-to-CN errors separately as a high-risk failure mode.
- Track CN-to-AD false impairment separately as a patient-anxiety and downstream-testing risk.
- Require uncertainty flagging for low-margin predictions.
- Monitor performance drift by site, scanner, and time.

## Deployment Gate

The system should not be used for clinical care until prospective validation, site calibration, quality-management procedures, cybersecurity review, and regulatory assessment have been completed.

## Regulatory Note

If the software is intended to provide diagnostic recommendations from MRI/clinical data, it may fall within medical-device software or Software as a Medical Device frameworks, depending on jurisdiction and intended use. Formal regulatory assessment is required before clinical deployment.

See `docs/REGULATORY_NOTES.md` for FDA references and the proposed translation boundary.
