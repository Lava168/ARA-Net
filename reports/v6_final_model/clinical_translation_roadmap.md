# Clinical Translation Roadmap

## Target Statement

The safest target is:

**An open-source, deployable research prototype with a defined pathway toward clinical decision-support validation.**

Do not claim:

**A clinically deployable diagnostic device ready for routine care.**

## Three Levels

| level | can we claim now? | what it means | next work |
|---|---|---|---|
| Open-source | Yes | Release code, split manifests, evaluation scripts, figure scripts, model-card documentation | Clean GitHub repo, README, license, environment file, reproducibility instructions |
| Deployable research prototype | Yes, after packaging | A runnable CLI/API/Docker pipeline for research inference and retrospective validation | Docker, CLI, sample input/output, model weights, preprocessing notes, logging |
| Clinical usability | Not yet as a claim | Potential decision-support use after prospective validation | IRB/prospective study, multi-site validation, calibration, DICOM/PACS workflow, uncertainty reporting, regulatory review |

## Reviewer-Safe Manuscript Wording

The revised framework is released as an open-source, deployable research prototype rather than as a clinical device. The software is intended to support reproducible retrospective evaluation and future prospective validation. Direct clinical use would require additional multi-center prospective testing, local calibration, workflow integration, uncertainty reporting, cybersecurity review, and regulatory assessment.

## What To Build Next

1. **Open-source release package**
   - `README.md`
   - `LICENSE`
   - `environment.yml` or `requirements.txt`
   - `Dockerfile`
   - `MODEL_CARD.md`
   - `DATA_CARD.md`
   - `REPRODUCE.md`
   - split manifests without protected data

2. **Research deployment pipeline**
   - CLI command for inference on preprocessed subject features
   - Optional REST API for demonstration
   - output JSON with class probabilities, predicted label, uncertainty, and atlas-region summary
   - no clinical diagnosis language in the UI/output

3. **Clinical translation package**
   - prospective validation protocol
   - inclusion/exclusion criteria
   - primary and secondary endpoints
   - failure-mode analysis
   - calibration and monitoring plan
   - clinician-facing interpretability report

## FDA / Regulatory Boundary

In the United States, FDA materials distinguish non-device clinical decision support from regulated software functions depending on intended use, user, data inputs, and whether the clinician can independently review the basis for the recommendation. AI/ML medical-device software is also treated under FDA's software-as-a-medical-device framework. Therefore, a model intended to provide diagnostic recommendations from MRI/clinical data may require regulatory assessment before clinical deployment.

## Practical Goal For This Paper

For this manuscript, the strongest and safest claim is:

**We provide open-source research software and a deployable retrospective validation pipeline, and we define the additional prospective and regulatory work required before clinical use.**

This lets the paper answer the reviewer without overclaiming.
