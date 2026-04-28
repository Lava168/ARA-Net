# Clinical Integration Guide

This document outlines practical ways to deploy ARA-Net in real clinical or
hospital information-system settings, including HIS, EMR/EHR, RIS, PACS, and
research data platforms.

ARA-Net itself is a research model. In production it should be wrapped by a
clinical integration layer that handles DICOM retrieval, preprocessing,
inference orchestration, authentication, audit logging, and structured result
delivery. The model output should be treated as clinical decision support, not
as a standalone diagnosis.

## Core Deployment Pattern

Most clinical deployments use the same high-level pipeline:

```text
HIS / EMR order
    |
    v
RIS / PACS imaging workflow
    |
    v
DICOM T1-weighted MRI retrieval
    |
    v
Preprocessing + FastSurfer / FreeSurfer parcellation
    |
    v
ARA-Net inference service
    |
    v
Structured result delivery to EMR / PACS / research database
```

ARA-Net expects:

- A preprocessed T1-weighted MRI volume, resampled to `96 x 112 x 96`.
- A corresponding 21-region atlas segmentation mask.
- The model returns 3-class probabilities (`CN`, `MCI`, `AD`) and a 21-region
  attention profile that can be visualized as region-level evidence.

## Option 1: FHIR-Native EMR Integration

Use this option when the hospital EMR supports HL7 FHIR R4/R5 APIs.

### Data Flow

```text
EMR creates ServiceRequest
  -> PACS stores ImagingStudy
  -> AI gateway reads ImagingStudy / DICOM StudyInstanceUID
  -> ARA-Net inference
  -> EMR receives DiagnosticReport + Observations
```

### Recommended FHIR Resources

| Resource | Purpose |
|---|---|
| `Patient` | Patient identifier, demographics, local MRN mapping |
| `ServiceRequest` | MRI order or AI-assessment request |
| `ImagingStudy` | Link to DICOM study, accession number, StudyInstanceUID |
| `DiagnosticReport` | Main AI result container |
| `Observation` | Class probabilities, risk score, region-attention values |
| `Provenance` | Model version, software version, runtime, responsible system |

### Example DiagnosticReport Payload

```json
{
  "resourceType": "DiagnosticReport",
  "status": "preliminary",
  "code": {
    "text": "ARA-Net Alzheimer's disease MRI decision-support result"
  },
  "subject": {
    "reference": "Patient/12345"
  },
  "imagingStudy": [
    {
      "reference": "ImagingStudy/1.2.840.example.study"
    }
  ],
  "conclusion": "ARA-Net predicts MCI with probability 0.61. This result is for clinical decision support and must be interpreted by a qualified clinician.",
  "result": [
    { "reference": "Observation/aranet-prob-cn" },
    { "reference": "Observation/aranet-prob-mci" },
    { "reference": "Observation/aranet-prob-ad" },
    { "reference": "Observation/aranet-region-attention" }
  ]
}
```

### Best Fit

- Modern EMR platforms with FHIR gateways.
- Sites that need structured, queryable AI output.
- Multi-system integration where EMR, PACS, and audit systems are separate.

## Option 2: HL7 v2 ORU Result Reporting

Use this option for hospitals with mature HL7 v2 interfaces but limited FHIR
support. This is common in legacy HIS/EMR environments.

### Data Flow

```text
PACS study-complete event
  -> Integration engine, e.g. Mirth / Rhapsody / Cloverleaf
  -> AI inference service
  -> HL7 v2 ORU^R01 message
  -> EMR result inbox
```

### Recommended OBX Segments

| OBX Field | Example |
|---|---|
| Predicted class | `MCI` |
| CN probability | `0.22` |
| MCI probability | `0.61` |
| AD probability | `0.17` |
| Top attended regions | `Right amygdala; bilateral hippocampus; ventricles` |
| Model version | `ARA-Net 0.1.0 / commit hash` |
| Input reference | Accession number + StudyInstanceUID |

### Best Fit

- Legacy HIS/EMR systems.
- Sites where radiology results already flow through ORU messages.
- Hospitals using an integration engine as the central interoperability hub.

## Option 3: PACS / DICOMweb-Centered Workflow

Use this option when the AI pipeline is triggered by newly archived imaging
studies rather than by EMR orders.

### Data Flow

```text
PACS receives DICOM study
  -> QIDO-RS finds eligible T1 MRI series
  -> WADO-RS retrieves DICOM instances
  -> DICOM-to-NIfTI conversion
  -> FastSurfer / FreeSurfer segmentation
  -> ARA-Net inference
  -> result returned to PACS as DICOM SR / secondary capture / web link
```

### Interfaces

| Interface | Role |
|---|---|
| DICOM QIDO-RS | Search studies and series |
| DICOM WADO-RS | Retrieve DICOM instances |
| DICOM STOW-RS | Store derived result objects back to PACS |
| DICOM SR | Structured AI result for PACS-native viewing |

### Best Fit

- Radiology-first workflow.
- PACS viewers that can display DICOM SR or derived AI overlays.
- Institutions that prefer not to write model output directly into EMR.

## Option 4: PACS Viewer Sidebar / AI Plugin

Use this option when the hospital wants a lightweight visual decision-support
panel next to the radiologist's viewer.

### Data Flow

```text
Radiologist opens MRI study in PACS viewer
  -> viewer sends StudyInstanceUID to AI plugin
  -> plugin queries inference-result service
  -> sidebar displays probabilities and region-attention ranking
```

### Suggested UI Elements

- Probability bars for `CN`, `MCI`, and `AD`.
- Top-5 attended anatomical regions.
- 21-region attention bar plot.
- Model version and inference timestamp.
- Disclaimer: "Decision-support output; not a standalone diagnosis."

### Best Fit

- Early clinical evaluation.
- Radiologist-facing interpretability review.
- Sites where EMR modification is slow or not immediately available.

## Option 5: Research Registry / Data Warehouse Integration

Use this option for retrospective cohort analysis, prospective registries, or
clinical research workflows.

### Data Flow

```text
Research data warehouse
  -> de-identified MRI cohort export
  -> batch preprocessing and inference
  -> CSV / Parquet / SQL tables
  -> statistical analysis and dashboard
```

### Suggested Output Table

| Column | Description |
|---|---|
| `study_uid_hash` | De-identified study identifier |
| `model_version` | ARA-Net version or git commit |
| `predicted_class` | `CN`, `MCI`, or `AD` |
| `prob_cn`, `prob_mci`, `prob_ad` | Class probabilities |
| `attention_region_01` ... `attention_region_21` | Region-attention vector |
| `preprocess_status` | Success / failure / quality-control flag |
| `inference_timestamp` | Runtime audit timestamp |

### Best Fit

- Retrospective validation.
- Multi-center generalization studies.
- Registry dashboards where direct EMR write-back is not required.

## Option 6: Private AI Gateway / Microservice Deployment

Use this option when the model is deployed inside the hospital network and
called by multiple internal systems.

### Recommended Components

```text
API gateway
  -> authentication and rate limiting
  -> job queue
  -> DICOM retrieval worker
  -> preprocessing worker
  -> GPU inference worker
  -> result store
  -> FHIR / HL7 / PACS adapter
```

### Minimal REST API

```http
POST /v1/aranet/jobs
Content-Type: application/json

{
  "study_instance_uid": "1.2.840.example.study",
  "accession_number": "MR202604280001",
  "callback_url": "https://emr.example.org/fhir"
}
```

```json
{
  "job_id": "aranet-20260428-0001",
  "status": "queued"
}
```

### Result Schema

```json
{
  "job_id": "aranet-20260428-0001",
  "status": "completed",
  "input": {
    "study_instance_uid": "1.2.840.example.study",
    "series_instance_uid": "1.2.840.example.series"
  },
  "model": {
    "name": "ARA-Net",
    "version": "0.1.0",
    "git_commit": "unknown"
  },
  "prediction": {
    "class": "MCI",
    "probabilities": {
      "CN": 0.22,
      "MCI": 0.61,
      "AD": 0.17
    }
  },
  "attention": {
    "num_regions": 21,
    "top_regions": [
      { "region": "Right-Amygdala", "weight": 0.083 },
      { "region": "Left-Hippocampus", "weight": 0.071 }
    ]
  }
}
```

### Best Fit

- Hospital-wide deployment.
- Multiple consumers: EMR, PACS viewer, research registry, QA dashboard.
- Environments requiring strict access control, logging, and version tracking.

## Recommended Path for a First Hospital Pilot

For a first real-world pilot, the safest sequence is:

1. Start with the **PACS / DICOMweb-centered workflow** to avoid touching EMR
   clinical documentation too early.
2. Add the **AI gateway / microservice layer** so preprocessing and inference
   are asynchronous and auditable.
3. Display results in a **PACS viewer sidebar** or research dashboard for
   clinician review.
4. After local validation, add **FHIR DiagnosticReport / Observation** write-back
   to the EMR.
5. Use **HL7 v2 ORU** only when the site does not support FHIR or when the EMR
   already expects radiology-related AI results in an ORU feed.

## Safety, Governance, and Compliance Notes

- Keep PHI out of application logs unless the deployment has an approved audit
  and retention policy.
- Store the model version, git commit, preprocessing version, input Study UID,
  and inference timestamp for every result.
- Use asynchronous processing; FastSurfer / FreeSurfer segmentation can dominate
  total runtime.
- Add quality-control checks for sequence type, voxel spacing, failed
  segmentation, empty regions, and out-of-distribution inputs.
- Validate locally before clinical use, especially across scanner vendors,
  acquisition protocols, and patient demographics.
- Include a visible disclaimer: ARA-Net provides decision-support information and
  should not replace clinician interpretation.

## Repository Scope

This repository provides:

- The ARA-Net model architecture.
- Training and evaluation code.
- Preprocessing helpers for cached research data.
- Shape and loss smoke tests.
- A reference description of how clinical integration can be performed.

This repository does not provide:

- A certified medical device.
- A production DICOM router.
- A production FHIR or HL7 interface engine.
- Hospital-specific authentication, authorization, or audit infrastructure.

