# Open-Source And Deployment Plan

## What Is Open

- Final ensemble configuration.
- Research inference CLI.
- Research API and browser-based research console.
- Dockerfile.
- Figure-generation scripts.
- Aggregate reports and manuscript-supporting figures.
- Model card, data card, and validation protocol draft.

## What Is Not Open

- Raw ADNI/AIBL/OASIS/IXI data.
- Restricted clinical spreadsheets.
- Subject-level prediction files with dataset identifiers.
- Private server paths and raw preprocessing caches.

## CLI Deployment

```bash
python deployment/research_inference.py \
  --input-csv examples/probability_input_example.csv \
  --output examples/predictions_subject.csv \
  --unit subject
```

## API Deployment

```bash
python deployment/research_api.py --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Metadata endpoint:

```bash
curl http://localhost:8080/metadata
```

## Web Console

```bash
python deployment/research_api.py --host 127.0.0.1 --port 8080
```

Open [http://localhost:8080](http://localhost:8080). The browser console supports CSV upload, subject-level or scan-level prediction, confidence review, label filtering, and prediction CSV export.

## Docker Deployment

```bash
docker build -t aranet-research .
docker run --rm -p 8080:8080 aranet-research
```

## Clinical Boundary

This package is deployable as a research prototype. It is not a cleared or approved clinical device and should not be used for patient care.

For regulatory context and official FDA references, see `docs/REGULATORY_NOTES.md`.
