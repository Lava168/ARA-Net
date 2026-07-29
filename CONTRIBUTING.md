# Contributing

Thank you for interest in ARA-Net / atlas-guided AD staging.

## Scope

- Prefer changes that improve **reproducibility**, **documentation clarity**, or
  **public smoke-test** quality.
- Do not commit restricted MRI volumes, row-level subject identifiers from
  ADNI/AIBL/OASIS, or private checkpoints.
- Keep NeuroGate / attention-training metrics out of ARA-Net RC-SPE claims.

## Development

```bash
python -m pip install -r requirements.txt
pytest -q
bash scripts/reproduce_paper.sh
```

## Pull requests

1. Open against `master`.
2. Describe what manuscript claim or smoke-test path you touched.
3. Confirm `pytest` passes.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
