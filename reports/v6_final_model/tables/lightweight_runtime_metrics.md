# ARA-Net Lightweight Runtime Metrics

| Item | Value |
|---|---:|
| Base probability streams | 6 |
| RC-SPE scalar parameters | 10 |
| Raw float64 parameter storage | 80 bytes |
| JSON config size | 1663 bytes |
| CPU mean runtime / batch | 0.139 ms |
| CPU median runtime / batch | 0.134 ms |
| CPU p95 runtime / batch | 0.162 ms |
| CPU mean runtime / scan row | 0.035 ms |
| CPU mean runtime / subject unit | 0.046 ms |
| CPU throughput | 28705 scan rows/s |
| GPU inference | Not applicable for the NumPy RC-SPE probability head |

## Evaluation Context

- Primary cohort: AIBL locked heldout.
- AIBL heldout accuracy: 0.903.
- AIBL heldout balanced accuracy: 0.833.
- AIBL heldout macro AUC: 0.937.

## Claim Boundary

Metrics cover the public RC-SPE probability head only; raw MRI preprocessing, atlas extraction, and base-model artifact size are excluded.
Do not describe these measurements as full raw-MRI end-to-end runtime.

## Reproducibility

- Python: 3.9.6.
- Platform: macOS-15.6.1-arm64-arm-64bit.
- NumPy: 2.0.2.
- Benchmark iterations: 5000.
- Workload: 4 scan rows aggregated into 3 subject units.
