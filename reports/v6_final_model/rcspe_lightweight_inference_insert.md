# RC-SPE Lightweight Inference Insert

## Measured Deployment-Side Numbers

- Locked inference head: ARA-Net V6 RC-SPE subject-level probability ensemble.
- Scalar ensemble/calibration parameters: 10.
- Parameter composition: 6 base-stream weights, 3 class-specific offsets, and 1 temperature parameter.
- Raw float64 parameter storage: 80 bytes.
- JSON configuration file size: 1,663 bytes.
- Benchmark environment: local arm64 CPU, Python 3.9.6, NumPy 2.0.2.
- Benchmark workload: 4 scan-level probability rows aggregated into 3 subject-level endpoint units.
- Mean wrapper runtime: approximately 0.14 ms per batch.
- Median wrapper runtime: 0.133 ms per batch.
- P95 wrapper runtime: 0.157 ms per batch.
- Mean runtime per scan row: 0.035 ms.
- Mean runtime per subject unit: 0.046 ms.
- Approximate throughput: 28,788 scan rows/s.

These numbers describe the public research deployment wrapper and locked RC-SPE inference head. They do not include upstream MRI preprocessing, atlas-feature extraction, or the internal parameters of the six base models.

## Manuscript Paragraph

Unlike end-to-end CNN or Transformer architectures that require millions of neural-network parameters for image segmentation or classification, the locked ARA-Net deployment-side inference head is a parameter-light probability ensemble. The final RC-SPE head contains only 10 scalar ensemble/calibration parameters: six non-negative base-model stream weights, three class-specific offsets for CN/MCI/AD, and one temperature-scaling parameter. Stored as double-precision values, these parameters require 80 bytes; the full JSON configuration file is 1.663 kB. In a local CPU benchmark using Python 3.9.6 and NumPy 2.0.2 on an arm64 machine, the wrapper processed four scan-level probability rows and aggregated them into three subject-level endpoint units in approximately 0.14 ms on average, corresponding to approximately 0.035 ms per scan row and 0.046 ms per subject-level unit. This design makes the final inference wrapper computationally lightweight and suitable for reproducible research deployment after upstream MRI preprocessing and base-model probability generation. These measurements refer to the locked RC-SPE inference head and do not include upstream MRI feature extraction, GPU image processing, or the internal parameters of the base models.

## Short Version

The locked ARA-Net RC-SPE inference head is extremely lightweight, containing only 10 scalar ensemble/calibration parameters and a 1.663 kB configuration file. In a CPU benchmark, it required approximately 0.14 ms to process a four-scan probability batch and aggregate three subject-level predictions. These numbers describe the deployment-side probability ensemble, excluding upstream MRI preprocessing and base-model inference.
