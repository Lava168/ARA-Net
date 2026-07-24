# Algorithm Innovation Evidence: RC-SPE

## Method Name

**RC-SPE: Risk-Constrained Subject-level Probability Ensemble.**

This evidence package uses private row-level prediction files on the server but exports only aggregate metrics, tables, and figures.

## Key Findings

- Final RC-SPE subject-level AIBL heldout BAcc=0.833, MCI recall=0.686, AD recall=0.852, AD-to-CN errors=0, IXI CN retention=1.000.
- Best single base model selected by tuning score: atlas+bio HGB; AIBL BAcc=0.756, IXI CN retention=0.997.
- Equal log-pooling without learned weights/offsets/temperature achieved AIBL BAcc=0.648, showing the locked final method adds more than simple pooling.
- The MCI-rescue profile raises MCI recall to 0.886 but reduces IXI CN retention to 0.959, supporting the risk-constrained final selection.

## Algorithmic Ablation Table

| Variant | AIBL BAcc | AIBL AUC | MCI recall | AD recall | AD->CN | IXI CN retention | ECE | NLL | OASIS stress BAcc |
|---|---|---|---|---|---|---|---|---|---|
| Best single base model | 0.756 | 0.945 | 0.571 | 0.741 | 2 | 0.997 | 0.122 | 0.416 | 0.310 |
| Arithmetic mean ensemble | 0.711 | 0.951 | 0.400 | 0.741 | 3 | 1.000 | 0.112 | 0.400 | 0.333 |
| Equal log-pooling | 0.648 | 0.954 | 0.171 | 0.778 | 5 | 1.000 | 0.094 | 0.400 | 0.333 |
| Final weights only | 0.815 | 0.955 | 0.657 | 0.815 | 0 | 1.000 | 0.136 | 0.366 | 0.333 |
| Weights + offsets | 0.823 | 0.926 | 0.657 | 0.852 | 0 | 1.000 | 0.219 | 0.475 | 0.310 |
| Weights + temperature | 0.815 | 0.954 | 0.657 | 0.815 | 0 | 1.000 | 0.052 | 0.298 | 0.333 |
| Full RC-SPE (scan-level) | 0.810 | 0.940 | 0.660 | 0.805 | 0 | 1.000 | 0.081 | 0.318 | 0.334 |
| Full RC-SPE (majority vote) | 0.810 | 0.870 | 0.686 | 0.778 | 0 | 1.000 | 0.099 | 2.579 | 0.334 |
| Full RC-SPE (subject-level) | 0.833 | 0.937 | 0.686 | 0.852 | 0 | 1.000 | 0.078 | 0.320 | 0.334 |
| MCI-rescue profile | 0.811 | 0.952 | 0.886 | 0.593 | 0 | 0.959 | 0.156 | 0.421 | 0.367 |
| AD-rescue profile | 0.698 | 0.930 | 0.143 | 0.963 | 0 | 1.000 | 0.293 | 0.610 | 0.328 |

## Calibration Table

| Variant | NLL | Brier | ECE | Acc | BAcc |
|---|---|---|---|---|---|
| Best single base model | 0.416 | 0.211 | 0.122 | 0.866 | 0.756 |
| Arithmetic mean ensemble | 0.400 | 0.205 | 0.112 | 0.866 | 0.711 |
| Equal log-pooling | 0.400 | 0.216 | 0.094 | 0.833 | 0.648 |
| Final weights only | 0.366 | 0.179 | 0.136 | 0.903 | 0.815 |
| Weights + offsets | 0.475 | 0.235 | 0.219 | 0.898 | 0.823 |
| Weights + temperature | 0.298 | 0.152 | 0.052 | 0.903 | 0.815 |
| Full RC-SPE (subject-level) | 0.320 | 0.160 | 0.078 | 0.903 | 0.833 |

## Leave-One-Model-Out Sensitivity

| Dropped model | AIBL BAcc | MCI recall | AD recall | AD->CN | IXI CN retention |
|---|---|---|---|---|---|
| atlas+bio HGB | 0.835 | 0.686 | 0.852 | 0 | 1.000 |
| atlas+clinical HGB | 0.835 | 0.686 | 0.852 | 0 | 1.000 |
| clinical+bio RF | 0.823 | 0.657 | 0.852 | 0 | 1.000 |
| clinical HGB | 0.823 | 0.657 | 0.852 | 0 | 0.997 |
| clinical RF | 0.833 | 0.686 | 0.852 | 0 | 1.000 |
| cascade RF-logreg | 0.833 | 0.686 | 0.852 | 0 | 1.000 |

## Manuscript Insert

The final algorithm should be described as a risk-constrained subject-level probability ensemble rather than as a generic model average. The ablation demonstrates the individual contribution of probability pooling, learned non-negative weights, class offsets, temperature scaling, and subject-level probability averaging. The risk curve shows why the final locked profile was preferred over a high-MCI-recall profile: the latter improves MCI recall but increases false impairment predictions in IXI healthy controls.
