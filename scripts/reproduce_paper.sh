#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Reproduce the manuscript-reported ARA-Net benchmark.
#
# Manuscript §2.5: 6 random seeds × 5 stratified subject-level folds = 30 runs
#                  AdamW (lr 5e-4 → 1e-6 cosine), weight-decay 1e-3,
#                  gradient clipping 1.0, 5-epoch linear warm-up, 80 epochs.
#
# Usage (from repository root):
#   bash scripts/reproduce_paper.sh /path/to/cache_real
#
# Notes:
#   * `DATA_ROOT` should contain `cache_real/` populated by
#     `chapter1_foundation/preprocess_adni15t.py` and `preprocess_oasis.py`.
#   * `--quick` mode is provided for smoke testing on a single fold/seed.
#   * Results are written to ./experiment_results/ .
# -----------------------------------------------------------------------------
set -euo pipefail

DATA_ROOT="${1:-sample_data}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_results}"
EPOCHS="${EPOCHS:-80}"
GPU="${GPU:-0}"

echo "=============================================================="
echo " ARA-Net :: full paper benchmark"
echo "   data root : ${DATA_ROOT}"
echo "   output    : ${OUTPUT_DIR}"
echo "   epochs    : ${EPOCHS}"
echo "   gpu       : ${GPU}"
echo "=============================================================="

python -m chapter1_foundation.run_experiment_v3 \
    --config configs/default.yaml \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --gpu "${GPU}" \
    --tensorboard
