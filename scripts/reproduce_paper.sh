#!/usr/bin/env bash
# Reproduce the public ARA-Net RC-SPE smoke-test package.
#
# This script intentionally works only with redistributable synthetic or
# aggregate inputs. It does not run private raw-MRI preprocessing or train
# restricted checkpoints.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/expected_results}"

mkdir -p "${OUTPUT_DIR}"

echo "ARA-Net public reproduction"
echo "  output: ${OUTPUT_DIR}"

"${PYTHON}" scripts/prepare_features.py \
  --metadata data/example_metadata.csv \
  --output "${OUTPUT_DIR}/demo_atlas_features.csv"

"${PYTHON}" scripts/train_base_models.py \
  --features "${OUTPUT_DIR}/demo_atlas_features.csv" \
  --output "${OUTPUT_DIR}/demo_probability_streams.csv"

"${PYTHON}" scripts/fit_rc_spe.py \
  --input-csv "${OUTPUT_DIR}/demo_probability_streams.csv" \
  --config deployment/final_ensemble_config.json \
  --output "${OUTPUT_DIR}/locked_rc_spe_config.json"

"${PYTHON}" scripts/evaluate_aibl.py \
  --input-csv "${OUTPUT_DIR}/demo_probability_streams.csv" \
  --config deployment/final_ensemble_config.json \
  --unit subject \
  --output "${OUTPUT_DIR}/aibl_demo_predictions.csv" \
  --metrics-json "${OUTPUT_DIR}/aibl_demo_metrics.json"

"${PYTHON}" scripts/evaluate_ixi.py \
  --input-csv "${OUTPUT_DIR}/demo_probability_streams.csv" \
  --config deployment/final_ensemble_config.json \
  --unit subject \
  --output "${OUTPUT_DIR}/ixi_demo_predictions.csv" \
  --metrics-json "${OUTPUT_DIR}/ixi_demo_metrics.json"

"${PYTHON}" scripts/evaluate_oasis.py \
  --input-csv "${OUTPUT_DIR}/demo_probability_streams.csv" \
  --config deployment/final_ensemble_config.json \
  --unit subject \
  --output "${OUTPUT_DIR}/oasis_demo_predictions.csv" \
  --metrics-json "${OUTPUT_DIR}/oasis_demo_metrics.json"

"${PYTHON}" scripts/reproduce_ablation.py \
  --output "${OUTPUT_DIR}/rc_spe_ablation_summary.json"

"${PYTHON}" scripts/reproduce_figures.py \
  --figure-dir assets/manuscript_figures \
  --output "${OUTPUT_DIR}/figure_manifest.json"

echo "Done."
