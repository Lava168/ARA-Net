#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics/chapter1_foundation"
PYTHON="/home/lry/anaconda3/bin/python"
EVAL_SCRIPT="/tmp/evaluate_external_ad_datasets.py"

cd "$PROJECT_ROOT"
mkdir -p outputs/analysis/logs

for shard in 0 1 2 3 4 5; do
  output_json="$PROJECT_ROOT/outputs/analysis/external_validation_v3_shard${shard}.json"
  log_file="$PROJECT_ROOT/outputs/analysis/logs/external_validation_v3_shard${shard}.log"
  nohup "$PYTHON" "$EVAL_SCRIPT" \
    --checkpoint-root "$PROJECT_ROOT/outputs/experiments/experiment_results_v3" \
    --checkpoint-glob 'seed_*/best_model_seed*_fold*.pth' \
    --datasets aibl,oasis,ixi \
    --batch-size 4 \
    --num-workers 2 \
    --device "cuda:${shard}" \
    --amp \
    --checkpoint-shard-index "${shard}" \
    --checkpoint-shard-count 6 \
    --output-json "$output_json" \
    > "$log_file" 2>&1 &
  echo "shard${shard}:$!"
done
