#!/bin/bash
# 实时显示 6 个 seed 的训练进度，每 30 秒刷新
cd "$(dirname "$0")/.."
echo "========== 6 个 Seed 训练进度 (v3, 每30秒刷新) =========="
echo ""
for seed in 42 153 264 375 486 597; do
  case $seed in
    42)  gpu=0 ;;
    153) gpu=1 ;;
    264) gpu=2 ;;
    375) gpu=3 ;;
    486) gpu=4 ;;
    597) gpu=5 ;;
    *)   gpu=? ;;
  esac
  f="chapter1_foundation/experiment_results_v3/seed_${seed}/train.log"
  echo ">>> GPU${gpu} seed_${seed} <<<"
  if [ -f "$f" ]; then
    tail -n 3 "$f"
  else
    echo "  无日志"
  fi
  echo ""
done
