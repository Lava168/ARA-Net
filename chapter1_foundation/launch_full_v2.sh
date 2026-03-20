#!/bin/bash
# =============================================================================
# Full pipeline: SSL pretraining (2982 volumes) + 6-seed experiment (2398 ADNI)
# =============================================================================
cd "/home/lry/atlas_guided_attention Alzheimer's Disease Dynamics"
export PYTHONUNBUFFERED=1

EPOCHS_SSL=100
EPOCHS_EXP=60
PATIENCE=15
DATA=sample_data
PRETRAIN_DIR=chapter1_foundation/pretrained_v2
OUTBASE=chapter1_foundation/experiment_results_v3
SSL_GPU=0

echo "============================================================"
echo "  STAGE 1: SSL Pretraining (2982 volumes, GPU ${SSL_GPU})"
echo "============================================================"
echo "  Epochs: ${EPOCHS_SSL}"
echo ""

mkdir -p "$PRETRAIN_DIR"

python3 -m chapter1_foundation.pretrain_ssl \
    --cache_dir "${DATA}/cache_real" \
    --output_dir "$PRETRAIN_DIR" \
    --epochs "$EPOCHS_SSL" \
    --batch_size 4 \
    --lr 1e-3 \
    --gpu "$SSL_GPU" \
    --base_channels 32 \
    --accum_steps 8 \
    2>&1 | tee "${PRETRAIN_DIR}/pretrain.log"

ENCODER="${PRETRAIN_DIR}/pretrained_encoder.pth"
if [ ! -f "$ENCODER" ]; then
    echo "ERROR: Pretrained encoder not found at ${ENCODER}"
    exit 1
fi
echo ""
echo "SSL pretraining done. Encoder: ${ENCODER}"
echo ""

echo "============================================================"
echo "  STAGE 2: Experiments (6 seeds x 5 folds x 6 models)"
echo "============================================================"
echo "  ADNI: ~2398 subjects, Epochs: ${EPOCHS_EXP}, Patience: ${PATIENCE}"
echo "  GPUs: 0-5 (one seed per GPU)"
echo ""

SEEDS=(42 153 264 375 486 597)

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=$i
    OUT="${OUTBASE}/seed_${SEED}"
    mkdir -p "$OUT"
    echo "  Seed ${SEED} on GPU ${GPU} -> ${OUT}"
    python3 -m chapter1_foundation.run_experiment_v3 \
        --data_root "$DATA" \
        --output_dir "$OUT" \
        --gpu "$GPU" \
        --epochs "$EPOCHS_EXP" \
        --patience "$PATIENCE" \
        --seeds "$SEED" \
        --batch_size 4 \
        --lr 5e-4 \
        --anat_weight 0.1 \
        --warmup 5 \
        --label_smoothing 0.1 \
        --weight_decay 1e-3 \
        --dropout 0.3 \
        --include_synth \
        --pretrained_encoder "$ENCODER" \
        --freeze_encoder_epochs 10 \
        > "${OUT}/train.log" 2>&1 &
done

echo ""
echo "All 6 seeds launched in parallel. Monitor with:"
echo "  tail -f ${OUTBASE}/seed_*/train.log"
echo "  nvidia-smi"
echo ""

wait
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================================"
