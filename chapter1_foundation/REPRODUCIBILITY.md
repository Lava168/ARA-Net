# Reproducibility Guide — ARA-Net (Chapter 1)

This document provides step-by-step instructions to reproduce the experiments and figures for the paper.

---

## 1. Environment Setup

```bash
# Create conda environment (Python 3.9+)
conda create -n aranet python=3.10 -y
conda activate aranet

# Install dependencies
pip install -r requirements.txt

# Or use Docker (one-click reproduction)
# docker build -t aranet .
# docker run --gpus all -v $(pwd)/sample_data:/workspace/sample_data -v $(pwd)/chapter1_foundation/pretrained_v2:/workspace/chapter1_foundation/pretrained_v2 aranet
```

---

## 2. Data Preparation

### 2.1 ADNI Data

1. **Apply for ADNI access** at [adni.loni.usc.edu](https://adni.loni.usc.edu/)
2. Download T1-weighted NIfTI files and the corresponding clinical CSV (e.g. `ADNI1_Complete_1Yr_1.5T_*.csv`)
3. Extract NIfTIs to a directory (e.g. `/path/to/adni_extracted/ADNI/`)

### 2.2 Preprocess ADNI → Cache

```bash
python -m chapter1_foundation.preprocess_adni15t \
  --adni_root /path/to/adni_extracted/ADNI \
  --csv_path /path/to/ADNI1_Complete_1Yr_1.5T_2_27_2026.csv \
  --output_cache sample_data/cache_real \
  --seg_dir sample_data/segmentations \
  --target_shape 96 112 96
```

### 2.3 Segmentation (FastSurfer)

If segmentations are missing, run FastSurfer batch processing:

```bash
python -m chapter1_foundation.batch_fastsurfer_seg \
  --cache_dir sample_data/cache_real \
  --adni_root /path/to/adni_extracted/ADNI \
  --seg_dir sample_data/segmentations \
  --workers 4
```

*Requires FastSurfer installed (Docker or conda).*

### 2.4 Optional: Synthesize & OASIS

```bash
# Synthesize balanced data
python -m chapter1_foundation.synthesize_data \
  --cache_dir sample_data/cache_real \
  --synth_output sample_data/cache_real \
  --target_per_class 150 \
  --seed 42

# Optional: integrate OASIS disc1
python -m chapter1_foundation.synthesize_data \
  --cache_dir sample_data/cache_real \
  --synth_output sample_data/cache_real \
  --target_per_class 0 \
  --oasis_tar /path/to/oasis_cs_freesurfer_disc1.tar.gz \
  --oasis_xlsx /path/to/oasis_cross-sectional.xlsx \
  --seed 42
```

---

## 3. Stage 1: SSL Pretraining

```bash
python -m chapter1_foundation.pretrain_ssl \
  --cache_dir sample_data/cache_real \
  --output_dir chapter1_foundation/pretrained_v2 \
  --epochs 100 \
  --batch_size 4 \
  --accum_steps 8 \
  --gpu 0
```

Output: `chapter1_foundation/pretrained_v2/pretrained_encoder.pth`

---

## 4. Stage 2: Supervised Experiments (v3)

### Option A: One-click (recommended)

```bash
./chapter1_foundation/launch_full_v2.sh
```

### Option B: Use config file

```bash
python -m chapter1_foundation.run_experiment_v3 \
  --config chapter1_foundation/config.yaml \
  --data_root sample_data \
  --output_dir chapter1_foundation/experiment_results_v3/seed_42 \
  --gpu 0 \
  --seeds 42
```

### Option C: Manual 6-seed parallel

```bash
for gpu in 0 1 2 3 4 5; do
  seed=$((42 + gpu * 111)); [ $gpu -eq 5 ] && seed=597
  python -m chapter1_foundation.run_experiment_v3 \
    --config chapter1_foundation/config.yaml \
    --data_root sample_data \
    --output_dir chapter1_foundation/experiment_results_v3/seed_${seed} \
    --gpu $gpu --seeds $seed \
    --pretrained_encoder chapter1_foundation/pretrained_v2/pretrained_encoder.pth \
    --freeze_encoder_epochs 10 &
done
wait
```

### 4.1 TensorBoard

```bash
tensorboard --logdir chapter1_foundation/experiment_results_v3 --port 6006
```

### 4.2 Monitor progress

```bash
./chapter1_foundation/watch_progress.sh
./chapter1_foundation/show_experiment_processes.sh
```

---

## 5. Generate Figures

After all 6 seeds complete (each with `all_results.json`):

```bash
python -m chapter1_foundation.generate_figures \
  chapter1_foundation/experiment_results_v3 \
  chapter1_foundation/figures_ssl
```

Output: `chapter1_foundation/figures_ssl/fig1_*.png`, `fig12_*.png`, etc.

---

## 6. Quick Test (2 folds, 1 seed, 10 epochs)

```bash
python -m chapter1_foundation.run_experiment_v3 \
  --config chapter1_foundation/config.yaml \
  --data_root sample_data \
  --output_dir chapter1_foundation/experiment_results_v3_quick \
  --gpu 0 \
  --quick
```

---

## 7. Hardware & Time Estimates

| Stage | GPU | Time (approx) |
|-------|-----|---------------|
| SSL pretraining | 1× GPU | ~8–12 h |
| Full experiment (6 seeds × 5 folds × 6 models) | 6× GPU | ~5–6 days |
| Figure generation | CPU | ~5 min |

---

## 8. Expected Outputs

- `experiment_results_v3/seed_*/all_results.json` — aggregated metrics
- `experiment_results_v3/seed_*/best_model_seed*_fold*.pth` — best checkpoints
- `figures_ssl/fig1_training_dynamics.png` … `fig12_bootstrap_ci.png` — publication figures
