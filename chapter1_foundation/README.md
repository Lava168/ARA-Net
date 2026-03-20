# Chapter 1: Atlas-Guided Attention (Foundation)

**完整复现步骤见项目根目录：** [REPRODUCIBILITY.md](../REPRODUCIBILITY.md)

---

## 目录结构

```
chapter1_foundation/
├── README.md                    # 本说明
├── launch_full_v2.sh            # 主启动脚本（SSL 预训练 + 6-seed 实验）
├── watch_progress.sh            # 监控 v3 训练进度（每 30 秒刷新）
├── show_experiment_processes.sh # 显示 v3 训练进程
├── config.yaml                  # 实验超参配置（可选）
├── pretrain_ssl.py              # 自监督预训练 (Models Genesis)
├── run_experiment.py            # 5-fold 训练（支持 --pretrained_encoder）
├── run_experiment_v3.py         # 全量实验 v3（6 models × 5 folds，支持 TensorBoard）
├── generate_figures.py          # Nature 风格出图
├── batch_fastsurfer_seg.py      # FastSurfer 批量分割
├── preprocess_adni15t.py        # ADNI 1.5T 预处理
├── synthesize_data.py           # 数据合成
├── recover_partial.py           # 从 checkpoint 恢复结果
├── pretrained_v2/               # 预训练权重（2,982 volumes）
│   ├── pretrained_encoder.pth
│   └── pretrain_history.json
├── experiment_results_v3/       # 主实验结果（2,401 ADNI，进行中）
│   ├── seed_42/
│   ├── seed_153/
│   └── ...
├── experiment_results_ssl/      # 306 样本 SSL 基线
├── figures_ssl/                 # 出图输出
├── logs/                        # 运行日志
├── archive/                     # 已归档的旧版本
│   ├── legacy_results/         # experiment_results, v2, v2_test 等
│   ├── old_pretrained/         # 旧预训练
│   └── deprecated_scripts/     # launch_parallel.sh, launch_v2.sh
├── models/
├── data/
├── losses/
├── augmentation.py
├── metrics.py
└── attention_analysis.py
```

---

## 快速开始

### 1. 自监督预训练

```bash
python -m chapter1_foundation.pretrain_ssl --cache_dir sample_data/cache_real \
  --output_dir chapter1_foundation/pretrained_v2 --epochs 100 --batch_size 4 \
  --accum_steps 8 --gpu 0
```

### 2. 全量实验（推荐：一键启动）

```bash
./chapter1_foundation/launch_full_v2.sh
```

或手动启动 6-seed 并行：

```bash
for gpu in 0 1 2 3 4 5; do
  seed=$((42 + gpu * 111)); [ $gpu -eq 5 ] && seed=597
  python -m chapter1_foundation.run_experiment_v3 --data_root sample_data \
    --output_dir chapter1_foundation/experiment_results_v3/seed_${seed} \
    --gpu $gpu --epochs 60 --n_folds 5 --seeds $seed --augment \
    --pretrained_encoder chapter1_foundation/pretrained_v2/pretrained_encoder.pth \
    --freeze_encoder_epochs 10 &
done
wait
```

### 3. 监控进度

```bash
./chapter1_foundation/watch_progress.sh
./chapter1_foundation/show_experiment_processes.sh
```

### 4. 出图

实验全部跑完后：

```bash
python -m chapter1_foundation.generate_figures \
  chapter1_foundation/experiment_results_v3 \
  chapter1_foundation/figures_ssl
```

### 5. TensorBoard

```bash
tensorboard --logdir chapter1_foundation/experiment_results_v3 --port 6006
```

### 6. 使用配置文件

```bash
python -m chapter1_foundation.run_experiment_v3 \
  --config chapter1_foundation/config.yaml \
  --data_root sample_data \
  --output_dir chapter1_foundation/experiment_results_v3/seed_42 \
  --gpu 0 --seeds 42
```

---

## 核心模块

| 模块 | 说明 |
|------|------|
| `models/atlas_guided_model.py` | ARA-Net（RegionPooling + AnatomyGuidedAttention） |
| `models/baselines.py` | Plain CNN, ResNet-18 3D, ViT 3D |
| `losses/geodesic_loss.py` | AnatomicalDistanceLoss |
| `data/foundation_loader.py` | RealCachedDataset, kfold_split |

---

更完整的项目索引见：**`UPDATES_AND_RESULTS.md`**。
