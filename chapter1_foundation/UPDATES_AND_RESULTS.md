# 项目更新与结果索引

本文档汇总 **atlas_guided_attention Alzheimer's Disease Dynamics** 的近期改动、结果位置与待办任务，便于在项目中快速查找。

---

## 一、新改动概览

| 类别 | 内容 | 位置 |
|------|------|------|
| **数据扩展** | FastSurfer 批量分割，ADNI 有效样本从 306 → **2,398**（CN=748, MCI=1156, AD=495） | `chapter1_foundation/batch_fastsurfer_seg.py` |
| **SSL 预训练 v2** | Models Genesis 风格，**2,982** 张无标签 MRI（ADNI+IXI），100 epochs | `chapter1_foundation/pretrain_ssl.py` |
| **预训练权重 v2** | 基于 2,982 体积的 encoder | `chapter1_foundation/pretrained_v2/pretrained_encoder.pth` |
| **全量实验 v3** | 2,401 ADNI + 144 SYNTH + 25 OASIS，6 seeds × 5 folds × 6 models | `chapter1_foundation/run_experiment_v3.py` |
| **一键启动脚本** | SSL 预训练 + 6-seed 并行实验 | `chapter1_foundation/launch_full_v2.sh` |
| **新增图表** | Fig 11 SSL 流程图、Fig 12 Bootstrap 95% CI | `chapter1_foundation/figures_ssl/fig11_*.png`, `fig12_*.png` |
| **出图脚本** | Nature 风格，12 张图（含 ARA-Net 命名） | `chapter1_foundation/generate_figures.py` |
| **模型命名** | ARA-Net (Anatomical Region Attention Network) | 图中显示名称 |
| **复现指南** | 数据准备 → 预训练 → 实验 → 出图 | `REPRODUCIBILITY.md` |
| **依赖** | requirements.txt | 项目根目录 |
| **配置** | config.yaml（可选） | `chapter1_foundation/config.yaml` |
| **TensorBoard** | `--tensorboard` 记录 loss/val_bacc | `run_experiment_v3.py` |
| **Docker** | 一键复现环境 | `Dockerfile` |

---

## 二、结果与输出位置

| 类型 | 路径 | 说明 |
|------|------|------|
| **SSL 预训练 v2 历史** | `chapter1_foundation/pretrained_v2/pretrain_history.json` | 100 epochs loss 曲线 |
| **实验 v3 结果** | `chapter1_foundation/experiment_results_v3/seed_*/` | 6 seeds 并行，每 seed 含 `all_results_partial.json`、`train.log` |
| **实验 v3 最佳模型** | `experiment_results_v3/seed_*/best_model_seed*_fold*.pth` | ARA-Net Full 的 checkpoint |
| **出图输出** | `chapter1_foundation/figures_ssl/` | Fig 1–12（PNG + PDF），图注见 `figure_captions.md` |
| **旧版 SSL 实验** | `chapter1_foundation/experiment_results_ssl/` | 306 样本，6 seeds |
| **FastSurfer 批处理摘要** | `sample_data/cache_real/fastsurfer_batch_summary.json` | 2,086 成功 / 3 失败 |
| **运行日志** | `chapter1_foundation/logs/` | fastsurfer_batch.log, launch_full_v2.log |
| **归档（旧版）** | `chapter1_foundation/archive/` | 已废弃的 experiment_results、v2、pretrained 等 |

---

## 三、当前实验状态（v3）

- **数据**：ADNI 2,401（CN=750, MCI=1156, AD=495）+ SYNTH 144 + OASIS 25
- **配置**：6 seeds × 5 folds × 6 models = 180 次训练，6 GPU 并行
- **进度**：约 12–15% 完成（各 seed 在 Fold 1 的第 4–5 个模型）
- **预计剩余**：约 5–6 天
- **监控**：`./chapter1_foundation/watch_progress.sh` 或 `tail -f chapter1_foundation/experiment_results_v3/seed_*/train.log`

---

## 四、实验跑完后的待办任务

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 用 v3 结果重新生成全部图表 | 见下方「出图」命令，直接传 `experiment_results_v3` 即可 |
| **P1** | IXI 外部验证 | ADNI 训练 → 581 IXI CN 推理，测 CN 特异性 |
| **P1** | OASIS 外部验证 | 解压 OASIS disc1–6，预处理后作为 held-out 三分类测试集 |
| **P2** | SOTA 对比表 | 检索 5–8 篇 ADNI 三分类论文，整理 Table 对比 |
| **P2** | 更新 figure_captions.md | 将硬编码数字替换为 v3 新结果 |
| **P2** | 更新 Fig 11 描述 | 887 volumes → 2,982 volumes |

---

## 五、如何跑完流程

1. **预训练（已完成）**  
   权重在 `chapter1_foundation/pretrained_v2/pretrained_encoder.pth`。

2. **全量实验（进行中）**  
   由 `launch_full_v2.sh` 启动，结果持续写入 `experiment_results_v3/seed_*/`。

3. **出图**  
   实验全部跑完后（每个 seed 目录下会有 `all_results.json`），在项目根目录执行：
   ```bash
   python -m chapter1_foundation.generate_figures \
     chapter1_foundation/experiment_results_v3 \
     chapter1_foundation/figures_ssl
   ```

4. **外部验证**  
   待 v3 实验完成后，按 `foundation_loader.py` 的 `site_filter` 支持实现 IXI/OASIS 推理脚本。

---

## 六、Chapter 1 详细说明

更详细的命令、目录结构及参数说明见：**`chapter1_foundation/README.md`**。

以上所有路径均相对于项目根目录 **`atlas_guided_attention Alzheimer's Disease Dynamics`**。
