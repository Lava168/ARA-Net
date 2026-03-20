# ARA-Net: Atlas-guided Region Attention Network for Interpretable Alzheimer's Disease Diagnosis from Structural MRI

<p align="center">
  <img src="chapter1_foundation/figures_ssl/fig11_ssl_pipeline.png" width="90%" alt="ARA-Net Architecture"/>
</p>

<p align="center">
  <b>ARA-Net architecture overview.</b> (a) Self-supervised pretraining on 887 unlabeled MRI volumes. (b) Supervised fine-tuning with atlas-guided region attention and anatomical regularization.
</p>

---

## Overview

ARA-Net is a deep learning framework for three-class Alzheimer's disease classification (CN / MCI / AD) from structural MRI, designed with **built-in interpretability** rather than post-hoc explanation.

The key idea: instead of applying attention over thousands of spatial voxel tokens, ARA-Net pools CNN features into **21 anatomically defined brain regions** (via FreeSurfer parcellation) and applies multi-head self-attention over these regions. This produces attention weights that directly correspond to clinically meaningful structures (e.g., hippocampus, amygdala, ventricles), enabling quantitative validation against established AD neuropathology.

### Key Results

| Metric | Value |
|--------|-------|
| Balanced Accuracy | 67.1% (95% CI: 66.2--67.9%) |
| Macro AUC | 0.830 +/- 0.016 |
| Protocol | 6 seeds x 5-fold CV (30 runs) |
| Data | 2,401 ADNI scans |

- 16/21 brain regions show statistically significant attention differences across diagnostic groups (p < 0.01)
- 8 regions exhibit monotonic CN -> MCI -> AD attention gradients
- Attention profiles generalize across datasets (cosine similarity > 0.98 on IXI and OASIS)
- Attention remains anatomically coherent even under misclassification

## Project Structure

```
.
├── chapter1_foundation/
│   ├── models/
│   │   ├── atlas_guided_model.py    # ARA-Net architecture (encoder + region attention + classifier)
│   │   └── baselines.py             # ResNet3D, ViT3D, PlainCNN baselines
│   ├── data/
│   │   └── foundation_loader.py     # Dataset loading and preprocessing
│   │
│   ├── pretrain_ssl.py              # Self-supervised pretraining (Models Genesis-style)
│   ├── run_experiment_v3.py         # Main training script (6-seed x 5-fold CV)
│   ├── config.yaml                  # Hyperparameter configuration
│   ├── augmentation.py              # MixUp and data augmentation
│   ├── metrics.py                   # Evaluation metrics
│   │
│   ├── attention_biomarker_analysis.py          # RDI, disease gradient, CAS analysis
│   ├── cross_dataset_interpretability.py        # Cross-dataset attention generalization
│   ├── error_conditioned_interpretability.py    # Error-conditioned analysis (CAS, Hit@K, cosine)
│   ├── external_validation.py                   # IXI / OASIS external evaluation
│   │
│   ├── preprocess_adni15t.py        # ADNI preprocessing pipeline
│   ├── preprocess_oasis.py          # OASIS preprocessing pipeline
│   ├── batch_fastsurfer_seg.py      # FreeSurfer parcellation batch processing
│   ├── synthesize_data.py           # Synthetic data generation
│   │
│   ├── generate_publication_figures.py  # Publication-quality figures (SciencePlots/Nature)
│   ├── generate_figures.py              # General figure generation
│   ├── generate_supplementary.py        # Supplementary material figures
│   ├── aggregate_results.py             # Cross-seed result aggregation
│   ├── sota_comparison.py               # SOTA comparison table
│   │
│   ├── figures_submission/          # Main paper figures (Nature style)
│   ├── figures_ssl/                 # Training and SSL figures
│   ├── figures_publication/         # Additional publication figures
│   ├── figures_biomarker/           # Biomarker analysis figures
│   ├── figures_cross_dataset/       # Cross-dataset figures
│   ├── figures_supplementary/       # Supplementary figures
│   │
│   └── paper_MedIA/                 # Manuscript
│       ├── main.tex / main.pdf          # Main paper (LaTeX)
│       ├── supplementary.tex / .pdf     # Supplementary materials
│       ├── main_chinese.tex / .pdf      # Chinese translation
│       └── ARA-Net_MedIA_Paper.docx     # Word version
```

## Method

### 1. Self-Supervised Pretraining

The 3D CNN encoder is pretrained on 887 unlabeled MRI volumes (306 ADNI + 581 IXI) using four Models Genesis-style corruption transforms:

- Non-linear intensity transformation
- Local pixel shuffling
- 3D block inpainting
- Out-painting

The encoder-decoder is trained to reconstruct original volumes from corrupted inputs (MSE + SSIM loss, 100 epochs).

### 2. Atlas-Guided Region Attention

After pretraining, the encoder is fine-tuned with a region attention module:

1. **3D CNN Encoder** (4-stage residual, 1 -> 32 -> 64 -> 128 -> 256 channels) extracts volumetric features
2. **Atlas-Guided Region Pooling** aggregates voxel features into 21 FreeSurfer-defined anatomical regions via masked average pooling
3. **Multi-Head Self-Attention** (4 heads, 2 layers) computes inter-region attention with anatomical validity masking
4. **Anatomical Regularization** encourages sparse, anatomically plausible attention distributions
5. **MLP Classifier** produces three-class diagnostic predictions

### 3. Attention-as-Biomarker Analysis

A systematic framework to validate clinical relevance of learned attention:

- **Region Discriminability Index (RDI)**: Kruskal-Wallis H-test + post-hoc pairwise comparisons with Bonferroni correction and Cohen's d effect sizes
- **Disease Progression Gradient**: Monotonicity analysis of CN -> MCI -> AD attention trajectories
- **Clinical Alignment Score (CAS)**: Proportion of attention allocated to AD-key regions (hippocampus, amygdala, ventricles)

## Datasets

| Dataset | Samples | Use |
|---------|---------|-----|
| ADNI | 2,401 scans | Training + evaluation (5-fold CV) |
| IXI | 581 scans | SSL pretraining + external validation |
| OASIS | 99 scans | SSL pretraining + external validation |
| SYNTH | synthetic | Training augmentation |

> **Note**: Raw imaging data is not included in this repository. See the [Data Availability](#data-availability) section.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- FreeSurfer (for brain parcellation)
- einops, scipy, scikit-learn, matplotlib, seaborn, scienceplots

## Quick Start

### Preprocessing

```bash
# 1. Run FreeSurfer parcellation on raw MRIs
python chapter1_foundation/batch_fastsurfer_seg.py --input_dir /path/to/nifti

# 2. Preprocess ADNI data
python chapter1_foundation/preprocess_adni15t.py
```

### Self-Supervised Pretraining

```bash
python chapter1_foundation/pretrain_ssl.py --epochs 100 --batch_size 8 --lr 1e-3
```

### Training (6-seed x 5-fold CV)

```bash
python chapter1_foundation/run_experiment_v3.py \
    --seeds 42 153 264 375 486 597 \
    --n_folds 5 \
    --epochs 60 \
    --batch_size 4 \
    --lr 5e-4
```

### Interpretability Analysis

```bash
# Attention-as-Biomarker analysis
python chapter1_foundation/attention_biomarker_analysis.py

# Cross-dataset generalization
python chapter1_foundation/cross_dataset_interpretability.py

# Error-conditioned interpretability
python chapter1_foundation/error_conditioned_interpretability.py
```

## Data Availability

- **ADNI**: Available from [adni.loni.usc.edu](https://adni.loni.usc.edu) (requires data use agreement)
- **IXI**: Publicly available at [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/)
- **OASIS**: Publicly available at [oasis-brains.org](https://www.oasis-brains.org)

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhao2026aranet,
  title={ARA-Net: Atlas-guided Region Attention Network for Interpretable Alzheimer's Disease Diagnosis from Structural MRI},
  author={Zhao, Yuanqin and Mao, Jian and Hu, Jinghong and Zhou, Bisong and Song, Zhuoyao and Guo, Tengfei and Ma, Shaohua},
  journal={Medical Image Analysis},
  year={2026}
}
```

## License

This project is for academic research purposes. Please contact the authors for commercial use.

---

# ARA-Net: 基于图谱引导区域注意力的可解释阿尔茨海默病结构MRI诊断

<p align="center">
  <img src="chapter1_foundation/figures_ssl/fig11_ssl_pipeline.png" width="90%" alt="ARA-Net 架构"/>
</p>

<p align="center">
  <b>ARA-Net 架构总览。</b> (a) 在887个无标签MRI上进行自监督预训练。(b) 基于图谱引导区域注意力和解剖正则化的监督微调。
</p>

---

## 概述

ARA-Net 是一个面向结构MRI的阿尔茨海默病三分类深度学习框架（CN / MCI / AD），具备**内建可解释性**而非事后解释。

核心思想：ARA-Net 不在数千个空间体素上做注意力计算，而是将CNN特征按 **21个FreeSurfer解剖区域** 进行池化，再在区域级别施加多头自注意力。由此产生的注意力权重直接对应临床可识别的脑结构（如海马体、杏仁核、脑室），可与已知的AD神经病理学进行定量验证。

### 主要结果

| 指标 | 数值 |
|------|------|
| 平衡准确率 | 67.1%（95% CI: 66.2--67.9%）|
| 宏观 AUC | 0.830 +/- 0.016 |
| 实验方案 | 6种子 x 5折交叉验证（30次运行）|
| 数据量 | 2,401 张ADNI扫描 |

- 21个脑区中有16个在诊断组间表现出统计显著的注意力差异（p < 0.01）
- 8个区域呈现单调的 CN -> MCI -> AD 注意力梯度
- 注意力分布在跨数据集间具有高度泛化性（IXI和OASIS上余弦相似度 > 0.98）
- 即使在误分类情况下，注意力模式仍保持解剖学一致性

## 方法

### 1. 自监督预训练

3D CNN 编码器在887个无标签MRI上进行 Models Genesis 风格的预训练，使用四种数据损坏变换：非线性强度变换、局部像素打乱、3D块修复和外绘。编码器-解码器通过重建原始体积进行训练（MSE + SSIM 损失，100个epoch）。

### 2. 图谱引导区域注意力

1. **3D CNN 编码器**（4阶段残差网络，通道 1 -> 32 -> 64 -> 128 -> 256）提取体积特征
2. **图谱引导区域池化** 通过掩码平均池化将体素特征聚合到21个FreeSurfer解剖区域
3. **多头自注意力**（4头，2层）计算区域间注意力，并使用解剖有效性掩码
4. **解剖正则化** 鼓励稀疏且解剖合理的注意力分布
5. **MLP 分类器** 输出三分类诊断结果

### 3. 注意力生物标志物分析框架

系统性验证注意力权重临床相关性的三个分析维度：

- **区域判别指数（RDI）**：Kruskal-Wallis H检验 + Bonferroni校正事后两两比较 + Cohen's d 效应量
- **疾病进展梯度**：CN -> MCI -> AD 注意力轨迹的单调性分析
- **临床对齐分数（CAS）**：注意力分配到AD关键区域（海马体、杏仁核、脑室）的比例

## 数据集

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| ADNI | 2,401 扫描 | 训练 + 评估（5折交叉验证）|
| IXI | 581 扫描 | 自监督预训练 + 外部验证 |
| OASIS | 99 扫描 | 自监督预训练 + 外部验证 |
| SYNTH | 合成数据 | 训练数据增强 |

> **注意**：本仓库不包含原始影像数据。请参阅上方[数据获取](#data-availability)章节。

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- FreeSurfer（用于脑区分割）
- einops, scipy, scikit-learn, matplotlib, seaborn, scienceplots

## 快速开始

### 预处理

```bash
# 1. 对原始MRI运行FreeSurfer分割
python chapter1_foundation/batch_fastsurfer_seg.py --input_dir /path/to/nifti

# 2. 预处理ADNI数据
python chapter1_foundation/preprocess_adni15t.py
```

### 自监督预训练

```bash
python chapter1_foundation/pretrain_ssl.py --epochs 100 --batch_size 8 --lr 1e-3
```

### 训练（6种子 x 5折交叉验证）

```bash
python chapter1_foundation/run_experiment_v3.py \
    --seeds 42 153 264 375 486 597 \
    --n_folds 5 \
    --epochs 60 \
    --batch_size 4 \
    --lr 5e-4
```

### 可解释性分析

```bash
# 注意力生物标志物分析
python chapter1_foundation/attention_biomarker_analysis.py

# 跨数据集泛化分析
python chapter1_foundation/cross_dataset_interpretability.py

# 错误条件下的可解释性分析
python chapter1_foundation/error_conditioned_interpretability.py
```

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{zhao2026aranet,
  title={ARA-Net: Atlas-guided Region Attention Network for Interpretable Alzheimer's Disease Diagnosis from Structural MRI},
  author={Zhao, Yuanqin and Mao, Jian and Hu, Jinghong and Zhou, Bisong and Song, Zhuoyao and Guo, Tengfei and Ma, Shaohua},
  journal={Medical Image Analysis},
  year={2026}
}
```

## 许可

本项目仅供学术研究使用。商业用途请联系作者。
