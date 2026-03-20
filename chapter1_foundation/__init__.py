"""
Chapter 1: The Foundation — ARA-Net (Anatomical Region Attention Network)

Atlas-guided region attention for three-class Alzheimer's disease classification
(CN / MCI / AD) from structural MRI.

Modules:
    models          - AtlasGuidedAttentionModel (3D CNN + region-pooled attention)
    losses         - AnatomicalDistanceLoss (anatomical distance constraint)
    data           - RealCachedDataset, kfold_split (ADNI / IXI / cached)
    augmentation   - 3D medical image augmentation transforms
    metrics        - AUC, F1, per-class, ROC, bootstrap CI
    attention_analysis - Regional attention extraction, group comparison

Scripts:
    pretrain_ssl.py       - Self-supervised pretraining (Models Genesis style)
    run_experiment_v3.py  - Full experiment: 6 models × 5 folds × 6 seeds
    run_experiment.py     - Legacy 5-fold training (single seed)
    generate_figures.py   - Nature-style publication figures
    batch_fastsurfer_seg.py - FastSurfer batch segmentation
    preprocess_adni15t.py - ADNI 1.5T preprocessing
    synthesize_data.py   - Data synthesis (elastic, atrophy, OASIS)
    recover_partial.py   - Recover results from checkpoints
"""
