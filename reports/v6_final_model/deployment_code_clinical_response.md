# Deployment, Open-Source, And Clinical-Use Response

## 中文判断

审稿人如果问“是否可部署、是否开源、是否临床可用”，建议这样回答：

- **可部署性：不能说已经可部署。** 当前是 research-grade reproducible pipeline，不是医院可直接上线的软件。
- **开源：可以承诺开放代码。** 代码、split manifest、训练/评估脚本、图表生成脚本可以开源；原始 ADNI/AIBL/OASIS/IXI 数据受数据协议限制，不能随论文直接公开。
- **临床可用性：不能说临床可用。** 只能说显示了作为 decision-support research tool 的潜力，需要前瞻性验证、外部多中心验证、校准、工作流测试和监管审批。

最安全的总句：

> The current system is not intended for direct clinical deployment. It is a research prototype with reproducible code and locked external validation evidence, intended to support further prospective evaluation rather than immediate clinical decision-making.

## Reviewer Response Text

We agree that deployment status, code availability, and clinical-use boundaries should be stated explicitly. The revised manuscript now clarifies that the proposed framework is **not a deployment-ready clinical device** and is **not intended for direct diagnostic use**. The current contribution is a research-grade, reproducible atlas-guided multimodal AD staging pipeline evaluated on leakage-free subject-level splits, including a locked AIBL heldout cohort and an IXI healthy negative-control cohort.

To improve reproducibility, we will release the model-training scripts, post-hoc ensemble calibration scripts, subject-level evaluation scripts, figure-generation code, and de-identified split manifests. The raw neuroimaging and clinical data from ADNI, AIBL, OASIS, and IXI are governed by their respective data-use agreements and therefore cannot be redistributed directly with the manuscript. We will provide instructions for reproducing the cohort construction and evaluation once users obtain the required dataset permissions.

Clinically, the revised model should be interpreted as a decision-support research prototype rather than a clinically deployable diagnostic system. Although the final subject-level model achieved strong locked AIBL heldout performance and preserved IXI healthy specificity, clinical deployment would require additional prospective validation, site-wise calibration, scanner/protocol robustness testing, workflow integration, uncertainty reporting, and appropriate regulatory review. We have revised the Discussion to make this boundary explicit.

## Manuscript Insert: Code Availability

Code used for cohort construction, atlas-feature extraction, model training, probability-ensemble calibration, subject-level evaluation, bootstrap analysis, error analysis, and figure generation will be made publicly available upon publication. The release will include de-identified split manifests and scripts needed to reproduce the reported tables and figures. Raw ADNI, AIBL, OASIS, and IXI data are subject to their original data-use agreements and cannot be redistributed by the authors; users must obtain access through the respective data providers.

## Manuscript Insert: Deployment And Clinical Use

The proposed framework is not intended for direct clinical deployment in its current form. It should be interpreted as a research-grade decision-support prototype. The locked AIBL heldout and IXI healthy-control results provide evidence of external subject-level staging performance and healthy-control specificity, but they do not replace prospective clinical validation. Before clinical use, the system would require prospective multi-center testing, scanner/protocol robustness assessment, local calibration, integration with radiology/neurology workflows, uncertainty reporting, and regulatory review.

## Manuscript Insert: Limitations Paragraph

A further limitation is that the present implementation is not a deployable clinical device. The pipeline was designed for reproducible retrospective research evaluation rather than real-time clinical operation. Although the final subject-level model improved AIBL heldout performance and IXI specificity, OASIS transfer remained weak, and no prospective clinical workflow study was performed. Therefore, the model should not be used as a standalone diagnostic tool. Its appropriate role at this stage is to motivate further validation of atlas-guided multimodal AD staging under prospective and multi-site conditions.

## Short Cover-Letter Sentence

We also added explicit statements on deployment and reproducibility: the revised framework is a research prototype rather than a clinical device, the analysis code and de-identified split manifests will be released, and direct clinical use would require prospective multi-center validation and regulatory review.
