#!/usr/bin/env python3
"""Build a V6 LaTeX replacement draft from the original ARA-Net LaTeX shell.

The generated draft keeps the original LaTeX formatting, author block,
affiliations, CRediT block, acknowledgments, competing-interest statement,
funding statement, and bibliography shell. The scientific body is replaced by
the V6 evidence package and seven main-figure placeholders.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_SOURCE = Path("/private/tmp/aranet_original_main.tex")
DEFAULT_OUTPUT = Path("reports/v6_final_model/latex/main_v6_replacement.tex")
DEFAULT_BODY = Path("reports/v6_final_model/latex/v6_body_expanded.tex")
DEFAULT_FIGURE_PLAN = Path("reports/v6_final_model/latex/v6_latex_7_figure_plan.md")


TITLE = (
    r"\title{\textbf{ARA-Net: Atlas-Guided Multimodal Alzheimer's Disease "
    r"Staging with Locked External Subject-Level Validation and Structural "
    r"Neurodegeneration Consistency}}"
)
TIMES_FONT_PACKAGE = r"\usepackage{mathptmx}"


def apply_times_font(header: str) -> str:
    if TIMES_FONT_PACKAGE in header:
        return header
    anchor = r"\usepackage{amsmath,amssymb,amsfonts}"
    if anchor in header:
        return header.replace(anchor, anchor + "\n" + TIMES_FONT_PACKAGE, 1)
    return header


def replace_title(header: str) -> str:
    lines = []
    for line in header.splitlines():
        if line.startswith(r"\title{"):
            lines.append(TITLE)
        else:
            lines.append(line)
    return apply_times_font("\n".join(lines).rstrip()) + "\n"


def extract_shell(original: str) -> tuple[str, str]:
    marker = r"\section*{Author Contributions (CRediT)}"
    if marker not in original:
        raise ValueError("Could not find original Author Contributions block.")
    header_end = original.index(r"%% Abstract")
    header = original[:header_end]
    endmatter = original[original.index(marker):]
    return replace_title(header), endmatter


def body_tex(body_path: Path | None = None) -> str:
    if body_path is not None and body_path.exists():
        return body_path.read_text(encoding="utf-8").rstrip() + "\n"
    return r"""
%% ====================================================================
%% Abstract
%% ====================================================================
\begin{abstract}

\textbf{Background.}
The original ARA-Net submission was limited by weak external classification evidence, an unsupported attention-based Clinical Alignment Score (CAS), and a non-significant Braak-stage analysis. We rebuilt the work as a subject-level, atlas-guided multimodal Alzheimer's disease (AD) staging study with explicit external classification, error analysis, and bounded biological validation.

\textbf{Methods.}
We constructed leakage-aware subject-level splits across ADNI, AIBL, IXI, and OASIS. ADNI was used for training, validation, and internal testing; AIBL was divided into adaptation training, adaptation validation, and a locked heldout external test; IXI served as a healthy negative-control cohort; and OASIS was retained only as an external stress test. The final model is a locked subject-level rescue probability ensemble that combines six base-model probability streams using log-probability pooling, class-specific offsets, temperature scaling, and subject-level probability averaging. The previous attention-only CAS claim was removed and replaced by atlas structural neurodegeneration consistency analysis in a priori AD-relevant regions.

\textbf{Results.}
The original v3 model did not support the earlier external-generalization claim, with AIBL balanced accuracy (BAcc) of 0.399 and IXI healthy CN retention of 0.439. The final subject-level rescue ensemble achieved AIBL heldout accuracy of 0.903, BAcc of 0.833, macro AUC of 0.937, AD-vs-CN AUC of 1.000, and CN/MCI/AD recall of 0.961/0.686/0.852. Bootstrap 95\% confidence intervals were 0.759--0.899 for BAcc, 0.531--0.839 for MCI recall, and 0.710--0.966 for AD recall. IXI healthy CN retention was 1.000. AIBL errors were concentrated near MCI/AD boundaries, with no AD subject misclassified as CN. OASIS transfer remained weak and is reported as a limitation. The AIBL heldout AD-key atlas-volume consistency score was 0.510 versus a uniform regional null of 0.286, with bootstrap CI 0.479--0.526 and permutation $p=0.026$.

\textbf{Conclusion.}
The revised work supports a domain-adapted, subject-level, atlas-guided multimodal AD staging framework with strong locked AIBL heldout performance, preserved IXI healthy specificity, and atlas-region structural neurodegeneration consistency. It does not claim pure zero-shot transfer, solved OASIS generalization, direct Braak-stage validation, or deployment-ready clinical performance.

\end{abstract}

\vspace{0.5em}
\noindent\textbf{Keywords:} Alzheimer's disease; structural MRI; atlas-guided multimodal learning; external validation; subject-level staging; neurodegeneration consistency; open-source research prototype

\vspace{0.5em}
\noindent\textbf{Abbreviations:} AD, Alzheimer's disease; ADNI, Alzheimer's Disease Neuroimaging Initiative; AIBL, Australian Imaging, Biomarkers and Lifestyle study; AUC, area under the receiver-operating-characteristic curve; BAcc, balanced accuracy; CAS, Clinical Alignment Score; CN, cognitively normal; IXI, Information eXtraction from Images dataset; MCI, mild cognitive impairment; MMSE, Mini-Mental State Examination; OASIS, Open Access Series of Imaging Studies; sMRI, structural magnetic resonance imaging.


%% ====================================================================
\section{Introduction}
\label{sec:intro}
%% ====================================================================

Structural MRI is widely used in AD research because it captures neurodegeneration patterns including medial temporal atrophy, ventricular enlargement, and broader brain-volume change~\citep{jack2018nia,frisoni2010neuroimaging}. Machine learning can use these signals to support CN/MCI/AD staging, but a model intended for scientific or translational use must demonstrate more than within-cohort classification performance. It must also show leakage-aware external evaluation, clear error behavior across clinically important classes, and appropriately bounded biological interpretation.

The original ARA-Net manuscript attempted to make explanation similarity and an attention-derived CAS central evidence for cross-dataset generalization and biological validity. The revised work deliberately changes that target. External generalization is evaluated directly through locked external classification, healthy negative-control testing, and subject-level endpoint aggregation. The old attention-only CAS is removed as a central biological claim. Because direct Braak-stage validation is unavailable and the original Braak correlation was non-significant, the biological claim is narrowed to a structural MRI neurodegeneration proxy.

We therefore rebuilt the study rather than making a narrow revision. The revised framework uses subject-level multi-cohort manifests, AIBL adaptation with a locked external heldout split, IXI healthy negative-control testing, OASIS stress testing, comparator models, bootstrap uncertainty, and MCI/AD error analysis. The final classifier is an atlas-guided multimodal probability ensemble that combines regional MRI features and core clinical variables, then averages repeated scans at subject level for the primary endpoint.

The revised contributions are:
\begin{enumerate}[leftmargin=*,itemsep=2pt,label=\textbf{C\arabic*.}]
    \item A leakage-aware subject-level protocol across ADNI, AIBL, IXI, and OASIS.
    \item Direct external CN/MCI/AD classification on a locked AIBL heldout endpoint.
    \item IXI healthy negative-control evaluation to quantify false impairment predictions.
    \item A locked probability rescue ensemble with class offsets, temperature scaling, and subject-level probability averaging.
    \item Bootstrap stability and subject-level MCI/AD error analysis.
    \item Replacement of the unsupported attention-only CAS/Braak claim with atlas structural neurodegeneration consistency analysis.
    \item An open-source research deployment package with explicit clinical-use boundaries.
\end{enumerate}

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.63\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 1. Revised study framework and central model objective.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} old v3 claim failure and V6 rebuild target; \textbf{(b)} subject-level cohort construction; \textbf{(c)} atlas-guided multimodal features; \textbf{(d)} locked rescue probability ensemble; \textbf{(e)} endpoint and claim-boundary summary. The visual emphasis should be that the paper is now an external subject-level AD staging framework, not an attention-only interpretability patch.
    \end{minipage}}
    \caption{\textbf{Revised ARA-Net V6 framework.} The revised workflow uses leakage-aware subject-level cohort construction, atlas-guided multimodal feature extraction, a locked rescue probability ensemble, and subject-level probability averaging. The primary endpoint is locked AIBL heldout CN/MCI/AD staging; IXI is a healthy negative-control specificity endpoint; OASIS is retained as a stress-test limitation. The original attention-only CAS and direct Braak-stage claims are replaced by atlas structural neurodegeneration consistency analysis and explicit claim-boundary language.}
    \label{fig:v6_framework}
\end{figure}


%% ====================================================================
\section{Methods}
\label{sec:methods}
%% ====================================================================

\subsection{Cohorts and leakage-aware subject-level splits}
\label{sec:data}

All splits were defined at the subject level to reduce leakage from repeated scans~\citep{varoquaux2017cross}. ADNI was divided into training, validation, and internal test sets. AIBL was divided into adaptation training, adaptation validation, and a locked heldout external test. IXI was used as a healthy negative-control cohort. OASIS was retained as an external stress-test cohort and was not used for final model tuning.

Table~\ref{tab:v6_splits} summarizes the split inventory. The subject counts in this table are unique participants used to define leakage-free cohort splits before endpoint aggregation. For longitudinal cohorts, the final endpoint units may differ from unique participants because repeated scans were averaged within subject-level diagnostic-state units.

\begin{table}[t]
\centering
\caption{V6 split inventory. Counts refer to scan inventory and unique split-level subjects.}
\label{tab:v6_splits}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Split & Scans & Subjects & Role \\
\midrule
ADNI train & 1{,}686 & 450 & Main training \\
ADNI validation & 355 & 97 & Validation and calibration \\
ADNI internal test & 360 & 96 & Internal risk analysis \\
AIBL adaptation training & 719 & 385 & External-domain adaptation \\
AIBL adaptation validation & 191 & 105 & External-domain calibration \\
AIBL locked heldout & 397 & 210 & Primary external test \\
OASIS external & 99 & 99 & Stress test \\
IXI healthy external & 581 & 581 & Healthy negative-control specificity \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.60\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 2. Data line and endpoint design.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} ADNI/AIBL/OASIS/IXI cohort blocks with scan and subject counts; \textbf{(b)} ADNI train--validation--internal-test split; \textbf{(c)} AIBL adaptation train/validation and locked heldout split; \textbf{(d)} endpoint aggregation from scans to subject-level diagnostic-state units; \textbf{(e)} OASIS explicitly shown as stress test, not tuning or success validation.
    \end{minipage}}
    \caption{\textbf{Leakage-aware multi-cohort data line.} The revised study separates model development, external-domain adaptation, locked external testing, healthy-control specificity testing, and stress testing. This figure should make clear that AIBL heldout is the primary external endpoint, IXI evaluates false impairment in healthy controls, and OASIS is preserved only as a limitation stress test.}
    \label{fig:v6_data_line}
\end{figure}

\subsection{Atlas-guided multimodal feature representation}
\label{sec:features}

MRI features were extracted from a 21-region atlas and included regional volumetric and intensity summaries. The main multimodal feature set combined atlas-derived MRI features with core clinical variables, including age, sex, education, APOE4, MMSE, and CDR-SB where available. Extended cognitive, biomarker, and volumetric clinical variables were used in sensitivity or comparator models rather than as the central scientific claim.

The revised model should be described as atlas-guided and multimodal. The primary scientific evidence is classification performance, healthy negative-control specificity, error analysis, and atlas structural neurodegeneration consistency.

\subsection{Candidate models and locked rescue ensemble}
\label{sec:model}

The revised experimental framework included atlas-only, cascade, atlas+clinical, clinical-only, biomarker-enhanced, and ensemble models. The final locked model was a subject-level balanced rescue probability ensemble. It combined six base-model probability streams:
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item AIBL-adapted atlas-biomarker-enhanced HGB.
    \item AIBL-adapted atlas-core-clinical HGB.
    \item AIBL-adapted clinical-biomarker-only RF balanced.
    \item AIBL-adapted clinical-core-only HGB.
    \item AIBL-adapted clinical-core-only RF balanced.
    \item RF-logistic regression ensemble component.
\end{itemize}

Let $p_{m,k}(x_i)$ denote the class probability for scan $i$, class $k$, and base model $m$, where $k \in \{\mathrm{CN}, \mathrm{MCI}, \mathrm{AD}\}$. The final ensemble uses log-probability pooling:
\begin{equation}
z_{i,k} = \frac{1}{T}\sum_{m=1}^{M} w_m \log\left(\max(p_{m,k}(x_i), \epsilon)\right) + b_k ,
\quad k \in \{\mathrm{CN}, \mathrm{MCI}, \mathrm{AD}\},
\label{eq:v6_log_pool}
\end{equation}
where $w_m \geq 0$, $\sum_m w_m = 1$, $T$ is a temperature parameter, $b_k$ is a class-specific offset, and $\epsilon$ prevents numerical underflow. Calibrated class probabilities are:
\begin{equation}
\tilde{p}_{i,k} = \frac{\exp(z_{i,k})}{\sum_c \exp(z_{i,c})}.
\label{eq:v6_softmax}
\end{equation}
For subject-level evaluation, repeated scans within subject-level diagnostic-state unit $s$ were averaged:
\begin{equation}
\bar{p}_{s,k} = \frac{1}{n_s}\sum_{i \in s}\tilde{p}_{i,k}, \qquad
\hat{y}_s = \arg\max_k \bar{p}_{s,k}.
\label{eq:v6_subject_average}
\end{equation}

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.60\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 3. Locked rescue ensemble model.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} six base probability streams; \textbf{(b)} log-probability pooling with weights; \textbf{(c)} class offsets and temperature scaling; \textbf{(d)} scan-level to subject-level averaging; \textbf{(e)} final CN/MCI/AD probability output and decision. The main visual message is that the core model target is calibrated subject-level AD staging.
    \end{minipage}}
    \caption{\textbf{Locked subject-level rescue probability ensemble.} The final model combines six base-model probability streams through log-probability pooling, class-specific offsets, temperature scaling, and subject-level probability averaging. This architecture is the central V6 model and should be shown as a calibrated staging system rather than a binary classifier or an attention-only network.}
    \label{fig:v6_model}
\end{figure}

\subsection{Evaluation metrics, uncertainty, and error analysis}
\label{sec:metrics}

We report accuracy, balanced accuracy, macro one-vs-rest AUC, AD-vs-CN AUC, per-class recall, precision, prediction distributions, and confusion matrices. For IXI, because all subjects are healthy controls, the primary metric is CN retention, equivalent to one minus the false impairment rate. For the locked AIBL heldout subject-level endpoint, uncertainty was estimated using 2{,}000 bootstrap resamples.

MCI and AD errors were analyzed after subject-level probability averaging. We summarized true/predicted transition rates and compared error groups by age, MMSE, CDR-SB, APOE4, atlas hippocampal volume, atlas lateral ventricular volume, AD-like atlas z-score, maximum predicted probability, and decision margin. The goal of this analysis was to determine whether errors reflected complete collapse of disease classes into CN or uncertainty near adjacent disease-stage boundaries.

\subsection{Structural neurodegeneration consistency}
\label{sec:bio}

The old attention-centered CAS was removed because it did not provide valid biological evidence. The revised biological analysis instead tests whether disease-associated atlas volume changes concentrate in a priori AD-relevant regions: bilateral hippocampus, bilateral amygdala, and bilateral lateral ventricles. The score was compared against a uniform regional null using bootstrap confidence intervals and permutation testing.

This analysis is an MRI neurodegeneration proxy. It does not provide postmortem-stage validation or explanation-map biomarker evidence.

\subsection{Open-source research deployment and clinical-use boundary}
\label{sec:deployment}

The public implementation includes reproducible analysis scripts, a research inference wrapper, an HTTP API/static frontend, aggregate reports, figures, documentation, and toy probability examples. Raw ADNI, AIBL, OASIS, and IXI data are governed by their original access agreements and are not redistributed.

The software is released as an open-source research prototype for retrospective evaluation and future prospective validation. It is not a medical device, is not cleared or approved for clinical use, and is not intended for standalone diagnosis or patient-care decisions.


%% ====================================================================
\section{Results}
\label{sec:results}
%% ====================================================================

\subsection{Original external failure and final external result}
\label{sec:external_results}

The original v3 model did not support the earlier cross-dataset generalization claim. On AIBL, it achieved accuracy 0.606, BAcc 0.399, and macro AUC 0.597. On IXI, only 0.439 of healthy controls were retained as CN, indicating a high false impairment rate.

The locked final rescue ensemble achieved substantially stronger AIBL heldout performance. At subject level, it achieved accuracy 0.903, BAcc 0.833, macro AUC 0.937, and AD-vs-CN AUC 1.000. CN/MCI/AD recall was 0.961/0.686/0.852. Bootstrap 95\% confidence intervals were 0.759--0.899 for BAcc, 0.894--0.974 for macro AUC, 0.531--0.839 for MCI recall, and 0.710--0.966 for AD recall. The scan-level reference result was similar: AIBL heldout accuracy 0.909, BAcc 0.820, macro AUC 0.939, AD-vs-CN AUC 0.998, and CN/MCI/AD recall 0.964/0.642/0.854. On IXI, the final model retained 1.000 of healthy controls as CN at both scan and subject levels.

\begin{table}[t]
\centering
\caption{Main external classification results. The locked primary endpoint is the final rescue ensemble at subject level on AIBL heldout.}
\label{tab:v6_classification}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lllrrrrlll@{}}
\toprule
Model/protocol & Unit & Test cohort & Endpoint $n$ & Acc & BAcc & Macro AUC & AD-vs-CN AUC/CN retention & CN/MCI/AD recall & Role \\
\midrule
Old v3 ensemble & scan & AIBL external & 1{,}307 & 0.606 & 0.399 & 0.597 & NA & NA & Failed external baseline \\
Old v3 ensemble & scan & IXI healthy & 581 & 0.439 & 0.439 & NA & CN retention 0.439 & 0.439/0.000/0.000 & Failed specificity baseline \\
v4 atlas+clinical HGB & scan & AIBL heldout & 397 & 0.882 & 0.741 & 0.942 & AD-vs-CN AUC 0.990 & 0.964/0.528/0.732 & Earlier rebuild \\
Final rescue ensemble & scan & AIBL heldout & 397 & 0.909 & 0.820 & 0.939 & AD-vs-CN AUC 0.998 & 0.964/0.642/0.854 & Scan-level reference \\
Final rescue ensemble & subject & AIBL heldout & 216 & 0.903 & 0.833 & 0.937 & AD-vs-CN AUC 1.000 & 0.961/0.686/0.852 & Locked primary result \\
Final rescue ensemble & subject & IXI healthy & 581 & 1.000 & 1.000 & NA & CN retention 1.000 & 1.000/0.000/0.000 & Specificity check \\
Final rescue ensemble & subject & OASIS stress & 99 & 0.586 & 0.334 & 0.554 & AD-vs-CN AUC 0.371 & 0.966/0.034/0.000 & Limitation \\
Clinical-only RF comparator & scan & AIBL heldout & 397 & 0.922 & 0.835 & 0.957 & AD-vs-CN AUC 0.997 & 0.970/0.755/0.780 & Upper-bound comparator \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.62\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 4. External classification performance.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} AIBL BAcc comparison: old v3, v4 atlas+clinical, final scan, final subject, clinical-only comparator; \textbf{(b)} AIBL macro AUC comparison; \textbf{(c)} final subject-level CN/MCI/AD recall; \textbf{(d)} IXI CN retention and false impairment rate; \textbf{(e)} OASIS as a small stress-test limitation panel. The central story is old external failure to new locked AIBL subject-level success with IXI specificity.
    \end{minipage}}
    \caption{\textbf{External classification performance.} The original v3 model failed external classification and healthy specificity. The final subject-level rescue ensemble substantially improves locked AIBL heldout CN/MCI/AD staging and preserves IXI healthy specificity. OASIS is shown as a stress-test limitation, not as a successful validation cohort.}
    \label{fig:v6_external_performance}
\end{figure}

\subsection{Comparator interpretation}
\label{sec:comparators}

The clinical-only RF comparator achieved AIBL heldout BAcc 0.835 and CN/MCI/AD recall 0.970/0.755/0.780 at scan level. This model is reported as a comparator and upper bound rather than the central ARA-Net model, because it does not retain the atlas-guided MRI component central to the revised scientific objective. This comparison is important: it shows that clinical variables carry substantial diagnostic signal and prevents overstating MRI-only interpretability.

\subsection{Subject-level MCI and AD error analysis}
\label{sec:error_results}

On the locked AIBL heldout subject-level set, errors were concentrated at disease-stage boundaries. Among 154 CN endpoint units, 148 were classified as CN, five as MCI, and one as AD. Among 35 MCI endpoint units, 24 were classified as MCI, two as CN, and nine as AD. Among 27 AD endpoint units, 23 were classified as AD and four as MCI; no AD endpoint unit was misclassified as CN.

\begin{table}[t]
\centering
\caption{AIBL heldout subject-level confusion matrix for the locked final rescue ensemble.}
\label{tab:v6_confusion}
\begin{tabular}{@{}lrrrr@{}}
\toprule
True label & Predicted CN & Predicted MCI & Predicted AD & Recall \\
\midrule
CN & 148 & 5 & 1 & 0.961 \\
MCI & 2 & 24 & 9 & 0.686 \\
AD & 0 & 4 & 23 & 0.852 \\
\bottomrule
\end{tabular}
\end{table}

The feature-profile analysis supported a boundary-error interpretation. AIBL AD endpoint units correctly classified as AD had lower MMSE, larger lateral ventricular volume, and higher AD-like atlas z-scores than AD endpoint units classified as CN/MCI. MCI endpoint units classified as AD had lower MMSE and more AD-like atlas profiles than MCI endpoint units classified correctly, consistent with a disease-severity boundary rather than arbitrary failure.

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.62\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 5. Subject-level confusion and error profile.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} AIBL heldout subject-level confusion heatmap; \textbf{(b)} internal subject-level confusion heatmap; \textbf{(c)} AIBL transition/alluvial flow; \textbf{(d)} MCI-to-CN, MCI-to-AD, AD-to-MCI, AD-to-CN error subtype bars; \textbf{(e)} error-feature profile summary using MMSE, ventricle volume, hippocampus volume, AD-like z-score, confidence, and margin. The key visual claim is zero AD-to-CN errors on AIBL and boundary-like residual mistakes.
    \end{minipage}}
    \caption{\textbf{Subject-level MCI/AD error behavior.} AIBL heldout errors concentrate near adjacent disease-stage boundaries rather than collapsing disease cases into CN. Internal calibration remains weaker and should be interpreted as a limitation.}
    \label{fig:v6_error}
\end{figure}

\subsection{Bootstrap stability}
\label{sec:stability}

Bootstrap analysis with 2{,}000 resamples quantified uncertainty for the locked AIBL heldout subject-level endpoint. The 95\% CI for BAcc was 0.759--0.899. The 95\% CI for MCI recall was 0.531--0.839, and the 95\% CI for AD recall was 0.710--0.966. These intervals support the final model as the locked primary result while preserving MCI staging uncertainty as a visible limitation.

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.60\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 6. Stability and uncertainty.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} bootstrap BAcc distribution with 95\% CI; \textbf{(b)} bootstrap MCI recall distribution; \textbf{(c)} bootstrap AD recall distribution; \textbf{(d)} forest plot of BAcc, macro AUC, MCI recall, AD recall; \textbf{(e)} small note that OASIS was excluded from tuning. The visual message is that the final endpoint is stable enough to lock, with MCI as the main uncertainty.
    \end{minipage}}
    \caption{\textbf{Bootstrap stability of the locked primary endpoint.} The final AIBL subject-level result was evaluated with 2{,}000 bootstrap resamples. The distributions and confidence intervals support the locked model while identifying MCI recall as the main residual uncertainty.}
    \label{fig:v6_stability}
\end{figure}

\subsection{Structural neurodegeneration consistency and OASIS stress test}
\label{sec:bio_results}

The AIBL heldout atlas-volume consistency score exceeded the uniform regional null in the a priori AD-key regions. The score was 0.510 versus a uniform null of 0.286, with error-difference 0.225, bootstrap CI 0.479--0.526, and permutation $p=0.026$.

Across all labeled AD-relevant data, the AD-key consistency score was 0.426 versus a uniform null of 0.286, with permutation $p=0.0207$. The ADNI-only internal check remained non-significant, with score 0.342 and $p=0.1843$. These results support a bounded structural MRI neurodegeneration proxy while making clear that postmortem-stage validation is not available.

OASIS remained weak without OASIS tuning. The final subject-level model achieved OASIS accuracy 0.586, BAcc 0.334, macro AUC 0.554, AD-vs-CN AUC 0.371, and CN/MCI/AD recall 0.966/0.034/0.000. This result is reported as an unresolved transfer limitation, not as successful external validation.

\begin{figure}[p]
    \centering
    \fbox{\begin{minipage}[c][0.64\textheight][c]{0.94\textwidth}
    \textbf{Placeholder for Figure 7. CAS/Braak replacement and biological consistency.}

    \vspace{0.8em}
    Panels to draw: \textbf{(a)} old attention-only CAS failure callout; \textbf{(b)} definition of AD-key atlas regions: bilateral hippocampus, amygdala, and lateral ventricles; \textbf{(c)} AIBL heldout structural consistency score 0.510 vs null 0.286 with CI and $p=0.026$; \textbf{(d)} pooled all-labeled-AD score 0.426 vs null 0.286, plus ADNI-only non-significant internal check; \textbf{(e)} claim-boundary badges: MRI proxy only, not attention biomarker, not direct Braak proof, not clinical device.
    \end{minipage}}
    \caption{\textbf{Biological consistency and replacement of unsupported CAS/Braak claims.} The invalid attention-only CAS claim is removed. The revised biological analysis tests whether atlas structural MRI changes concentrate in a priori AD-relevant regions. The supported claim is a bounded MRI neurodegeneration proxy, not attention-map biomarker discovery or direct Braak-stage validation.}
    \label{fig:v6_biology}
\end{figure}


%% ====================================================================
\section{Discussion}
\label{sec:discussion}
%% ====================================================================

The revised study addresses the major weaknesses of the original manuscript by changing both the evidence base and the claim boundary. Cross-dataset generalization is no longer inferred from explanation similarity. It is evaluated using a locked AIBL heldout subject-level test and an IXI healthy negative-control cohort. The unsupported attention-centered CAS claim is removed and replaced by an atlas-region structural neurodegeneration consistency analysis. The non-significant disease-stage result is no longer used to claim direct pathological staging.

The final model is strongest when interpreted as a domain-adapted external AD staging framework. AIBL adaptation data were used for model fitting and calibration, but AIBL heldout endpoint units remained locked and were not used for final endpoint evaluation. This distinction is important: the work does not solve pure ADNI-to-AIBL zero-shot staging, but it does demonstrate that anatomically grounded MRI features and core clinical variables can support robust heldout subject-level staging within an external cohort.

The error analysis clarifies the clinical meaning of the remaining failures. On AIBL heldout, AD endpoint units were not missed as CN; residual AD errors were classified as MCI. MCI errors were split between correct MCI and AD, with only two MCI endpoint units classified as CN. This pattern is preferable to a model that preserves overall accuracy by collapsing minority disease classes into CN, but it also shows that precise MCI/AD boundary staging remains challenging.

The biological analysis is intentionally bounded. The AIBL heldout atlas-volume consistency result supports disease-consistent structural MRI change in AD-relevant regions. It does not demonstrate explanation-map biomarkers, and it does not provide postmortem-stage validation. This narrower claim is more defensible and better aligned with the available data.

Several limitations remain. OASIS transfer is not solved and should be treated as a stress-test failure requiring larger and cleaner external cohorts. The internal subject-level calibration pattern remains modest, especially for CN specificity within the internal split. Clinical variables contain substantial diagnostic signal, as shown by the clinical-only comparator. The 21-region atlas is anatomically interpretable but coarse, and finer parcellations may better capture cortical AD patterns. Finally, although AIBL heldout performance is strong, the model is not presented as deployment-ready clinical software; it is best framed as an open-source research prototype requiring prospective validation.


%% ====================================================================
\section{Conclusion}
\label{sec:conclusion}
%% ====================================================================

This revised work is a substantive rebuild of the original ARA-Net study. It replaces unsupported attention-centered claims with locked external subject-level classification, healthy negative-control specificity, bootstrap uncertainty, MCI/AD error analysis, and a bounded atlas structural neurodegeneration consistency analysis. The strongest supported claim is domain-adapted external AD staging with an MRI neurodegeneration proxy, not pure zero-shot generalization, postmortem-stage validation, OASIS success, or clinical deployment readiness.


%% ====================================================================
%% End-matter
%% ====================================================================

\section*{Data Availability}
Raw ADNI, AIBL, OASIS, and IXI data are governed by their original data-use agreements and are not redistributed. Public repository files are limited to code, aggregate reports, figures, documentation, final model configuration, and toy probability examples. Row-level subject/scan predictions, private clinical spreadsheets, MRI volumes, and model checkpoints are excluded from the public package.

\section*{Code Availability}
The public research implementation, documentation, aggregate reports, deployment wrappers, and final ensemble configuration are available at \url{https://github.com/Lava168/ARA-Net}. The released software is a research prototype; it is not a medical device and is not cleared or approved for clinical use.

"""


def figure_plan_md() -> str:
    return """# V6 LaTeX Seven-Figure Plan

This plan is for the LaTeX replacement manuscript. The author block, affiliations, funding statement, and competing-interest statement should remain unchanged from the original LaTeX file. The seven main figures should emphasize the final model objective: domain-adapted external subject-level CN/MCI/AD staging with bounded biological validation.

## Figure 1. Revised Study Framework And Central Model Objective

Purpose: Make the paper visibly different from the old v3 attention-only manuscript.

Panels:
- A: Old v3 failure points: weak AIBL external classification, poor IXI specificity, invalid attention-only CAS, non-significant Braak analysis.
- B: V6 objective: subject-level atlas-guided multimodal AD staging.
- C: Final model path: atlas MRI features + core clinical variables -> base models -> rescue ensemble -> subject-level probability averaging.
- D: Endpoint hierarchy: AIBL heldout primary, IXI specificity, OASIS stress-test limitation.
- E: Claim boundary: structural MRI proxy only; not attention biomarker, direct Braak validation, or clinical device.

## Figure 2. Data Line And Endpoint Design

Purpose: Show that the data work is a real rebuild.

Panels:
- A: ADNI train/validation/internal test inventory.
- B: AIBL adaptation train/validation and locked heldout inventory.
- C: IXI healthy negative-control cohort.
- D: OASIS stress-test cohort.
- E: Scan-to-subject endpoint aggregation and leakage control.

## Figure 3. Locked Rescue Ensemble Model

Purpose: Make the core model clear and three-class, not binary.

Panels:
- A: Six base-model probability streams.
- B: Log-probability pooling weights.
- C: Class offsets and temperature scaling.
- D: Subject-level probability averaging across repeated scans.
- E: CN/MCI/AD output with confidence and margin.

## Figure 4. External Classification Performance

Purpose: Directly answer the external classification criticism.

Panels:
- A: AIBL BAcc comparison: old v3, v4 atlas+clinical, final scan-level, final subject-level, clinical-only comparator.
- B: AIBL macro AUC comparison.
- C: Final AIBL CN/MCI/AD recall.
- D: IXI healthy CN retention and false impairment rate.
- E: OASIS stress-test mini-panel showing weak transfer as a limitation.

## Figure 5. Subject-Level Confusion And Error Profile

Purpose: Show that remaining errors are boundary-like.

Panels:
- A: AIBL heldout subject-level confusion matrix.
- B: Internal subject-level confusion matrix.
- C: AIBL true-to-predicted transition flow.
- D: Error subtype bars: MCI-to-CN, MCI-to-AD, AD-to-MCI, AD-to-CN.
- E: Error feature profiles: MMSE, hippocampus volume, ventricle volume, AD-like z-score, max probability, margin.

## Figure 6. Bootstrap Stability And Uncertainty

Purpose: Show the main result is stable enough to lock.

Panels:
- A: Bootstrap BAcc distribution with 95% CI.
- B: Bootstrap MCI recall distribution.
- C: Bootstrap AD recall distribution.
- D: Forest plot for BAcc, macro AUC, MCI recall, AD recall.
- E: Tuning boundary note: OASIS excluded from model selection.

## Figure 7. CAS/Braak Replacement And Biological Consistency

Purpose: Solve the CAS/Braak reviewer criticism without overclaiming.

Panels:
- A: Old attention-only CAS failure callout.
- B: A priori AD-key atlas regions: bilateral hippocampus, amygdala, lateral ventricles.
- C: AIBL heldout score 0.510 vs uniform null 0.286, CI 0.479-0.526, p=0.026.
- D: All-labeled AD score 0.426 vs null 0.286; ADNI-only non-significant check.
- E: Claim-boundary badges: MRI proxy only; not attention-map biomarker; not direct Braak-stage proof; not clinical deployment.

## What To Avoid

- Do not present OASIS as solved.
- Do not make clinical-only RF look like the central ARA-Net model.
- Do not draw attention maps as validated biomarkers.
- Do not claim direct Braak-stage validation.
- Do not imply clinical deployment readiness.
"""


def build(source: Path, output: Path, figure_plan: Path, body_path: Path | None = None) -> None:
    original = source.read_text(encoding="utf-8")
    header, endmatter = extract_shell(original)
    content = header + body_tex(body_path) + endmatter
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    figure_plan.write_text(figure_plan_md(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--figure-plan", type=Path, default=DEFAULT_FIGURE_PLAN)
    args = parser.parse_args()
    build(args.source, args.output, args.figure_plan, args.body)
    print(f"[saved] {args.output}")
    print(f"[saved] {args.figure_plan}")


if __name__ == "__main__":
    main()
