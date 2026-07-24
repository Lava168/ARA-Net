const CLASS_ORDER = ["CN", "MCI", "AD"];
const LOW_MARGIN_THRESHOLD = 0.12;

const figures = [
  {
    id: "external",
    title: "External rescue",
    caption: "AIBL locked heldout, IXI negative-control specificity, and OASIS domain-shift limitation are presented together so the external claim is not overstated.",
    file: "assets/v6_final_model/figures/figure2_final_external_rescue.png",
  },
  {
    id: "study-design",
    title: "Evidence chain",
    caption: "Leakage-free cohort flow, locked model selection, subject-level endpoint definition, and evidence layers for the revised manuscript.",
    file: "assets/v6_final_model/figures/figure1_overall_evidence_chain_study_design.png",
  },
  {
    id: "feature-system",
    title: "Atlas-guided feature system",
    caption: "Atlas, clinical, biomarker, and probability-stream evidence used by the V6 final model package.",
    file: "assets/v6_final_model/figures/figure2_atlas_guided_multimodal_feature_system.png",
  },
  {
    id: "aggregation",
    title: "RC-SPE aggregation",
    caption: "Subject-level probability aggregation and lightweight rescue ensemble head for repeated MRI scans.",
    file: "assets/v6_final_model/figures/figure3_rcspe_architecture_nbe_style.png",
  },
  {
    id: "confusion",
    title: "Subject confusion",
    caption: "AIBL subject-level confusion matrix shows stable CN and AD recognition, with residual uncertainty mainly around the MCI/AD boundary.",
    file: "assets/v6_final_model/figures/figure3_final_subject_confusion.png",
  },
  {
    id: "stability",
    title: "Bootstrap stability",
    caption: "Bootstrap and sensitivity evidence for the locked final model.",
    file: "assets/v6_final_model/figures/figure4_final_bootstrap_stability.png",
  },
  {
    id: "error-profiles",
    title: "Case error profiles",
    caption: "Representative real FastSurfer cases show CN-correct, AD-correct, AD-to-MCI, and MCI-to-AD anatomical differences.",
    file: "assets/v6_final_model/figures/figure5_final_error_profiles.png",
  },
  {
    id: "ui-exhibit",
    title: "Clinical exhibit UI",
    caption: "UI screenshot from the V6 evidence pack: prediction probability, model boundary, case-level evidence, and review language.",
    file: "assets/v6_final_model/figures/figure_ui_clinical_exhibit_screenshot.png",
  },
];

const cases = [
  {
    id: "AIBL_14",
    label: "CN correct",
    trueLabel: "CN",
    predLabel: "CN",
    probs: { CN: 0.9252389641919763, MCI: 0.06141841636914536, AD: 0.013342619438878461 },
    facts: {
      Age: "66.8",
      MMSE: "30.0",
      "AD-like z": "-0.700",
      "Hippocampus vol": "0.00929",
      "Ventricle vol": "0.01696",
      "Scans": "5",
    },
    narrative: "Control-like case with high CN probability, low AD probability, and low AD-like atlas z. It anchors the negative end of the Figure 5 case montage.",
  },
  {
    id: "AIBL_10",
    label: "AD correct",
    trueLabel: "AD",
    predLabel: "AD",
    probs: { CN: 0.011933553208902957, MCI: 0.025390642292583143, AD: 0.9626758044985139 },
    facts: {
      Age: "82.0",
      MMSE: "21.0",
      "CDR-SB": "1.0",
      "AD-like z": "2.292",
      "Hippocampus vol": "0.00678",
      "Ventricle vol": "0.08246",
      "Scans": "1",
    },
    narrative: "High-confidence AD case with strong AD-like atlas signal and enlarged lateral ventricles. This is the clean positive reference in the case montage.",
  },
  {
    id: "AIBL_1368",
    label: "AD -> MCI",
    trueLabel: "AD",
    predLabel: "MCI",
    probs: { CN: 0.18913741091777217, MCI: 0.5543338383877341, AD: 0.2565287506944937 },
    facts: {
      Age: "83.0",
      MMSE: "24.0",
      "CDR-SB": "0.5",
      "AD-like z": "-0.538",
      "Hippocampus vol": "0.00887",
      "Ventricle vol": "0.03145",
      "Scans": "1",
    },
    narrative: "Residual AD-to-MCI boundary error. The model does not collapse this AD case into CN; uncertainty stays in the adjacent disease stage.",
  },
  {
    id: "AIBL_1020",
    label: "MCI -> AD",
    trueLabel: "MCI",
    predLabel: "AD",
    probs: { CN: 0.017848381450815316, MCI: 0.0560655849005306, AD: 0.926086033648654 },
    facts: {
      Age: "75.0",
      MMSE: "25.0",
      "CDR-SB": "1.0",
      "AD-like z": "1.453",
      "Hippocampus vol": "0.00558",
      "Ventricle vol": "0.02367",
      "Scans": "1",
    },
    narrative: "MCI-to-AD boundary error with high AD probability and AD-like atlas z. This is useful for explaining why MCI remains the main uncertainty source.",
  },
  {
    id: "IXI002-Guys-0828-T1",
    label: "IXI real CN",
    trueLabel: "CN",
    predLabel: "CN",
    probs: { CN: 0.6275849833934992, MCI: 0.2907514601543644, AD: 0.08166355645213642 },
    facts: {
      Dataset: "IXI negative control",
      "Prediction unit": "subject",
      "Margin": "0.337",
      "NIfTI": "server cached",
      "Render stack": "PyVista / VTK / Nilearn / NiiVue",
    },
    narrative: "Real external healthy-control case with visible MRI-derived evidence renders. It demonstrates the UI path without turning the render into a clinical lesion claim.",
  },
];

const els = {
  casePicker: document.getElementById("casePicker"),
  caseTitle: document.getElementById("caseTitle"),
  caseNarrative: document.getElementById("caseNarrative"),
  caseFacts: document.getElementById("caseFacts"),
  probStack: document.getElementById("probStack"),
  figureNav: document.getElementById("figureNav"),
  figureFocusImage: document.getElementById("figureFocusImage"),
  figureFocusCaption: document.getElementById("figureFocusCaption"),
  confusionMatrix: document.getElementById("confusionMatrix"),
  confusionReading: document.getElementById("confusionReading"),
  errorBars: document.getElementById("errorBars"),
  errorTable: document.getElementById("errorTable"),
  caseStage: document.querySelector(".case-stage"),
  analysisCsv: document.getElementById("analysisCsv"),
  analysisDrop: document.getElementById("analysisDrop"),
  analysisFileName: document.getElementById("analysisFileName"),
  analysisFileMeta: document.getElementById("analysisFileMeta"),
  analysisMri: document.getElementById("analysisMri"),
  mriAnalysisDrop: document.getElementById("mriAnalysisDrop"),
  mriAnalysisName: document.getElementById("mriAnalysisName"),
  mriAnalysisMeta: document.getElementById("mriAnalysisMeta"),
  loadDemoButton: document.getElementById("loadDemoButton"),
  clearUploadButton: document.getElementById("clearUploadButton"),
  uploadMode: document.getElementById("uploadMode"),
  uploadResultCard: document.getElementById("uploadResultCard"),
  uploadSubject: document.getElementById("uploadSubject"),
  uploadNarrative: document.getElementById("uploadNarrative"),
  uploadPredLabel: document.getElementById("uploadPredLabel"),
  uploadProbStack: document.getElementById("uploadProbStack"),
  uploadRows: document.getElementById("uploadRows"),
  uploadSubjects: document.getElementById("uploadSubjects"),
  uploadConfidence: document.getElementById("uploadConfidence"),
  uploadLowMargin: document.getElementById("uploadLowMargin"),
  uploadTable: document.getElementById("uploadTable"),
};

const demoUploadRows = [
  {
    subject_id: "IXI002-Guys-0828-T1",
    scan_id: "IXI002-Guys-0828-T1__subject_mean",
    prob_CN: "0.6275849833934992",
    prob_MCI: "0.2907514601543644",
    prob_AD: "0.08166355645213642",
  },
];

function fmt(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "NA";
}

function pct(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "NA";
}

function classFromProbabilities(probs) {
  return CLASS_ORDER
    .map((label) => ({ label, value: Number(probs[label] || 0) }))
    .sort((a, b) => b.value - a.value);
}

function average(values) {
  const valid = values.map(Number).filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : 0;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  const input = text.replace(/^\uFEFF/, "");

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    const next = input[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(field);
      if (row.some((cell) => cell.trim() !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    if (row.some((cell) => cell.trim() !== "")) rows.push(row);
  }

  const headers = rows.shift() || [];
  return rows.map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""])));
}

async function loadCsv(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return parseCsv(await response.text());
}

function renderCasePicker(activeId = cases[0].id) {
  els.casePicker.innerHTML = cases.map((entry) => `
    <button class="case-option ${entry.id === activeId ? "active" : ""}" type="button" data-case="${escapeHtml(entry.id)}">
      <span>
        <strong>${escapeHtml(entry.id)}</strong>
        <span>${escapeHtml(entry.label)}</span>
      </span>
      <em class="label-chip ${escapeHtml(entry.predLabel)}">${escapeHtml(entry.predLabel)}</em>
    </button>
  `).join("");

  els.casePicker.querySelectorAll("[data-case]").forEach((button) => {
    button.addEventListener("click", () => renderCase(button.dataset.case));
  });
}

function renderCase(caseId) {
  const entry = cases.find((item) => item.id === caseId) || cases[0];
  renderCasePicker(entry.id);
  els.caseTitle.textContent = `${entry.id}: ${entry.trueLabel} -> ${entry.predLabel}`;
  els.caseNarrative.textContent = entry.narrative;

  els.probStack.innerHTML = CLASS_ORDER.map((label) => {
    const value = entry.probs[label] || 0;
    return `
      <div class="prob-row">
        <span>${label}</span>
        <span class="prob-track"><i class="prob-fill ${label}" style="width: ${Math.max(0, Math.min(100, value * 100))}%"></i></span>
        <strong>${fmt(value)}</strong>
      </div>
    `;
  }).join("");

  els.caseFacts.innerHTML = Object.entries(entry.facts).map(([key, value]) => `
    <div>
      <dt>${escapeHtml(key)}</dt>
      <dd>${escapeHtml(value)}</dd>
    </div>
  `).join("");

  if (els.caseStage) {
    els.caseStage.classList.remove("updated");
    window.requestAnimationFrame(() => els.caseStage.classList.add("updated"));
  }
}

function renderFigureNav(activeId = figures[0].id) {
  els.figureNav.innerHTML = figures.map((figure) => `
    <button class="figure-button ${figure.id === activeId ? "active" : ""}" type="button" data-figure="${escapeHtml(figure.id)}">
      <strong>${escapeHtml(figure.title)}</strong>
      <span>${escapeHtml(figure.caption)}</span>
    </button>
  `).join("");

  els.figureNav.querySelectorAll("[data-figure]").forEach((button) => {
    button.addEventListener("click", () => renderFigure(button.dataset.figure));
  });
}

function renderFigure(figureId) {
  const figure = figures.find((item) => item.id === figureId) || figures[0];
  renderFigureNav(figure.id);
  els.figureFocusImage.classList.add("switching");
  window.setTimeout(() => {
    els.figureFocusImage.src = figure.file;
    els.figureFocusImage.alt = figure.title;
    els.figureFocusImage.addEventListener("load", () => {
      els.figureFocusImage.classList.remove("switching");
    }, { once: true });
  }, 120);
  els.figureFocusCaption.textContent = figure.caption;
}

function probabilityColumnsFor(row, label) {
  const suffix = `prob_${label}`.toLowerCase();
  return Object.keys(row).filter((key) => key.toLowerCase().endsWith(suffix));
}

function predictionFromRow(row, index) {
  const direct = Object.fromEntries(CLASS_ORDER.map((label) => [label, Number(row[`prob_${label}`])]));
  const hasDirect = CLASS_ORDER.every((label) => Number.isFinite(direct[label]));
  const probs = hasDirect
    ? direct
    : Object.fromEntries(CLASS_ORDER.map((label) => [
      label,
      average(probabilityColumnsFor(row, label).map((column) => row[column])),
    ]));

  const total = CLASS_ORDER.reduce((sum, label) => sum + Number(probs[label] || 0), 0);
  const normalized = total > 0
    ? Object.fromEntries(CLASS_ORDER.map((label) => [label, Number(probs[label] || 0) / total]))
    : { CN: 0, MCI: 0, AD: 0 };

  return {
    subject_id: row.subject_id || row.subject || row.Subject || `uploaded_subject_${index + 1}`,
    scan_id: row.scan_id || row.scan || row.Scan || `scan_${index + 1}`,
    probs: normalized,
  };
}

function aggregateSubjects(rows) {
  const scanPredictions = rows.map(predictionFromRow);
  const grouped = new Map();

  scanPredictions.forEach((row) => {
    if (!grouped.has(row.subject_id)) grouped.set(row.subject_id, []);
    grouped.get(row.subject_id).push(row);
  });

  return [...grouped.entries()].map(([subjectId, subjectRows]) => {
    const probs = Object.fromEntries(CLASS_ORDER.map((label) => [
      label,
      average(subjectRows.map((row) => row.probs[label])),
    ]));
    const ranked = classFromProbabilities(probs);
    return {
      subject_id: subjectId,
      n_scans: subjectRows.length,
      predLabel: ranked[0].label,
      confidence: ranked[0].value,
      margin: ranked[0].value - ranked[1].value,
      probs,
    };
  }).sort((a, b) => b.confidence - a.confidence);
}

function renderUploadProbs(probs) {
  els.uploadProbStack.innerHTML = CLASS_ORDER.map((label) => {
    const value = Number(probs[label] || 0);
    return `
      <div class="prob-row">
        <span>${label}</span>
        <span class="prob-track"><i class="prob-fill ${label}" style="width: ${Math.max(0, Math.min(100, value * 100))}%"></i></span>
        <strong>${fmt(value)}</strong>
      </div>
    `;
  }).join("");
}

function setPipeline(activeSteps) {
  document.querySelectorAll(".analysis-step").forEach((step) => {
    step.classList.toggle("active", activeSteps.includes(step.dataset.step));
  });
}

function animatePipeline() {
  const steps = ["file", "prob", "subject", "review"];
  setPipeline([]);
  steps.forEach((_, index) => {
    window.setTimeout(() => setPipeline(steps.slice(0, index + 1)), index * 110);
  });
}

function renderUploadAnalysis(rows, sourceName = "Uploaded CSV") {
  const subjects = aggregateSubjects(rows);
  const focus = subjects[0];
  const lowMarginCount = subjects.filter((row) => row.margin < LOW_MARGIN_THRESHOLD).length;

  els.analysisFileName.textContent = sourceName;
  els.analysisFileMeta.textContent = `${rows.length} scan rows, ${subjects.length} subject endpoints`;
  els.uploadRows.textContent = String(rows.length);
  els.uploadSubjects.textContent = String(subjects.length);
  els.uploadLowMargin.textContent = String(lowMarginCount);

  if (!focus) {
    els.uploadSubject.textContent = "No analyzable rows";
    els.uploadNarrative.textContent = "The uploaded file did not contain CN/MCI/AD probability columns.";
    els.uploadPredLabel.textContent = "-";
    els.uploadPredLabel.className = "large-label";
    els.uploadConfidence.textContent = "-";
    els.uploadProbStack.innerHTML = "";
    els.uploadTable.innerHTML = "";
    return;
  }

  els.uploadMode.textContent = sourceName.includes("demo") || sourceName.includes("IXI002") ? "Demo replay" : "Uploaded";
  els.uploadSubject.textContent = focus.subject_id;
  els.uploadPredLabel.textContent = focus.predLabel;
  els.uploadPredLabel.className = `large-label ${focus.predLabel}`;
  els.uploadConfidence.textContent = fmt(focus.confidence);
  els.uploadNarrative.textContent = `${focus.n_scans} scan row${focus.n_scans === 1 ? "" : "s"} aggregated to a subject-level ${focus.predLabel} endpoint with margin ${fmt(focus.margin)}.`;
  renderUploadProbs(focus.probs);

  els.uploadTable.innerHTML = subjects.slice(0, 8).map((row) => `
    <tr>
      <td>${escapeHtml(row.subject_id)}</td>
      <td><span class="label-chip ${escapeHtml(row.predLabel)}">${escapeHtml(row.predLabel)}</span></td>
      <td>${fmt(row.probs.CN)}</td>
      <td>${fmt(row.probs.MCI)}</td>
      <td>${fmt(row.probs.AD)}</td>
    </tr>
  `).join("");

  animatePipeline();
  els.uploadResultCard.classList.remove("updated", "processing");
  window.requestAnimationFrame(() => els.uploadResultCard.classList.add("updated"));
}

async function handleCsvFile(file) {
  if (!file) return;
  els.uploadResultCard.classList.add("processing");
  try {
    const text = await file.text();
    const rows = parseCsv(text);
    renderUploadAnalysis(rows, file.name);
  } catch (error) {
    els.analysisFileName.textContent = file.name;
    els.analysisFileMeta.textContent = "Could not parse CSV";
    els.uploadSubject.textContent = "Upload error";
    els.uploadNarrative.textContent = error.message;
  } finally {
    window.setTimeout(() => els.uploadResultCard.classList.remove("processing"), 460);
  }
}

function clearUploadAnalysis() {
  setPipeline(["file"]);
  els.uploadMode.textContent = "Local preview";
  els.analysisFileName.textContent = "Probability stream CSV";
  els.analysisFileMeta.textContent = "No file loaded";
  els.uploadSubject.textContent = "Awaiting upload";
  els.uploadNarrative.textContent = "Upload a probability CSV or load the V6 demo case.";
  els.uploadPredLabel.textContent = "-";
  els.uploadPredLabel.className = "large-label";
  els.uploadRows.textContent = "0";
  els.uploadSubjects.textContent = "0";
  els.uploadConfidence.textContent = "-";
  els.uploadLowMargin.textContent = "0";
  renderUploadProbs({ CN: 0, MCI: 0, AD: 0 });
  els.uploadTable.innerHTML = `<tr><td colspan="5">No uploaded rows.</td></tr>`;
}

function wireDropzone(dropzone, callback) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    callback(file);
  });
}

function wireUploadControls() {
  els.analysisCsv.addEventListener("change", () => handleCsvFile(els.analysisCsv.files[0]));
  els.loadDemoButton.addEventListener("click", () => renderUploadAnalysis(demoUploadRows, "IXI002 demo probability CSV"));
  els.clearUploadButton.addEventListener("click", clearUploadAnalysis);

  els.analysisMri.addEventListener("change", () => {
    const file = els.analysisMri.files[0];
    if (!file) return;
    els.mriAnalysisName.textContent = file.name;
    els.mriAnalysisMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB staged for imaging review`;
    els.mriAnalysisDrop.classList.add("dragging");
    window.setTimeout(() => els.mriAnalysisDrop.classList.remove("dragging"), 520);
  });

  wireDropzone(els.analysisDrop, handleCsvFile);
  wireDropzone(els.mriAnalysisDrop, (file) => {
    if (!file) return;
    els.mriAnalysisName.textContent = file.name;
    els.mriAnalysisMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB staged for imaging review`;
  });
}

function renderConfusion(rows) {
  const byTrue = Object.fromEntries(CLASS_ORDER.map((label) => [label, {}]));
  rows.forEach((row) => {
    byTrue[row.true_label][row.pred_label] = row;
  });

  const body = CLASS_ORDER.map((truth) => {
    const cells = CLASS_ORDER.map((pred) => {
      const row = byTrue[truth][pred] || { n: 0, rate_within_true: 0 };
      return `<td><strong>${escapeHtml(row.n)}</strong><span>${pct(row.rate_within_true)}</span></td>`;
    }).join("");
    return `<tr><th>${truth}</th>${cells}</tr>`;
  }).join("");

  els.confusionMatrix.innerHTML = `
    <table class="matrix">
      <thead>
        <tr><th>True / Pred</th>${CLASS_ORDER.map((label) => `<th>${label}</th>`).join("")}</tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;

  const mciAd = rows.find((row) => row.true_label === "MCI" && row.pred_label === "AD");
  const adCn = rows.find((row) => row.true_label === "AD" && row.pred_label === "CN");
  els.confusionReading.textContent = `MCI-to-AD is ${pct(mciAd?.rate_within_true)}, while AD-to-CN is ${pct(adCn?.rate_within_true)}. This supports a boundary-error narrative rather than severe disease-to-normal collapse.`;
}

function renderErrorBars(rows) {
  const values = rows.map((row) => Number(row.atlas_ad_like_z_mean)).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;

  els.errorBars.innerHTML = rows.map((row) => {
    const value = Number(row.atlas_ad_like_z_mean);
    const width = Number.isFinite(value) ? ((value - min) / span) * 88 + 8 : 0;
    return `
      <div class="group-bar">
        <span>${escapeHtml(row.group)}</span>
        <span class="group-track"><i class="group-fill" style="width: ${width}%"></i></span>
        <strong>${fmt(value, 2)}</strong>
      </div>
    `;
  }).join("");
}

function renderErrorTable(rows) {
  els.errorTable.innerHTML = rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.id)}</td>
      <td>${escapeHtml(row.trueLabel)}</td>
      <td><span class="label-chip ${escapeHtml(row.predLabel)}">${escapeHtml(row.predLabel)}</span></td>
      <td>${fmt(row.probs.CN)}</td>
      <td>${fmt(row.probs.MCI)}</td>
      <td>${fmt(row.probs.AD)}</td>
      <td>${escapeHtml(row.facts["AD-like z"] || "NA")}</td>
    </tr>
  `).join("");
}

async function renderCsvPanels() {
  try {
    const [confusionRows, errorGroupRows] = await Promise.all([
      loadCsv("assets/v6_final_model/tables/aibl_heldout_confusion_transitions.csv"),
      loadCsv("assets/v6_final_model/tables/aibl_heldout_error_group_features.csv"),
    ]);

    renderConfusion(confusionRows);
    renderErrorBars(errorGroupRows);
    renderErrorTable(cases);
  } catch (error) {
    els.confusionMatrix.textContent = "V6 analysis tables could not be loaded.";
    els.errorBars.textContent = "V6 error-group table could not be loaded.";
    renderErrorTable(cases);
  }
}

wireUploadControls();
renderUploadAnalysis(demoUploadRows, "IXI002 demo probability CSV");
renderCase(cases[0].id);
renderFigure(figures[0].id);
renderCsvPanels();
