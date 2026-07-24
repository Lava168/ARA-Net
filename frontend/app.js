const CLASSES = ["CN", "MCI", "AD"];
const LOW_MARGIN_THRESHOLD = 0.12;

const sampleCsv = `subject_id,scan_id,aibl_adapted_atlas_biomarker_enhanced__hgb__prob_CN,aibl_adapted_atlas_biomarker_enhanced__hgb__prob_MCI,aibl_adapted_atlas_biomarker_enhanced__hgb__prob_AD,aibl_adapted_atlas_core_clinical__hgb__prob_CN,aibl_adapted_atlas_core_clinical__hgb__prob_MCI,aibl_adapted_atlas_core_clinical__hgb__prob_AD,aibl_adapted_clinical_biomarker_only__rf_balanced__prob_CN,aibl_adapted_clinical_biomarker_only__rf_balanced__prob_MCI,aibl_adapted_clinical_biomarker_only__rf_balanced__prob_AD,aibl_adapted_clinical_core_only__hgb__prob_CN,aibl_adapted_clinical_core_only__hgb__prob_MCI,aibl_adapted_clinical_core_only__hgb__prob_AD,aibl_adapted_clinical_core_only__rf_balanced__prob_CN,aibl_adapted_clinical_core_only__rf_balanced__prob_MCI,aibl_adapted_clinical_core_only__rf_balanced__prob_AD,rf__logreg__prob_CN,rf__logreg__prob_MCI,rf__logreg__prob_AD
example_subject_001,example_scan_001,0.82,0.14,0.04,0.78,0.17,0.05,0.88,0.10,0.02,0.76,0.19,0.05,0.83,0.13,0.04,0.80,0.16,0.04
example_subject_002,example_scan_001,0.10,0.28,0.62,0.08,0.25,0.67,0.12,0.30,0.58,0.09,0.33,0.58,0.11,0.24,0.65,0.10,0.30,0.60
example_subject_002,example_scan_002,0.13,0.32,0.55,0.09,0.29,0.62,0.14,0.34,0.52,0.12,0.35,0.53,0.13,0.29,0.58,0.11,0.34,0.55
example_subject_003,example_scan_001,0.08,0.84,0.08,0.07,0.86,0.07,0.09,0.82,0.09,0.08,0.85,0.07,0.07,0.87,0.06,0.08,0.84,0.08`;

const state = {
  unit: "subject",
  sourceName: "",
  sourceRows: [],
  predictions: [],
  payload: null,
};

const el = {
  apiDot: byId("apiDot"),
  apiState: byId("apiState"),
  classBars: byId("classBars"),
  clinicalNotice: byId("clinicalNotice"),
  csvFile: byId("csvFile"),
  downloadButton: byId("downloadButton"),
  dropzone: byId("dropzone"),
  endpoint: byId("endpoint"),
  fileMeta: byId("fileMeta"),
  fileName: byId("fileName"),
  focusConfidence: byId("focusConfidence"),
  focusLabel: byId("focusLabel"),
  focusMargin: byId("focusMargin"),
  focusScans: byId("focusScans"),
  focusSubject: byId("focusSubject"),
  labelFilter: byId("labelFilter"),
  metricAdRecall: byId("metricAdRecall"),
  metricAuc: byId("metricAuc"),
  metricBacc: byId("metricBacc"),
  metricConfidence: byId("metricConfidence"),
  metricLowMargin: byId("metricLowMargin"),
  metricMciRecall: byId("metricMciRecall"),
  metricMode: byId("metricMode"),
  metricRows: byId("metricRows"),
  modelList: byId("modelList"),
  modelName: byId("modelName"),
  modelVersion: byId("modelVersion"),
  predictionBody: byId("predictionBody"),
  probabilityStack: byId("probabilityStack"),
  resultTitle: byId("resultTitle"),
  runButton: byId("runButton"),
  runPill: byId("runPill"),
  sampleButton: byId("sampleButton"),
  tableTitle: byId("tableTitle"),
};

if (window.location.protocol === "file:") {
  el.endpoint.value = "http://127.0.0.1:8080/predict";
}

function byId(id) {
  return document.getElementById(id);
}

function fmt(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "0%";
}

function setRunState(kind, label) {
  el.runPill.className = `run-pill ${kind || ""}`.trim();
  el.runPill.textContent = label;
}

function setApiState(kind, label) {
  el.apiDot.className = `status-dot ${kind || ""}`.trim();
  el.apiState.textContent = label;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  const normalized = text.replace(/^\uFEFF/, "");

  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index];
    const next = normalized[index + 1];
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
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(field);
      if (row.some((cell) => cell.trim() !== "")) {
        rows.push(row);
      }
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    if (row.some((cell) => cell.trim() !== "")) {
      rows.push(row);
    }
  }
  if (rows.length < 2) {
    throw new Error("CSV needs a header and at least one data row.");
  }

  const header = rows[0].map((cell) => cell.trim());
  return rows.slice(1).map((cells) => {
    const record = {};
    header.forEach((column, index) => {
      record[column] = (cells[index] ?? "").trim();
    });
    return record;
  });
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function predictionsToCsv(rows) {
  if (!rows.length) return "";
  const columns = [
    "subject_id",
    "scan_id",
    "prediction_unit",
    "n_scans",
    "predicted_label",
    "confidence",
    "margin",
    "prob_CN",
    "prob_MCI",
    "prob_AD",
    "clinical_use_notice",
  ];
  const header = columns.join(",");
  const body = rows.map((row) => columns.map((column) => csvEscape(row[column])).join(","));
  return [header, ...body].join("\n");
}

function ingestCsv(text, name) {
  const rows = parseCsv(text);
  state.sourceRows = rows;
  state.sourceName = name;
  state.predictions = [];
  state.payload = null;
  el.fileName.textContent = name;
  el.fileMeta.textContent = `${rows.length} scan rows loaded`;
  el.runButton.disabled = false;
  el.downloadButton.disabled = true;
  el.labelFilter.value = "all";
  setRunState("", "Ready");
  renderEmptyTable("No prediction rows.");
  renderSummary();
  renderFocus(null);
}

async function loadMetadata() {
  try {
    const response = await fetch("/metadata", { cache: "no-store" });
    if (!response.ok) throw new Error("metadata unavailable");
    const metadata = await response.json();
    setApiState("online", "API online");
    el.modelName.textContent = metadata.model;
    el.modelVersion.textContent = metadata.version;
    el.clinicalNotice.textContent = metadata.clinical_use_notice;
    renderEvaluation(metadata.primary_evaluation || {});
    renderBaseModels(metadata.base_models || []);
  } catch {
    setApiState("error", window.location.protocol === "file:" ? "Open via API server" : "API offline");
  }
}

function renderEvaluation(evaluation) {
  const recall = evaluation.recall || {};
  el.metricBacc.textContent = fmt(evaluation.balanced_accuracy);
  el.metricAuc.textContent = fmt(evaluation.macro_auc_ovr);
  el.metricAdRecall.textContent = fmt(recall.AD);
  el.metricMciRecall.textContent = fmt(recall.MCI);
}

function renderBaseModels(models) {
  el.modelList.innerHTML = "";
  models.forEach((model) => {
    const item = document.createElement("li");
    item.textContent = model.replaceAll("__", " / ").replaceAll("_", " ");
    el.modelList.appendChild(item);
  });
}

function renderSummary() {
  const rows = state.predictions;
  const counts = Object.fromEntries(CLASSES.map((label) => [label, 0]));
  rows.forEach((row) => {
    if (counts[row.predicted_label] != null) {
      counts[row.predicted_label] += 1;
    }
  });
  const total = rows.length || 0;
  const meanConfidence = total
    ? rows.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / total
    : null;
  const lowMargin = total
    ? rows.filter((row) => Number(row.margin || 0) < LOW_MARGIN_THRESHOLD).length
    : null;

  el.metricRows.textContent = String(total);
  el.metricConfidence.textContent = meanConfidence == null ? "-" : fmt(meanConfidence);
  el.metricLowMargin.textContent = lowMargin == null ? "-" : String(lowMargin);
  el.metricMode.textContent = state.unit[0].toUpperCase() + state.unit.slice(1);
  el.resultTitle.textContent = total ? `${total} ${state.unit}-level predictions` : "Awaiting input";

  el.classBars.innerHTML = CLASSES.map((label) => {
    const width = total ? (counts[label] / total) * 100 : 0;
    return `
      <div class="class-row">
        <span>${label}</span>
        <div><i style="width: ${width}%"></i></div>
        <strong>${counts[label]}</strong>
      </div>
    `;
  }).join("");
}

function renderFocus(row) {
  const selected = row || null;
  el.focusSubject.textContent = selected ? selected.subject_id : "No prediction";
  el.focusLabel.textContent = selected ? selected.predicted_label : "-";
  el.focusLabel.className = `label-chip ${selected ? selected.predicted_label : ""}`.trim();
  el.focusConfidence.textContent = `Confidence ${selected ? fmt(selected.confidence) : "-"}`;
  el.focusMargin.textContent = `Margin ${selected ? fmt(selected.margin) : "-"}`;
  el.focusScans.textContent = `Scans ${selected && selected.n_scans ? selected.n_scans : "-"}`;
  el.probabilityStack.innerHTML = CLASSES.map((label) => {
    const value = selected ? Number(selected[`prob_${label}`] || 0) : 0;
    return `
      <div class="prob-row">
        <span>${label}</span>
        <div><i style="width: ${percent(value)}"></i></div>
        <strong>${fmt(value)}</strong>
      </div>
    `;
  }).join("");
}

function renderEmptyTable(message, title = "No predictions") {
  el.tableTitle.textContent = title;
  el.predictionBody.innerHTML = `<tr><td colspan="8" class="empty-cell">${message}</td></tr>`;
}

function renderTable() {
  const filter = el.labelFilter.value;
  const rows = filter === "all"
    ? state.predictions
    : state.predictions.filter((row) => row.predicted_label === filter);

  el.tableTitle.textContent = state.predictions.length
    ? `${rows.length} visible of ${state.predictions.length}`
    : "No rows loaded";
  if (!rows.length) {
    renderEmptyTable(
      state.predictions.length ? "No rows match the selected label." : "No prediction rows.",
      state.predictions.length ? `0 visible of ${state.predictions.length}` : "No rows loaded",
    );
    return;
  }
  el.predictionBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.subject_id)}</td>
      <td>${escapeHtml(row.scan_id || "")}</td>
      <td><span class="prediction-label ${row.predicted_label}">${row.predicted_label}</span></td>
      <td>${fmt(row.confidence)}</td>
      <td>${fmt(row.margin)}</td>
      <td>${fmt(row.prob_CN)}</td>
      <td>${fmt(row.prob_MCI)}</td>
      <td>${fmt(row.prob_AD)}</td>
    </tr>
  `).join("");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function drawBrainCanvas(canvas) {
  const label = canvas.dataset.case || "CN";
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const severity = { CN: 0.18, MCI: 0.46, AD: 0.74 }[label] ?? 0.3;

  ctx.clearRect(0, 0, width, height);
  const background = ctx.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#0f172a");
  background.addColorStop(1, "#1f2937");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.translate(width / 2, height / 2 + 2);
  ctx.scale(1.02, 1);

  const brainGradient = ctx.createRadialGradient(0, -8, 12, 0, 0, 70);
  brainGradient.addColorStop(0, "#e5e7eb");
  brainGradient.addColorStop(0.65, "#94a3b8");
  brainGradient.addColorStop(1, "#475569");
  ctx.fillStyle = brainGradient;
  ctx.beginPath();
  ctx.ellipse(-32, 0, 46, 58, -0.12, 0, Math.PI * 2);
  ctx.ellipse(32, 0, 46, 58, 0.12, 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.ellipse(0, 2, 8 + severity * 9, 54, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(-23, 22, 7 + severity * 12, 9 + severity * 10, -0.15, 0, Math.PI * 2);
  ctx.ellipse(23, 22, 7 + severity * 12, 9 + severity * 10, 0.15, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";

  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth = 2;
  for (let offset = -48; offset <= 48; offset += 16) {
    ctx.beginPath();
    ctx.moveTo(offset, -42);
    ctx.bezierCurveTo(offset * 0.68, -18, offset * 0.68, 18, offset, 42);
    ctx.stroke();
  }

  const focus = { CN: "#22c55e", MCI: "#f59e0b", AD: "#ef4444" }[label] || "#38bdf8";
  ctx.strokeStyle = focus;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.ellipse(0, 12, 34 + severity * 14, 28 + severity * 8, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.font = "700 13px ui-sans-serif, system-ui";
  ctx.fillText(label, 14, 24);
}

function drawCaseCanvases() {
  document.querySelectorAll(".brain-canvas").forEach(drawBrainCanvas);
}

async function runPrediction() {
  if (!state.sourceRows.length) return;
  setRunState("running", "Running");
  el.runButton.disabled = true;
  el.downloadButton.disabled = true;
  try {
    const response = await fetch(el.endpoint.value.trim() || "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ unit: state.unit, rows: state.sourceRows }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }
    state.payload = payload;
    state.predictions = payload.predictions || [];
    const focus = [...state.predictions].sort((a, b) => Number(b.confidence || 0) - Number(a.confidence || 0))[0];
    el.clinicalNotice.textContent = payload.clinical_use_notice || el.clinicalNotice.textContent;
    el.downloadButton.disabled = state.predictions.length === 0;
    setRunState("success", "Complete");
    renderSummary();
    renderFocus(focus);
    renderTable();
  } catch (error) {
    setRunState("error", "Error");
    renderEmptyTable(error.message);
  } finally {
    el.runButton.disabled = state.sourceRows.length === 0;
  }
}

async function loadSample() {
  try {
    const response = await fetch("/examples/probability_input_example.csv", { cache: "no-store" });
    if (!response.ok) throw new Error("sample unavailable");
    ingestCsv(await response.text(), "probability_input_example.csv");
  } catch {
    ingestCsv(sampleCsv, "embedded_probability_sample.csv");
  }
}

function downloadPredictions() {
  const csv = predictionsToCsv(state.predictions);
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `aranet_${state.unit}_predictions.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

el.csvFile.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    ingestCsv(await file.text(), file.name);
  } catch (error) {
    setRunState("error", "CSV Error");
    renderEmptyTable(error.message);
  }
});

el.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  el.dropzone.classList.add("dragging");
});

el.dropzone.addEventListener("dragleave", () => {
  el.dropzone.classList.remove("dragging");
});

el.dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  el.dropzone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (!file) return;
  try {
    ingestCsv(await file.text(), file.name);
  } catch (error) {
    setRunState("error", "CSV Error");
    renderEmptyTable(error.message);
  }
});

document.querySelectorAll("[data-unit]").forEach((button) => {
  button.addEventListener("click", () => {
    state.unit = button.dataset.unit;
    document.querySelectorAll("[data-unit]").forEach((item) => item.classList.toggle("active", item === button));
    el.metricMode.textContent = state.unit[0].toUpperCase() + state.unit.slice(1);
  });
});

el.runButton.addEventListener("click", runPrediction);
el.sampleButton.addEventListener("click", loadSample);
el.labelFilter.addEventListener("change", renderTable);
el.downloadButton.addEventListener("click", downloadPredictions);

renderSummary();
renderFocus(null);
drawCaseCanvases();
loadMetadata();
