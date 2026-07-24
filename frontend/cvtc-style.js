const views = {
  fused: {
    label: "PYVISTA / VTK",
    image: "assets/vtk_niivue/aibl10_pyvista_fused_review.png",
    title: "AIBL_10_I164086",
    readout: ["labels 17/53 extracted as mesh", "labels 4/43 extracted as mesh"],
    points: [
      ["Hippocampus", "17 / 53", "7,198", "Segmentation label mesh"],
      ["Amygdala", "18 / 54", "2,672", "Segmentation label mesh"],
      ["Lateral ventricles", "4 / 43", "84,812", "Segmentation label mesh"],
      ["Brain tissue mask", "seg > 0", "1,609,295", "marching_cubes surface"],
    ],
  },
  shell: {
    label: "VTK BRAIN SHELL",
    image: "assets/vtk_niivue/aibl10_pyvista_vtk_brain_shell.png",
    title: "AIBL_10_I164086",
    readout: ["segmentation-derived tissue surface", "mesh label positions retained"],
    points: [
      ["Brain tissue mask", "seg > 0", "1,609,295", "marching_cubes mesh"],
      ["Hippocampus ROI", "17 / 53", "7,198", "surface label anchor"],
      ["Amygdala ROI", "18 / 54", "2,672", "surface label anchor"],
      ["Ventricle ROI", "4 / 43", "84,812", "surface label anchor"],
    ],
  },
  statmap: {
    label: "NILEARN STAT-MAP",
    image: "assets/vtk_niivue/aibl10_nilearn_stat_map.png",
    title: "AIBL_10_I164086",
    readout: ["orthogonal T1 support map", "publication-style statistical overlay"],
    points: [
      ["Axial overlay", "x / y / z", "orthogonal", "Nilearn stat-map panel"],
      ["Coronal overlay", "x / y / z", "orthogonal", "Nilearn stat-map panel"],
      ["Sagittal overlay", "x / y / z", "orthogonal", "Nilearn stat-map panel"],
      ["ATE colorbar", "AD-key", "continuous", "publication scale"],
    ],
  },
  glass: {
    label: "NILEARN GLASS BRAIN",
    image: "assets/vtk_niivue/aibl10_nilearn_glass_brain.png",
    title: "AIBL_10_I164086",
    readout: ["whole-brain projection", "supplementary glass-brain view"],
    points: [
      ["Whole-brain ATE", "AD-key", "projection", "Nilearn glass-brain panel"],
      ["Left projection", "left", "projection", "glass-brain support"],
      ["Central projection", "center", "projection", "glass-brain support"],
      ["Right projection", "right", "projection", "glass-brain support"],
    ],
  },
};

const regionChanges = [
  {
    region: "Hippocampus",
    prior: "Medial temporal atrophy",
    cn: 0.00828,
    mci: 0.007776,
    ad: 0.007322,
    deltaPercent: -11.57,
    effect: -1.233,
    direction: "decrease",
    unit: "volume ratio",
  },
  {
    region: "Amygdala",
    prior: "Medial temporal atrophy",
    cn: 0.003306,
    mci: 0.003039,
    ad: 0.002851,
    deltaPercent: -13.78,
    effect: -1.15,
    direction: "decrease",
    unit: "volume ratio",
  },
  {
    region: "Lateral ventricles",
    prior: "Ventricular enlargement",
    cn: 0.032151,
    mci: 0.043443,
    ad: 0.057479,
    deltaPercent: 78.78,
    effect: 1.64,
    direction: "increase",
    unit: "volume ratio",
  },
  {
    region: "Cortex",
    prior: "Broad cortical reserve",
    cn: 0.448491,
    mci: 0.435161,
    ad: 0.440197,
    deltaPercent: -1.85,
    effect: -0.453,
    direction: "decrease",
    unit: "volume ratio",
  },
  {
    region: "Atlas AD-like z",
    prior: "Global atlas score",
    cn: -0.058102,
    mci: 0.605236,
    ad: 1.250468,
    deltaPercent: null,
    effect: 1.8,
    direction: "increase",
    unit: "z score",
  },
];

const cases = {
  ixi: {
    label: "CN",
    confidence: 0.628,
    margin: 0.337,
    runnerUp: "MCI",
    probs: { CN: 0.628, MCI: 0.291, AD: 0.082 },
    report: "ARA-Net classifies IXI002-Guys-0828-T1 as CN with moderate confidence. The central plate shows a PyVista/VTK segmentation-derived brain tissue mesh with diffuse ATE overlay and atlas labels; it is a research visualization, not a lesion mask or clinical diagnosis.",
  },
  "aibl-ad": {
    label: "AD",
    confidence: 0.963,
    margin: 0.938,
    runnerUp: "MCI",
    probs: { CN: 0.012, MCI: 0.025, AD: 0.963 },
    report: "ARA-Net classifies AIBL_10_I164086 as AD with high confidence. The central plate uses a high-resolution FastSurfer-derived PyVista/VTK mesh with diffuse ATE-style evidence and explicit hippocampus, amygdala, and lateral-ventricle labels; it is a research visualization, not a lesion mask or clinical diagnosis.",
  },
  "aibl-mci": {
    label: "MCI",
    confidence: 0.686,
    margin: 0.515,
    runnerUp: "AD",
    probs: { CN: 0.143, MCI: 0.686, AD: 0.171 },
    report: "ARA-Net classifies DEMO-AIBL-MCI as MCI. The interface should be read as a boundary-stage review aid and kept inside the research-use boundary.",
  },
};

const el = {
  caseTitle: document.getElementById("caseTitle"),
  coordinateBody: document.getElementById("coordinateBody"),
  confidenceText: document.getElementById("confidenceText"),
  heatmapLayer: document.getElementById("heatmapLayer"),
  hippoReadout: document.getElementById("hippoReadout"),
  labelBadge: document.getElementById("labelBadge"),
  marginText: document.getElementById("marginText"),
  maskToggle: document.getElementById("maskToggle"),
  opacityControl: document.getElementById("opacityControl"),
  planeLabel: document.getElementById("planeLabel"),
  probBars: document.getElementById("probBars"),
  processButton: document.getElementById("processButton"),
  radiusControl: document.getElementById("radiusControl"),
  regionChangeGrid: document.getElementById("regionChangeGrid"),
  reportText: document.getElementById("reportText"),
  scanImage: document.getElementById("scanImage"),
  saveButton: document.getElementById("saveButton"),
  scanStage: document.getElementById("scanStage"),
  sliceNote: document.getElementById("sliceNote"),
  ventricleReadout: document.getElementById("ventricleReadout"),
};

const state = {
  view: "fused",
  case: "aibl-ad",
  mask: true,
  slice: "axial",
};

function fmt(value) {
  return Number(value).toFixed(3);
}

function percent(value) {
  return `${Math.round(Math.max(0, Math.min(1, Number(value))) * 100)}%`;
}

function signed(value, digits = 2) {
  if (!Number.isFinite(value)) return "NA";
  const fixed = Math.abs(value).toFixed(digits);
  return `${value > 0 ? "+" : value < 0 ? "-" : ""}${fixed}`;
}

function colorFor(label) {
  return { CN: "#0f766e", MCI: "#b45309", AD: "#b42318" }[label] || "#2563eb";
}

function renderView() {
  const view = views[state.view];
  el.scanImage.src = view.image;
  el.caseTitle.textContent = view.title;
  el.planeLabel.textContent = view.label;
  el.hippoReadout.textContent = view.readout[0];
  el.ventricleReadout.textContent = view.readout[1];
  el.scanStage.dataset.view = state.view;
  el.coordinateBody.innerHTML = view.points.map(([region, x, y, evidence]) => `
    <tr>
      <td>${region}</td>
      <td>${x}</td>
      <td>${y}</td>
      <td>${evidence}</td>
    </tr>
  `).join("");
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
  });
}

function renderSliceFocus() {
  const labels = {
    axial: "Axial T1 slice selected for hippocampus and ventricle cross-check. The central image remains the 3D PyVista/VTK render.",
    coronal: "Coronal T1 slice selected for bilateral medial-temporal cross-check. The central image remains the 3D PyVista/VTK render.",
    sagittal: "Sagittal T1 slice selected for side-projection anatomy cross-check. The central image remains the 3D PyVista/VTK render.",
  };
  if (el.sliceNote) el.sliceNote.textContent = labels[state.slice] || labels.axial;
  document.querySelectorAll("[data-slice]").forEach((button) => {
    button.classList.toggle("active", button.dataset.slice === state.slice);
  });
}

function renderCase() {
  const current = cases[state.case];
  const color = colorFor(current.label);
  el.labelBadge.textContent = `Predicted ${current.label}`;
  el.labelBadge.style.color = color;
  el.confidenceText.textContent = `${fmt(current.confidence)} confidence`;
  el.marginText.textContent = `Margin ${fmt(current.margin)} over ${current.runnerUp}`;
  el.reportText.textContent = current.report;
  el.probBars.innerHTML = Object.entries(current.probs).map(([label, value]) => `
    <div class="prob-row">
      <span>${label}</span>
      <i><em style="width: ${percent(value)}"></em></i>
      <strong>${fmt(value)}</strong>
    </div>
  `).join("");
  document.querySelectorAll("[data-case]").forEach((button) => {
    button.classList.toggle("active", button.dataset.case === state.case);
  });
}

function renderRegionChanges() {
  if (!el.regionChangeGrid) return;
  el.regionChangeGrid.innerHTML = regionChanges.map((item) => {
    const values = [item.cn, item.mci, item.ad];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const left = values.map((value) => 9 + ((value - min) / span) * 82);
    const delta = item.deltaPercent === null ? `z ${signed(item.ad - item.cn, 3)}` : `${signed(item.deltaPercent, 1)}%`;
    const directionText = item.direction === "increase" ? "AD higher" : "AD lower";
    return `
      <article class="region-card ${item.direction}">
        <div class="region-card-head">
          <span>${item.region}</span>
          <strong>${directionText}</strong>
        </div>
        <div class="region-spark" role="img" aria-label="${item.region} CN ${item.cn}, MCI ${item.mci}, AD ${item.ad}">
          <i style="left: ${left[0]}%"></i>
          <i style="left: ${left[1]}%"></i>
          <i style="left: ${left[2]}%"></i>
          <b style="left: ${Math.min(left[0], left[2])}%; width: ${Math.abs(left[2] - left[0])}%"></b>
        </div>
        <div class="region-values">
          <span>CN ${item.cn.toFixed(4)}</span>
          <span>MCI ${item.mci.toFixed(4)}</span>
          <span>AD ${item.ad.toFixed(4)}</span>
        </div>
        <div class="region-delta">
          <span>${item.prior}</span>
          <strong>${delta} | d=${signed(item.effect, 2)}</strong>
        </div>
      </article>
    `;
  }).join("");
}

function updateHeatmapControls() {
  const opacity = Number(el.opacityControl.value) / 100;
  const radius = Number(el.radiusControl.value);
  const edge = (radius - 8) / 26;
  el.scanImage.style.filter = `contrast(${(1.02 + edge * 0.16).toFixed(3)}) saturate(${(1.02 + opacity * 0.18).toFixed(3)})`;
  el.maskToggle.classList.toggle("active", state.mask);
  el.maskToggle.textContent = state.mask ? "Segmentation mesh" : "Minimal overlay";
}

function flashButton(button, label) {
  const original = button.textContent;
  button.textContent = label;
  window.setTimeout(() => {
    button.textContent = original;
  }, 1200);
}

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
  return Promise.resolve();
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    renderView();
  });
});

document.querySelectorAll("[data-case]").forEach((button) => {
  button.addEventListener("click", () => {
    state.case = button.dataset.case;
    renderCase();
  });
});

document.querySelectorAll("[data-slice]").forEach((button) => {
  button.addEventListener("click", () => {
    state.slice = button.dataset.slice;
    renderSliceFocus();
  });
});

el.opacityControl.addEventListener("input", updateHeatmapControls);
el.radiusControl.addEventListener("input", updateHeatmapControls);
el.maskToggle.addEventListener("click", () => {
  state.mask = !state.mask;
  updateHeatmapControls();
});
el.processButton.addEventListener("click", () => flashButton(el.processButton, "ARA-Net ready"));
el.saveButton.addEventListener("click", () => flashButton(el.saveButton, "Plate exported"));
document.getElementById("copyReportButton").addEventListener("click", async (event) => {
  await copyText(el.reportText.textContent);
  flashButton(event.currentTarget, "Copied");
});

renderView();
renderCase();
renderSliceFocus();
renderRegionChanges();
updateHeatmapControls();
