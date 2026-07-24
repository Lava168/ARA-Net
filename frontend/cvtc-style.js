const views = {
  fused: {
    label: "PYVISTA / VTK",
    image: "assets/vtk_niivue/ixi002_pyvista_fused_review.png",
    title: "IXI002-Guys-0828-T1",
    readout: ["surface-localized medial temporal evidence", "3D atlas label retained"],
    points: [
      ["Hippocampus label", 415, 430, "PyVista surface marker"],
      ["Amygdala label", 610, 414, "ATE-adjacent atlas ROI"],
      ["Lateral ventricle shell", 508, 298, "VTK structural contour"],
      ["Diffuse ATE field", 500, 370, "transparent overlay"],
    ],
  },
  shell: {
    label: "VTK BRAIN SHELL",
    image: "assets/vtk_niivue/ixi002_pyvista_vtk_brain_shell.png",
    title: "IXI002-Guys-0828-T1",
    readout: ["translucent brain-shell surface", "mesh label positions retained"],
    points: [
      ["Brain mask surface", 500, 316, "marching_cubes mesh"],
      ["Hippocampus ROI", 392, 430, "surface label anchor"],
      ["Amygdala ROI", 620, 420, "surface label anchor"],
      ["Ventricle ROI", 508, 286, "surface label anchor"],
    ],
  },
  statmap: {
    label: "NILEARN STAT-MAP",
    image: "assets/vtk_niivue/ixi002_nilearn_stat_map.png",
    title: "IXI002-Guys-0828-T1",
    readout: ["orthogonal T1 support map", "publication-style statistical overlay"],
    points: [
      ["Axial overlay", 258, 330, "Nilearn panel"],
      ["Coronal overlay", 500, 330, "Nilearn panel"],
      ["Sagittal overlay", 740, 330, "Nilearn panel"],
      ["ATE colorbar", 880, 330, "publication scale"],
    ],
  },
  glass: {
    label: "NILEARN GLASS BRAIN",
    image: "assets/vtk_niivue/ixi002_nilearn_glass_brain.png",
    title: "IXI002-Guys-0828-T1",
    readout: ["whole-brain projection", "supplementary glass-brain view"],
    points: [
      ["Whole-brain ATE", 500, 324, "projection overlay"],
      ["Left projection", 292, 330, "glass-brain panel"],
      ["Central projection", 524, 330, "glass-brain panel"],
      ["Right projection", 742, 330, "glass-brain panel"],
    ],
  },
};

const cases = {
  ixi: {
    label: "CN",
    confidence: 0.628,
    margin: 0.337,
    runnerUp: "MCI",
    probs: { CN: 0.628, MCI: 0.291, AD: 0.082 },
    report: "ARA-Net classifies IXI002-Guys-0828-T1 as CN with moderate confidence. The central plate shows a PyVista/VTK 3D brain shell with diffuse ATE overlay and atlas labels; it is a research visualization, not a lesion mask or clinical diagnosis.",
  },
  "aibl-ad": {
    label: "AD",
    confidence: 0.852,
    margin: 0.704,
    runnerUp: "MCI",
    probs: { CN: 0.000, MCI: 0.148, AD: 0.852 },
    report: "ARA-Net classifies DEMO-AIBL-AD as AD with high confidence. The 3D plate emphasizes hippocampus, amygdala, and ventricle directions consistent with AD priors, while the 2D slices remain anatomical support.",
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
  reportText: document.getElementById("reportText"),
  scanImage: document.getElementById("scanImage"),
  saveButton: document.getElementById("saveButton"),
  scanStage: document.getElementById("scanStage"),
  sliceNote: document.getElementById("sliceNote"),
  ventricleReadout: document.getElementById("ventricleReadout"),
};

const state = {
  view: "fused",
  case: "ixi",
  mask: true,
  slice: "axial",
};

function fmt(value) {
  return Number(value).toFixed(3);
}

function percent(value) {
  return `${Math.round(Math.max(0, Math.min(1, Number(value))) * 100)}%`;
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

function updateHeatmapControls() {
  const opacity = Number(el.opacityControl.value) / 100;
  const radius = Number(el.radiusControl.value);
  el.heatmapLayer.style.opacity = opacity;
  el.heatmapLayer.style.filter = `blur(${Math.max(0, (radius - 8) / 11)}px) saturate(1.24)`;
  el.heatmapLayer.classList.toggle("mask-on", state.mask);
  el.maskToggle.classList.toggle("active", state.mask);
  el.maskToggle.textContent = state.mask ? "Brain-shell constrained" : "Full ATE field";
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
updateHeatmapControls();
