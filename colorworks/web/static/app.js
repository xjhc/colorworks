const state = {
  asset: null,
  output: null,
  tone_map: null,
  edge_mask: null,
  structure_tensor: null,
  orientation_field: null,
  renderTimer: null,
  sourceObjectUrl: null,
  activeTab: "source", // "source" | "tone_map" | "edge_mask" | "orientation_field" | "final"
  vectorView: "orientation_hsv",
  schemas: null,

  // Composition and Presets state
  composition: {
    paper_color: { hex: "#f4ebd9" },
    layers: []
  },
  presets: []
};

const els = {
  assetInput: document.querySelector("#assetInput"),
  assetName: document.querySelector("#assetName"),
  assetSize: document.querySelector("#assetSize"),
  assetChecksum: document.querySelector("#assetChecksum"),
  mainViewer: document.querySelector("#mainViewer"),
  noArtifactMsg: document.querySelector("#noArtifactMsg"),
  pipelineSelect: document.querySelector("#pipelineSelect"),
  dynamicControls: document.querySelector("#dynamicControls"),
  renderStatus: document.querySelector("#renderStatus"),
  renderTime: document.querySelector("#renderTime"),
  outputChecksum: document.querySelector("#outputChecksum"),
  exportLink: document.querySelector("#exportLink"),
  recipeName: document.querySelector("#recipeName"),
  recipeSelect: document.querySelector("#recipeSelect"),
  saveRecipe: document.querySelector("#saveRecipe"),
  reloadRecipe: document.querySelector("#reloadRecipe"),
  tabButtons: document.querySelectorAll(".tab-btn"),
  tabToneMap: document.querySelector("#tabToneMap"),
  tabEdgeMask: document.querySelector("#tabEdgeMask"),
  tabOrientation: document.querySelector("#tabOrientation"),
  vectorControls: document.querySelector("#vectorControls"),
  vectorViewSelect: document.querySelector("#vectorViewSelect"),
  exportSvgBtn: document.querySelector("#exportSvgBtn"),

  // Presets and Layers Elements
  presetSelect: document.querySelector("#presetSelect"),
  loadPreset: document.querySelector("#loadPreset"),
  deletePreset: document.querySelector("#deletePreset"),
  newPresetName: document.querySelector("#newPresetName"),
  savePreset: document.querySelector("#savePreset"),
  paperColorPicker: document.querySelector("#paperColorPicker"),
  addLayerBtn: document.querySelector("#addLayerBtn"),
};

function getActivePipeline() {
  return els.pipelineSelect.value;
}

function getActiveAlgorithm(mode = getActivePipeline()) {
  if (!state.schemas) return null;
  return state.schemas.algorithms.find((a) => a.id === mode) || null;
}

function modeUsesComposition(mode = getActivePipeline()) {
  if (mode === "ordered_bayer") return false;
  const algo = getActiveAlgorithm(mode);
  return !algo || algo.role !== "renderer";
}

function patternDefinition(kind) {
  if (!state.schemas || !state.schemas.patterns) return null;
  return state.schemas.patterns.find((p) => p.kind === kind) || null;
}

function artifactSourceOptions(suitableAs) {
  const algo = getActiveAlgorithm();
  const options = [];
  if (algo && algo.artifact_kinds) {
    algo.artifact_kinds.forEach((artifact) => {
      if (artifact.suitable_as && artifact.suitable_as.includes(suitableAs)) {
        options.push({ value: artifact.name, label: artifact.label || artifact.name });
      }
    });
  }
  if (suitableAs === "density_source" && !options.some((opt) => opt.value === "tone_map")) {
    options.push({ value: "tone_map", label: "Tone Map" });
  }
  return options;
}

function compositionHasStrokeLayer(composition = state.composition) {
  return !!composition && Array.isArray(composition.layers) && composition.layers.some((layer) => {
    return layer.pattern && (layer.pattern.kind === "hatch" || layer.pattern.kind === "crosshatch");
  });
}

function setStatus(message) {
  els.renderStatus.textContent = message;
}

// Read parameters from UI controls
function getParams() {
  const mode = getActivePipeline();
  const params = {};

  if (mode === "ordered_bayer") {
    const matrixEl = document.querySelector("#matrixSize");
    const threshEl = document.querySelector("#threshold");
    const contrastEl = document.querySelector("#contrast");
    return {
      matrix_size: matrixEl ? Number(matrixEl.value) : 8,
      threshold: threshEl ? Number(threshEl.value) : 0.0,
      contrast: contrastEl ? Number(contrastEl.value) : 1.0,
    };
  }

  if (state.schemas) {
    const algo = state.schemas.algorithms.find((a) => a.id === mode);
    if (algo) {
      algo.parameters.forEach((param) => {
        const el = document.querySelector(`#param-${param.key}`);
        if (el) {
          if (param.type === "bool") {
            params[param.key] = el.checked;
          } else if (param.type === "str") {
            params[param.key] = el.value;
          } else {
            params[param.key] = Number(el.value);
          }
        }
      });
    }
  }
  return params;
}

// Read composition
function getComposition() {
  const mode = getActivePipeline();
  if (!modeUsesComposition(mode)) return null;

  const preserveEl = document.querySelector("#param-preserve_edges");
  const preserve_edges = preserveEl ? preserveEl.checked : null;

  // Synchronize paper color
  if (els.paperColorPicker) {
    state.composition.paper_color.hex = els.paperColorPicker.value;
  }

  // Set mask_source dynamically on layers based on edge preservation
  state.composition.layers.forEach((layer) => {
    if (layer.pattern) {
      if (!layer.pattern.coordinates) {
        layer.pattern.coordinates = { space: "image_px" };
      }
      if (preserve_edges !== null) {
        layer.pattern.mask_source = preserve_edges ? "edge_mask" : null;
      }
      const patDef = patternDefinition(layer.pattern.kind);
      if (!patDef || !patDef.accepts_orientation) {
        layer.pattern.orientation_source = null;
      }
    }
  });

  return state.composition;
}

function updateControlLabels() {
  document.querySelectorAll(".control").forEach((ctrl) => {
    const input = ctrl.querySelector("input[type='range']");
    const output = ctrl.querySelector("output");
    if (input && output) {
      output.textContent = Number(input.value).toFixed(2);
    }
  });
}

function scheduleRender() {
  updateControlLabels();
  if (!state.asset) {
    return;
  }
  clearTimeout(state.renderTimer);
  state.renderTimer = window.setTimeout(renderNow, 90);
}

// ── iterative (Phase 3) algorithms ───────────────────────────────────────────

const ITERATIVE_ALGORITHMS = new Set(["cvt_stippling", "dbs"]);

let _activeSSE = null;
let _sessionId = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);

function isIterative(mode) {
  if (ITERATIVE_ALGORITHMS.has(mode)) return true;
  const algo = getActiveAlgorithm(mode);
  return algo && algo.execution_profile && algo.execution_profile.is_iterative;
}

async function renderIterative(mode, payload) {
  if (_activeSSE) { _activeSSE.close(); _activeSSE = null; }

  payload.session_id = _sessionId;
  const submitResp = await fetch("/api/preview_runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!submitResp.ok) {
    const err = await submitResp.json().catch(() => ({}));
    setStatus(err.error || "Submit failed");
    return;
  }
  const run = await submitResp.json();
  const runId = run.id;
  setStatus(`Running (${runId.slice(0, 10)}…)`);

  const sse = new EventSource(`/api/preview_runs/${runId}/events`);
  _activeSSE = sse;

  sse.addEventListener("iteration", (evt) => {
    const data = JSON.parse(evt.data);
    const it = data.iteration !== undefined ? data.iteration + 1 : "?";
    const energy = data.energy !== null && data.energy !== undefined
      ? data.energy.toFixed(3) : "";
    setStatus(`Iteration ${it}${energy ? "  energy=" + energy : ""}`);
  });

  sse.addEventListener("completed", (evt) => {
    const data = JSON.parse(evt.data);
    sse.close(); _activeSSE = null;
    setStatus("Complete — loading output…");
    // CVT stippling: primary_artifact_id points to stipple_points,
    // final_artifact_id points to the raster image
    const finalId = data.final_artifact_id || data.primary_artifact_id;
    if (finalId) {
      state.output = {
        checksum: finalId,
        url: `/api/artifacts/${finalId}`,
        width: state.asset.width,
        height: state.asset.height,
        render_ms: 0,
      };
      els.exportLink.href = state.output.url;
      els.exportLink.download = `colorworks-${finalId.slice(0, 12)}.png`;
      els.exportLink.classList.remove("disabled");
      updateViewer();
      refreshChrome();
      setStatus(`${state.asset.width} × ${state.asset.height}`);
    }
  });

  sse.addEventListener("cancelled", () => {
    sse.close(); _activeSSE = null;
    setStatus("Cancelled");
  });

  sse.addEventListener("failed", (evt) => {
    sse.close(); _activeSSE = null;
    const data = JSON.parse(evt.data);
    setStatus("Failed: " + (data.error || "unknown"));
  });

  sse.onerror = () => {
    sse.close(); _activeSSE = null;
  };
}

// ─────────────────────────────────────────────────────────────────────────────

async function renderNow() {
  if (!state.asset) {
    return;
  }
  setStatus("Rendering");

  const mode = getActivePipeline();
  const payload = {
    asset_id: state.asset.id,
    renderer_id: mode,
    params: getParams(),
    seed: 42,
  };
  if (modeUsesComposition(mode)) {
    payload.composition = getComposition();
  }

  if (isIterative(mode)) {
    await renderIterative(mode, payload);
    return;
  }

  const response = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const result = await response.json();
  if (!response.ok) {
    setStatus(result.error || "Render failed");
    return;
  }

  state.output = result.output;
  if (result.artifacts) {
    state.tone_map = result.artifacts.tone_map || null;
    state.edge_mask = result.artifacts.edge_mask || null;
    state.structure_tensor = result.artifacts.structure_tensor || null;
    state.orientation_field = result.artifacts.orientation_field || null;
  } else {
    state.tone_map = null;
    state.edge_mask = null;
    state.structure_tensor = null;
    state.orientation_field = null;
  }

  refreshChrome();
  updateViewer();

  els.renderTime.textContent = `${state.output.render_ms} ms`;
  els.outputChecksum.textContent = state.output.checksum.slice(0, 16);
  els.exportLink.href = state.output.url;
  els.exportLink.download = `colorworks-${state.output.checksum.slice(0, 12)}.png`;
  els.exportLink.classList.remove("disabled");
  refreshChrome();
  setStatus(`${state.output.width} x ${state.output.height}`);
}

function syncActiveTabButtons() {
  els.tabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === state.activeTab);
  });
}

function refreshChrome() {
  if (els.tabOrientation) {
    els.tabOrientation.style.display = state.orientation_field ? "" : "none";
  }
  if (!state.orientation_field && state.activeTab === "orientation_field") {
    state.activeTab = state.output ? "final" : "source";
  }
  if (els.vectorControls) {
    els.vectorControls.style.display =
      state.orientation_field && state.activeTab === "orientation_field" ? "flex" : "none";
  }
  if (els.exportSvgBtn) {
    const enabled = !!state.asset && modeUsesComposition() && compositionHasStrokeLayer();
    els.exportSvgBtn.disabled = !enabled;
    els.exportSvgBtn.classList.toggle("disabled", !enabled);
  }
  syncActiveTabButtons();
}

function updateViewer() {
  refreshChrome();
  const tab = state.activeTab;
  els.mainViewer.style.display = "none";
  els.noArtifactMsg.style.display = "none";

  if (tab === "source") {
    if (state.sourceObjectUrl) {
      els.mainViewer.src = state.sourceObjectUrl;
      els.mainViewer.style.display = "block";
    } else {
      els.noArtifactMsg.textContent = "No source image loaded";
      els.noArtifactMsg.style.display = "block";
    }
  } else if (tab === "tone_map") {
    if (state.tone_map && state.tone_map.url) {
      els.mainViewer.src = `${state.tone_map.url}?v=${state.tone_map.id}`;
      els.mainViewer.style.display = "block";
    } else {
      els.noArtifactMsg.textContent = "Tone map not generated in current pipeline";
      els.noArtifactMsg.style.display = "block";
    }
  } else if (tab === "edge_mask") {
    if (state.edge_mask && state.edge_mask.url) {
      els.mainViewer.src = `${state.edge_mask.url}?v=${state.edge_mask.id}`;
      els.mainViewer.style.display = "block";
    } else {
      els.noArtifactMsg.textContent = "Edge mask not generated (or edge preservation disabled)";
      els.noArtifactMsg.style.display = "block";
    }
  } else if (tab === "orientation_field") {
    if (state.orientation_field && state.orientation_field.url) {
      const view = state.vectorView || "orientation_hsv";
      els.mainViewer.src = `${state.orientation_field.url}?view=${encodeURIComponent(view)}&v=${state.orientation_field.id}`;
      els.mainViewer.style.display = "block";
    } else {
      els.noArtifactMsg.textContent = "Orientation field not generated in current pipeline";
      els.noArtifactMsg.style.display = "block";
    }
  } else if (tab === "final") {
    if (state.output) {
      els.mainViewer.src = `${state.output.url}?v=${state.output.checksum}`;
      els.mainViewer.style.display = "block";
    } else {
      els.noArtifactMsg.textContent = "Click render or adjust parameters to generate output";
      els.noArtifactMsg.style.display = "block";
    }
  }
}

async function uploadAsset(file) {
  if (!file) {
    return;
  }
  if (state.sourceObjectUrl) {
    URL.revokeObjectURL(state.sourceObjectUrl);
  }
  state.sourceObjectUrl = URL.createObjectURL(file);
  if (state.activeTab === "source") {
    els.mainViewer.src = state.sourceObjectUrl;
    els.mainViewer.style.display = "block";
  }
  setStatus("Waiting");

  const response = await fetch("/api/assets", {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "X-Filename": encodeURIComponent(file.name),
    },
    body: file,
  });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "Upload failed");
    return;
  }
  state.asset = payload.asset;
  els.assetName.textContent = state.asset.original_filename;
  els.assetSize.textContent = `${state.asset.width} x ${state.asset.height}`;
  els.assetChecksum.textContent = state.asset.checksum.slice(0, 16);
  scheduleRender();
}

async function saveRecipe() {
  if (!state.asset) {
    setStatus("Load a raster first");
    return;
  }
  const mode = getActivePipeline();
  const payload = {
    name: els.recipeName.value,
    asset_id: state.asset.id,
    renderer_id: mode,
    params: getParams(),
  };
  if (modeUsesComposition(mode)) {
    payload.composition = getComposition();
  }

  const response = await fetch("/api/recipes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    setStatus(result.error || "Save failed");
    return;
  }
  await loadRecipeList();
  els.recipeSelect.value = result.id;
  setStatus("Recipe saved");
}

async function loadRecipeList() {
  const response = await fetch("/api/recipes");
  const payload = await response.json();
  els.recipeSelect.innerHTML = "";
  for (const recipe of payload.recipes) {
    const option = document.createElement("option");
    option.value = recipe.id;
    option.textContent = recipe.name;
    els.recipeSelect.append(option);
  }
}

async function reloadRecipe() {
  const recipeId = els.recipeSelect.value;
  if (!recipeId) {
    return;
  }
  const response = await fetch(`/api/recipes/${recipeId}`);
  const recipe = await response.json();
  if (!response.ok) {
    setStatus(recipe.error || "Load failed");
    return;
  }

  els.recipeName.value = recipe.name;
  els.pipelineSelect.value = recipe.renderer_id || "ordered_bayer";

  // Re-generate parameters controls for the loaded pipeline mode
  renderControlsForMode(recipe.renderer_id || "ordered_bayer");

  // Populate loaded values
  const recipeMode = recipe.renderer_id || "ordered_bayer";
  if (recipeMode !== "ordered_bayer" && state.schemas) {
    const algo = state.schemas.algorithms.find((a) => a.id === recipeMode);
    if (algo) {
      algo.parameters.forEach((param) => {
        const el = document.querySelector(`#param-${param.key}`);
        if (el) {
          if (param.type === "bool") {
            el.checked = !!recipe.params[param.key];
          } else {
            el.value = recipe.params[param.key];
          }
        }
      });
      updateVisibility();
    }

    // Load composition
    if (modeUsesComposition(recipeMode) && recipe.composition && Array.isArray(recipe.composition.layers)) {
      state.composition = JSON.parse(JSON.stringify(recipe.composition));
      if (els.paperColorPicker && state.composition.paper_color) {
        els.paperColorPicker.value = state.composition.paper_color.hex;
      }
      renderLayersUI();
    }
  } else {
    // ordered_bayer
    const matrixEl = document.querySelector("#matrixSize");
    const threshEl = document.querySelector("#threshold");
    const contrastEl = document.querySelector("#contrast");

    if (matrixEl) matrixEl.value = String(recipe.params.matrix_size);
    if (threshEl) threshEl.value = String(recipe.params.threshold);
    if (contrastEl) contrastEl.value = String(recipe.params.contrast);
  }

  updateControlLabels();

  if (!state.asset || state.asset.id !== recipe.asset.id) {
    const assetResponse = await fetch(`/api/assets/${recipe.asset.id}`);
    const assetPayload = await assetResponse.json();
    if (!assetResponse.ok) {
      setStatus(assetPayload.error || "Asset missing");
      return;
    }
    state.asset = assetPayload.asset;
    state.sourceObjectUrl = `/api/assets/${state.asset.id}/image`;
    els.assetName.textContent = state.asset.original_filename;
    els.assetSize.textContent = `${state.asset.width} x ${state.asset.height}`;
    els.assetChecksum.textContent = state.asset.checksum.slice(0, 16);
  }

  scheduleRender();
}

function defaultWaveComposition() {
  return {
    paper_color: { hex: "#f4ebd9" },
    layers: [
      {
        name: "ink",
        color: { hex: "#1a1a1a" },
        role: "shadow",
        density_source: "tone_map",
        pattern: {
          kind: "wave",
          params: { frequency: 8.0, angle_deg: 45.0, phase: 0.0 },
          mask_source: "edge_mask",
          orientation_source: null,
          coordinates: { space: "image_px" }
        },
        threshold: null,
        blend_mode: "normal",
        opacity: 1.0,
        priority: 0
      }
    ]
  };
}

function defaultStructureComposition() {
  return {
    paper_color: { hex: "#f4ebd9" },
    layers: [
      {
        name: "flow_hatch",
        color: { hex: "#1a1a1a" },
        role: "shadow",
        density_source: "tone_map",
        pattern: {
          kind: "hatch",
          params: { frequency: 8.0, angle_deg: 45.0, phase: 0.0 },
          mask_source: null,
          orientation_source: "orientation_field",
          coordinates: { space: "image_px" }
        },
        threshold: null,
        blend_mode: "normal",
        opacity: 1.0,
        priority: 0
      }
    ]
  };
}

function isInitialWaveComposition(composition = state.composition) {
  if (!composition || composition.layers.length !== 1) return false;
  const layer = composition.layers[0];
  return layer.name === "ink" &&
    layer.density_source === "tone_map" &&
    layer.pattern &&
    layer.pattern.kind === "wave" &&
    layer.pattern.mask_source === "edge_mask";
}

function ensureCompositionForMode(mode) {
  if (!modeUsesComposition(mode)) return;
  if (!state.composition || !state.composition.layers || state.composition.layers.length === 0) {
    state.composition = mode === "structure_analyzer" ? defaultStructureComposition() : defaultWaveComposition();
  } else if (mode === "structure_analyzer" && isInitialWaveComposition()) {
    state.composition = defaultStructureComposition();
  }
  if (els.paperColorPicker && state.composition.paper_color) {
    els.paperColorPicker.value = state.composition.paper_color.hex;
  }
}

// Generate dynamic parameter controls from schema
function renderControlsForMode(mode) {
  ensureCompositionForMode(mode);
  els.dynamicControls.innerHTML = "";

  if (mode === "ordered_bayer") {
    // Generate Bayer controls statically to preserve exactly how Phase 0 worked
    els.dynamicControls.innerHTML = `
      <label class="control">
        <span>Bayer matrix</span>
        <select id="matrixSize" class="select">
          <option value="2">2 x 2</option>
          <option value="4">4 x 4</option>
          <option value="8" selected>8 x 8</option>
          <option value="16">16 x 16</option>
        </select>
      </label>

      <label class="control">
        <span>Threshold</span>
        <output id="thresholdValue">0.00</output>
        <input id="threshold" type="range" min="-0.5" max="0.5" step="0.01" value="0">
      </label>

      <label class="control">
        <span>Contrast</span>
        <output id="contrastValue">1.00</output>
        <input id="contrast" type="range" min="0.1" max="3" step="0.05" value="1">
      </label>
    `;

    // Bind events
    const controls = [
      document.querySelector("#matrixSize"),
      document.querySelector("#threshold"),
      document.querySelector("#contrast"),
    ];
    controls.forEach((ctrl) => {
      ctrl.addEventListener("input", scheduleRender);
      ctrl.addEventListener("change", scheduleRender);
    });
  } else if (state.schemas) {
    const algo = state.schemas.algorithms.find((a) => a.id === mode);
    if (algo) {
      const heading = document.createElement("h3");
      heading.textContent = `${algo.name} Params`;
      heading.style.margin = "8px 0 4px 0";
      heading.style.fontSize = "13px";
      els.dynamicControls.appendChild(heading);

      algo.parameters.forEach((param) => {
        const row = document.createElement("div");
        row.id = `row-${param.key}`;
        row.className = "control";

        if (param.type === "bool") {
          row.style.gridTemplateColumns = "1fr auto";
          const span = document.createElement("span");
          span.textContent = param.label;

          const checkbox = document.createElement("input");
          checkbox.id = `param-${param.key}`;
          checkbox.type = "checkbox";
          checkbox.checked = !!param.default;

          row.appendChild(span);
          row.appendChild(checkbox);

          checkbox.addEventListener("change", (e) => {
            updateVisibility();
            scheduleRender();
          });
        } else if (param.type === "str" && param.ui_hint === "color") {
          // Color input
          row.style.gridTemplateColumns = "1fr auto";
          const span = document.createElement("span");
          span.textContent = param.label;

          const colorInput = document.createElement("input");
          colorInput.id = `param-${param.key}`;
          colorInput.type = "color";
          colorInput.value = param.default;
          colorInput.style.width = "40px";
          colorInput.style.height = "24px";
          colorInput.style.border = "none";
          colorInput.style.cursor = "pointer";
          colorInput.style.padding = "0";

          row.appendChild(span);
          row.appendChild(colorInput);

          colorInput.addEventListener("input", scheduleRender);
          colorInput.addEventListener("change", scheduleRender);
        } else {
          // Float or Int
          row.style.gridTemplateColumns = "1fr auto";
          const span = document.createElement("span");
          span.textContent = param.label;

          const output = document.createElement("output");
          output.textContent = Number(param.default).toFixed(2);

          const slider = document.createElement("input");
          slider.id = `param-${param.key}`;
          slider.type = "range";
          slider.min = String(param.min !== undefined ? param.min : 0);
          slider.max = String(param.max !== undefined ? param.max : 1);
          slider.step = String(param.step !== undefined ? param.step : 0.01);
          slider.value = String(param.default);

          row.appendChild(span);
          row.appendChild(output);
          row.appendChild(slider);

          slider.addEventListener("input", (e) => {
            output.textContent = Number(e.target.value).toFixed(2);
            scheduleRender();
          });
          slider.addEventListener("change", scheduleRender);
        }

        els.dynamicControls.appendChild(row);
      });
    }
    updateVisibility();
  }

  const layersPanel = document.querySelector("#layersPanel");
  if (layersPanel) {
    layersPanel.style.display = modeUsesComposition(mode) ? "block" : "none";
  }
  renderLayersUI();
  refreshChrome();
}

function updateVisibility() {
  const mode = getActivePipeline();
  if (!state.schemas) return;

  const algo = state.schemas.algorithms.find((a) => a.id === mode);
  if (!algo) return;

  algo.parameters.forEach((param) => {
    const row = document.querySelector(`#row-${param.key}`);
    if (row && param.visible_when) {
      const condition = param.visible_when;
      const dependInput = document.querySelector(`#param-${condition.param_key}`);
      if (dependInput) {
        const dependValue = dependInput.type === "checkbox" ? dependInput.checked : dependInput.value;
        const isVisible = String(dependValue) === String(condition.value);
        row.style.display = isVisible ? "grid" : "none";
      }
    }
  });
}

// Render dynamic layers UI
function renderLayersUI() {
  const container = document.querySelector("#layersList");
  if (!container) return;
  container.innerHTML = "";
  if (!state.composition || !Array.isArray(state.composition.layers)) return;

  state.composition.layers.forEach((layer, index) => {
    const item = document.createElement("div");
    item.className = "layer-item";
    item.dataset.index = index;

    // Header with reordering and removal buttons
    const header = document.createElement("div");
    header.className = "layer-header";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = layer.name || `Layer ${index + 1}`;

    const actions = document.createElement("div");
    actions.className = "layer-actions";

    const upBtn = document.createElement("button");
    upBtn.type = "button";
    upBtn.textContent = "▲";
    upBtn.disabled = index === 0;
    upBtn.addEventListener("click", () => moveLayer(index, -1));

    const downBtn = document.createElement("button");
    downBtn.type = "button";
    downBtn.textContent = "▼";
    downBtn.disabled = index === state.composition.layers.length - 1;
    downBtn.addEventListener("click", () => moveLayer(index, 1));

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-remove-layer";
    removeBtn.textContent = "✖";
    removeBtn.addEventListener("click", () => removeLayer(index));

    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    actions.appendChild(removeBtn);
    header.appendChild(nameSpan);
    header.appendChild(actions);
    item.appendChild(header);

    const patDef = patternDefinition(layer.pattern.kind);

    // Color Input
    const colorLabel = document.createElement("label");
    colorLabel.className = "control";
    colorLabel.innerHTML = `<span>Ink Color</span>`;
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = layer.color.hex;
    colorInput.style.width = "40px";
    colorInput.style.height = "24px";
    colorInput.style.border = "none";
    colorInput.style.cursor = "pointer";
    colorInput.style.padding = "0";
    colorInput.addEventListener("input", (e) => {
      layer.color.hex = e.target.value;
      scheduleRender();
    });
    colorLabel.appendChild(colorInput);
    item.appendChild(colorLabel);

    // Blend Mode Select
    const blendLabel = document.createElement("label");
    blendLabel.className = "control";
    blendLabel.innerHTML = `<span>Blend Mode</span>`;
    const blendSelect = document.createElement("select");
    blendSelect.className = "select";
    ["normal", "multiply"].forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m.charAt(0).toUpperCase() + m.slice(1);
      if (layer.blend_mode === m) opt.selected = true;
      blendSelect.appendChild(opt);
    });
    blendSelect.addEventListener("change", (e) => {
      layer.blend_mode = e.target.value;
      scheduleRender();
    });
    blendLabel.appendChild(blendSelect);
    item.appendChild(blendLabel);

    // Density Source Select
    const densityLabel = document.createElement("label");
    densityLabel.className = "control";
    densityLabel.innerHTML = `<span>Density Source</span>`;
    const densitySelect = document.createElement("select");
    densitySelect.className = "select";
    artifactSourceOptions("density_source").forEach((source) => {
      const opt = document.createElement("option");
      opt.value = source.value;
      opt.textContent = source.label;
      densitySelect.appendChild(opt);
    });
    densitySelect.value = layer.density_source || "tone_map";
    densitySelect.addEventListener("change", (e) => {
      layer.density_source = e.target.value;
      scheduleRender();
    });
    densityLabel.appendChild(densitySelect);
    item.appendChild(densityLabel);

    // Pattern Kind Select
    const patternLabel = document.createElement("label");
    patternLabel.className = "control";
    patternLabel.innerHTML = `<span>Pattern Kind</span>`;
    const patternSelect = document.createElement("select");
    patternSelect.className = "select";

    if (state.schemas && state.schemas.patterns) {
      state.schemas.patterns.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.kind;
        opt.textContent = p.name;
        if (layer.pattern.kind === p.kind) opt.selected = true;
        patternSelect.appendChild(opt);
      });
    }

    patternSelect.addEventListener("change", (e) => {
      layer.pattern.kind = e.target.value;
      // Populate defaults for new pattern
      const patDef = state.schemas.patterns.find((p) => p.kind === e.target.value);
      layer.pattern.params = {};
      if (patDef) {
        patDef.parameters.forEach((p) => {
          layer.pattern.params[p.key] = p.default;
        });
      }
      if (!patDef || !patDef.accepts_orientation) {
        layer.pattern.orientation_source = null;
      } else if (!layer.pattern.orientation_source && getActivePipeline() === "structure_analyzer") {
        layer.pattern.orientation_source = "orientation_field";
      }
      renderLayersUI();
      scheduleRender();
    });
    patternLabel.appendChild(patternSelect);
    item.appendChild(patternLabel);

    if (patDef && patDef.accepts_orientation) {
      const orientationLabel = document.createElement("label");
      orientationLabel.className = "control";
      orientationLabel.innerHTML = `<span>Orientation Source</span>`;
      const orientationSelect = document.createElement("select");
      orientationSelect.className = "select";

      const noneOpt = document.createElement("option");
      noneOpt.value = "";
      noneOpt.textContent = "None";
      orientationSelect.appendChild(noneOpt);

      const orientationOptions = artifactSourceOptions("orientation_source");
      orientationOptions.forEach((source) => {
        const opt = document.createElement("option");
        opt.value = source.value;
        opt.textContent = source.label;
        orientationSelect.appendChild(opt);
      });

      if (
        layer.pattern.orientation_source &&
        !orientationOptions.some((source) => source.value === layer.pattern.orientation_source)
      ) {
        const opt = document.createElement("option");
        opt.value = layer.pattern.orientation_source;
        opt.textContent = layer.pattern.orientation_source;
        orientationSelect.appendChild(opt);
      }

      orientationSelect.value = layer.pattern.orientation_source || "";
      orientationSelect.addEventListener("change", (e) => {
        layer.pattern.orientation_source = e.target.value || null;
        scheduleRender();
      });
      orientationLabel.appendChild(orientationSelect);
      item.appendChild(orientationLabel);
    }

    // Pattern parameters sub-sliders
    const patternControls = document.createElement("div");
    patternControls.className = "layer-pattern-controls";

    if (patDef) {
      patDef.parameters.forEach((param) => {
        const row = document.createElement("div");
        row.className = "control";
        row.style.gridTemplateColumns = "1fr auto";

        const span = document.createElement("span");
        span.textContent = param.label;

        const val = layer.pattern.params[param.key] !== undefined ? layer.pattern.params[param.key] : param.default;
        const output = document.createElement("output");

        if (param.type === "int") {
          output.textContent = parseInt(val);
        } else {
          output.textContent = Number(val).toFixed(2);
        }

        let inputEl;
        if (param.options) {
          inputEl = document.createElement("select");
          inputEl.className = "select";
          param.options.forEach((opt) => {
            const o = document.createElement("option");
            o.value = opt.value;
            o.textContent = opt.label;
            if (val === opt.value) o.selected = true;
            inputEl.appendChild(o);
          });
          inputEl.addEventListener("change", (e) => {
            layer.pattern.params[param.key] = Number(e.target.value);
            scheduleRender();
          });
        } else {
          inputEl = document.createElement("input");
          inputEl.type = "range";
          inputEl.min = String(param.min !== undefined ? param.min : 0);
          inputEl.max = String(param.max !== undefined ? param.max : 100);
          inputEl.step = String(param.step !== undefined ? param.step : 1);
          inputEl.value = String(val);
          inputEl.addEventListener("input", (e) => {
            if (param.type === "int") {
              output.textContent = parseInt(e.target.value);
              layer.pattern.params[param.key] = parseInt(e.target.value);
            } else {
              output.textContent = Number(e.target.value).toFixed(2);
              layer.pattern.params[param.key] = Number(e.target.value);
            }
            scheduleRender();
          });
          inputEl.addEventListener("change", scheduleRender);
        }

        row.appendChild(span);
        row.appendChild(output);
        row.appendChild(inputEl);
        patternControls.appendChild(row);
      });
    }

    item.appendChild(patternControls);
    container.appendChild(item);
  });
}

function moveLayer(index, direction) {
  const targetIndex = index + direction;
  if (targetIndex < 0 || targetIndex >= state.composition.layers.length) return;

  const temp = state.composition.layers[index];
  state.composition.layers[index] = state.composition.layers[targetIndex];
  state.composition.layers[targetIndex] = temp;

  state.composition.layers.forEach((l, idx) => {
    l.priority = idx;
  });

  renderLayersUI();
  scheduleRender();
}

function removeLayer(index) {
  state.composition.layers.splice(index, 1);
  state.composition.layers.forEach((l, idx) => {
    l.priority = idx;
  });
  renderLayersUI();
  scheduleRender();
}

function addLayer() {
  const structureMode = getActivePipeline() === "structure_analyzer";
  const newLayer = {
    name: `Layer ${state.composition.layers.length + 1}`,
    color: { hex: "#1a1a1a" },
    role: "shadow",
    density_source: "tone_map",
    pattern: {
      kind: structureMode ? "hatch" : "wave",
      params: { frequency: 8.0, angle_deg: 45.0, phase: 0.0 },
      mask_source: structureMode ? null : "edge_mask",
      orientation_source: structureMode ? "orientation_field" : null,
      coordinates: { space: "image_px" }
    },
    threshold: null,
    blend_mode: "normal",
    opacity: 1.0,
    priority: state.composition.layers.length
  };
  state.composition.layers.push(newLayer);
  renderLayersUI();
  scheduleRender();
}

// Preset CRUD logic
async function loadPresetsList() {
  const res = await fetch("/api/presets");
  const data = await res.json();
  state.presets = data.presets;

  if (els.presetSelect) {
    els.presetSelect.innerHTML = "";
    state.presets.forEach((preset) => {
      const opt = document.createElement("option");
      opt.value = preset.id;
      opt.textContent = preset.name + (preset.is_builtin ? " (Built-in)" : "");
      els.presetSelect.appendChild(opt);
    });
  }
}

async function applyPreset() {
  if (!els.presetSelect || !els.presetSelect.value) return;
  const preset = state.presets.find((p) => p.id === els.presetSelect.value);
  if (!preset) return;

  // Load pipeline
  els.pipelineSelect.value = preset.renderer_id;
  renderControlsForMode(preset.renderer_id);

  // Set parameters
  if (state.schemas) {
    const algo = state.schemas.algorithms.find((a) => a.id === preset.renderer_id);
    if (algo) {
      algo.parameters.forEach((param) => {
        const el = document.querySelector(`#param-${param.key}`);
        if (el) {
          if (param.type === "bool") {
            el.checked = !!preset.params[param.key];
          } else if (param.type === "str") {
            el.value = preset.params[param.key];
          } else {
            el.value = preset.params[param.key];
          }
        }
      });
      updateVisibility();
    }
  }

  // Load composition if present
  if (modeUsesComposition(preset.renderer_id) && preset.composition && Array.isArray(preset.composition.layers)) {
    state.composition = JSON.parse(JSON.stringify(preset.composition));
    if (els.paperColorPicker && state.composition.paper_color) {
      els.paperColorPicker.value = state.composition.paper_color.hex;
    }
    renderLayersUI();
  }

  scheduleRender();
  setStatus(`Applied preset: ${preset.name}`);
}

async function savePreset() {
  if (!els.newPresetName || !els.newPresetName.value.trim()) {
    setStatus("Enter a preset name first");
    return;
  }

  const mode = getActivePipeline();
  const payload = {
    name: els.newPresetName.value.trim(),
    renderer_id: mode,
    params: getParams(),
    composition: modeUsesComposition(mode) ? getComposition() : null
  };

  const response = await fetch("/api/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const result = await response.json();
  if (!response.ok) {
    setStatus(result.error || "Save preset failed");
    return;
  }

  els.newPresetName.value = "";
  await loadPresetsList();
  if (els.presetSelect) els.presetSelect.value = result.id;
  setStatus("Preset saved");
}

async function deletePreset() {
  if (!els.presetSelect || !els.presetSelect.value) return;

  const presetId = els.presetSelect.value;
  const preset = state.presets.find((p) => p.id === presetId);
  if (preset && preset.is_builtin) {
    setStatus("Cannot delete a built-in preset");
    return;
  }

  const response = await fetch(`/api/presets/${presetId}`, {
    method: "DELETE"
  });

  if (!response.ok) {
    const err = await response.json();
    setStatus(err.error || "Delete failed");
    return;
  }

  await loadPresetsList();
  setStatus("Preset deleted");
}

async function exportSvg() {
  if (!state.asset) {
    setStatus("Load a raster first");
    return;
  }
  const mode = getActivePipeline();
  if (!modeUsesComposition(mode)) {
    setStatus("SVG export needs a composited pipeline");
    return;
  }

  const composition = getComposition();
  if (!compositionHasStrokeLayer(composition)) {
    setStatus("SVG export needs a hatch or crosshatch layer");
    return;
  }

  const payload = {
    asset_id: state.asset.id,
    renderer_id: mode,
    params: getParams(),
    composition,
  };

  const response = await fetch("/api/export/svg", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    let message = "SVG export failed";
    try {
      const err = await response.json();
      message = err.error || message;
    } catch (e) {
      // Keep the generic message if the server did not return JSON.
    }
    setStatus(message);
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `colorworks-${state.asset.id.slice(0, 8)}.svg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus("SVG exported");
}

// Gallery state and functions
const galleryState = {
  manifest: [],
  selectedRuns: {} // fixture_name -> selected run object
};

function enterGallery() {
  document.querySelector(".rail-left").style.display = "none";
  document.querySelector(".workspace").style.display = "none";
  document.querySelector(".rail-right").style.display = "none";
  const gall = document.getElementById("galleryView");
  gall.style.display = "flex";
  document.querySelector(".app-shell").style.gridTemplateColumns = "1fr";
  loadGallery();
}

function exitGallery() {
  document.querySelector(".rail-left").style.display = "flex";
  document.querySelector(".workspace").style.display = "grid";
  document.querySelector(".rail-right").style.display = "flex";
  const gall = document.getElementById("galleryView");
  gall.style.display = "none";
  document.querySelector(".app-shell").style.gridTemplateColumns = "minmax(220px, 280px) minmax(0, 1fr) minmax(230px, 300px)";
}

async function loadGallery() {
  const content = document.getElementById("galleryContent");
  content.innerHTML = '<div style="padding: 40px; text-align: center; font-style: italic; color: var(--muted);">Loading manifest and outputs...</div>';

  try {
    const res = await fetch("/api/comparison/manifest");
    if (!res.ok) {
      throw new Error(`Failed to load manifest: ${res.statusText}`);
    }
    const manifest = await res.json();
    galleryState.manifest = manifest;

    manifest.forEach(run => {
      const fix = run.fixture_name;
      if (!galleryState.selectedRuns[fix]) {
        galleryState.selectedRuns[fix] = run;
      } else {
        const exists = manifest.some(r => r.fixture_name === fix && r.run_id === galleryState.selectedRuns[fix].run_id);
        if (!exists) {
          galleryState.selectedRuns[fix] = run;
        }
      }
    });

    populateFixtureFilter(manifest);
    renderGallery();
  } catch (err) {
    content.innerHTML = `<div style="padding: 40px; text-align: center; color: #e05638; font-weight: 500;">
      Error loading gallery: ${err.message}<br>
      <span style="font-size: 12px; color: var(--muted); font-weight: normal; margin-top: 8px; display: inline-block;">
        Ensure comparison harness has run: <code>python -m colorworks.algorithms.comparison_harness</code>
      </span>
    </div>`;
  }
}

function populateFixtureFilter(manifest) {
  const filter = document.getElementById("galleryFixtureFilter");
  if (!filter) return;
  const currentVal = filter.value;
  const fixtures = [...new Set(manifest.map(r => r.fixture_name))];

  filter.innerHTML = '<option value="all">All Fixtures</option>';
  fixtures.forEach(fix => {
    const opt = document.createElement("option");
    opt.value = fix;
    opt.textContent = fix;
    filter.appendChild(opt);
  });

  if (fixtures.includes(currentVal)) {
    filter.value = currentVal;
  } else {
    filter.value = "all";
  }
}

function renderGallery() {
  const content = document.getElementById("galleryContent");
  if (!content) return;
  const fixtureFilter = document.getElementById("galleryFixtureFilter").value;
  const kindFilter = document.getElementById("galleryKindFilter").value;

  let filtered = galleryState.manifest;
  if (fixtureFilter !== "all") {
    filtered = filtered.filter(r => r.fixture_name === fixtureFilter);
  }

  const grouped = {};
  filtered.forEach(run => {
    const fix = run.fixture_name;
    if (!grouped[fix]) grouped[fix] = [];
    grouped[fix].push(run);
  });

  content.innerHTML = "";
  const fixturesToRender = Object.keys(grouped);
  if (fixturesToRender.length === 0) {
    content.innerHTML = '<div style="padding: 40px; text-align: center; font-style: italic; color: var(--muted);">No matching comparison outputs found.</div>';
    return;
  }

  fixturesToRender.forEach(fixtureName => {
    const runs = grouped[fixtureName];
    let displayRuns = runs;
    if (kindFilter !== "all") {
      displayRuns = runs.filter(r => r.kind === kindFilter);
    }

    let selectedRun = galleryState.selectedRuns[fixtureName];
    if (displayRuns.length > 0 && !displayRuns.some(r => r.run_id === selectedRun.run_id)) {
      selectedRun = displayRuns[0];
    }

    const fixtureCard = document.createElement("div");
    fixtureCard.className = "fixture-comparison-card";
    fixtureCard.style.cssText = "border: 1px solid var(--line); border-radius: 8px; background: #fffdf7; overflow: hidden; display: flex; flex-direction: column;";

    const header = document.createElement("div");
    header.style.cssText = "background: var(--line); padding: 10px 16px; display: flex; justify-content: space-between; align-items: center;";
    header.innerHTML = `
      <h3 style="margin: 0; font-size: 15px; font-weight: 700; text-transform: capitalize; color: var(--ink);">Fixture: ${fixtureName}</h3>
      <span style="font-size: 11px; color: var(--muted); font-family: monospace;">Source Checksum: ${runs[0].fixture_checksum.slice(0, 12)}...</span>
    `;
    fixtureCard.appendChild(header);

    const body = document.createElement("div");
    body.style.cssText = "display: grid; grid-template-columns: 320px 1fr; min-height: 380px;";

    const comparePane = document.createElement("div");
    comparePane.style.cssText = "border-right: 1px solid var(--line); padding: 16px; display: flex; flex-direction: column; gap: 12px; background: #fffdf7;";

    const imagesContainer = document.createElement("div");
    imagesContainer.style.cssText = "display: flex; gap: 8px; height: 150px;";

    const sourceBox = document.createElement("div");
    sourceBox.style.cssText = "flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid var(--line); border-radius: 4px; padding: 4px; background: var(--panel); position: relative;";
    sourceBox.innerHTML = `
      <span style="position: absolute; top: 4px; left: 4px; font-size: 8px; background: var(--muted); color: var(--paper); padding: 1px 4px; border-radius: 2px; font-weight: 600;">SOURCE</span>
      <img src="${runs[0].source_url}" style="max-height: 110px; max-width: 100%; object-fit: contain; image-rendering: pixelated; border: 1px solid rgba(0,0,0,0.05);">
      <span style="font-size: 10px; margin-top: 4px; font-weight: 600; color: var(--muted);">${runs[0].width}x${runs[0].height}</span>
    `;
    imagesContainer.appendChild(sourceBox);

    const outputBox = document.createElement("div");
    outputBox.style.cssText = "flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid var(--line); border-radius: 4px; padding: 4px; background: var(--panel); position: relative;";

    const comparedImg = document.createElement("img");
    comparedImg.style.cssText = "max-height: 110px; max-width: 100%; object-fit: contain; image-rendering: pixelated; border: 1px solid rgba(0,0,0,0.05);";
    if (selectedRun) {
      comparedImg.src = selectedRun.output_url;
    }

    const comparedRes = document.createElement("span");
    comparedRes.style.cssText = "font-size: 10px; margin-top: 4px; font-weight: 600; color: var(--muted);";
    if (selectedRun) {
      comparedRes.textContent = `${selectedRun.width}x${selectedRun.height}`;
    }

    outputBox.innerHTML = `<span style="position: absolute; top: 4px; left: 4px; font-size: 8px; background: var(--accent); color: var(--paper); padding: 1px 4px; border-radius: 2px; font-weight: 600;">COMPARED</span>`;
    outputBox.appendChild(comparedImg);
    outputBox.appendChild(comparedRes);
    imagesContainer.appendChild(outputBox);
    comparePane.appendChild(imagesContainer);

    const details = document.createElement("div");
    details.style.cssText = "flex: 1; border-top: 1px solid var(--line); padding-top: 12px; font-size: 12px; display: flex; flex-direction: column; gap: 8px;";

    function populateDetails(run) {
      if (!run) {
        details.innerHTML = '<div style="font-style: italic; color: var(--muted);">No output selected.</div>';
        return;
      }
      const isPreset = run.kind === "preset";
      const kindLabel = isPreset ? "Preset" : "Algorithm";
      const kindColor = isPreset ? "#0f6f78" : "#121212";

      details.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
          <span style="font-weight: 700; font-size: 13px; color: var(--ink);">${run.preset_id || run.algorithm_id}</span>
          <span style="font-size: 10px; font-weight: 700; color: white; background: ${kindColor}; padding: 1px 6px; border-radius: 10px; text-transform: uppercase;">${kindLabel}</span>
        </div>
        <div style="font-size: 11px; color: var(--muted); margin-bottom: 4px;">
          Renderer: <code>${run.algorithm_id}</code>
        </div>
        <dl style="margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; line-height: 1.4;">
          <dt style="color: var(--muted);">Runtime:</dt>
          <dd style="margin: 0; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums;">${run.runtime_ms.toFixed(2)} ms</dd>

          <dt style="color: var(--muted);">MSE:</dt>
          <dd style="margin: 0; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; font-family: monospace;">${run.metrics.mse.toFixed(5)}</dd>

          <dt style="color: var(--muted);">Mean Intensity:</dt>
          <dd style="margin: 0; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; font-family: monospace;">${run.metrics.mean_intensity.toFixed(4)}</dd>

          <dt style="color: var(--muted); align-self: center;">Checksum:</dt>
          <dd style="margin: 0; font-family: monospace; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 140px; text-align: right;" title="${run.checksum}">${run.checksum.slice(0, 16)}...</dd>
        </dl>
      `;
    }

    populateDetails(selectedRun);
    comparePane.appendChild(details);
    body.appendChild(comparePane);

    const gridPane = document.createElement("div");
    gridPane.style.cssText = "padding: 16px; overflow-y: auto; background: #faf8f1;";

    const grid = document.createElement("div");
    grid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px;";

    displayRuns.forEach(run => {
      const isSelected = selectedRun && run.run_id === selectedRun.run_id;
      const thumb = document.createElement("div");
      thumb.style.cssText = `
        border: 1px solid ${isSelected ? "var(--accent)" : "var(--line)"};
        border-radius: 6px;
        padding: 8px;
        background: #fffdf7;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        box-shadow: ${isSelected ? "0 0 0 1px var(--accent)" : "none"};
        transition: all 0.15s ease;
      `;

      thumb.innerHTML = `
        <img src="${run.output_url}" style="height: 60px; max-width: 100%; object-fit: contain; image-rendering: pixelated; border: 1px solid rgba(0,0,0,0.05);">
        <span style="font-size: 11px; font-weight: 600; text-align: center; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; width: 100%; color: var(--ink);" title="${run.preset_id || run.algorithm_id}">
          ${run.preset_id || run.algorithm_id}
        </span>
        <span style="font-size: 9px; color: var(--muted); font-variant-numeric: tabular-nums;">
          MSE: ${run.metrics.mse.toFixed(4)}
        </span>
      `;

      thumb.addEventListener("click", () => {
        galleryState.selectedRuns[fixtureName] = run;
        comparedImg.src = run.output_url;
        comparedRes.textContent = `${run.width}x${run.height}`;
        populateDetails(run);

        Array.from(grid.children).forEach(child => {
          child.style.borderColor = "var(--line)";
          child.style.boxShadow = "none";
        });
        thumb.style.borderColor = "var(--accent)";
        thumb.style.boxShadow = "0 0 0 1px var(--accent)";
      });

      grid.appendChild(thumb);
    });

    gridPane.appendChild(grid);
    body.appendChild(gridPane);
    fixtureCard.appendChild(body);
    content.appendChild(fixtureCard);
  });
}

// Fetch schemas and setup initial application state
async function init() {
  try {
    const res = await fetch("/api/schemas");
    state.schemas = await res.json();
  } catch (err) {
    console.error("Failed to load schemas: ", err);
  }

  // Setup tab button click handlers
  els.tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeTab = btn.getAttribute("data-tab");
      updateViewer();
    });
  });

  // Handle pipeline mode change
  els.pipelineSelect.addEventListener("change", (e) => {
    state.tone_map = null;
    state.edge_mask = null;
    state.structure_tensor = null;
    state.orientation_field = null;
    renderControlsForMode(e.target.value);
    scheduleRender();
  });

  if (els.vectorViewSelect) {
    els.vectorViewSelect.addEventListener("change", (e) => {
      state.vectorView = e.target.value;
      updateViewer();
    });
  }

  els.assetInput.addEventListener("change", (event) => {
    uploadAsset(event.target.files[0]);
  });

  els.saveRecipe.addEventListener("click", saveRecipe);
  els.reloadRecipe.addEventListener("click", reloadRecipe);

  // Setup Presets and Layers UI event listeners
  if (els.loadPreset) els.loadPreset.addEventListener("click", applyPreset);
  if (els.deletePreset) els.deletePreset.addEventListener("click", deletePreset);
  if (els.savePreset) els.savePreset.addEventListener("click", savePreset);
  if (els.addLayerBtn) els.addLayerBtn.addEventListener("click", addLayer);
  if (els.exportSvgBtn) els.exportSvgBtn.addEventListener("click", exportSvg);
  if (els.paperColorPicker) {
    els.paperColorPicker.addEventListener("change", (e) => {
      state.composition.paper_color.hex = e.target.value;
      scheduleRender();
    });
  }

  // Gallery Navigation listeners
  const enterGalleryBtn = document.querySelector("#enterGalleryBtn");
  const exitGalleryBtn = document.querySelector("#exitGalleryBtn");
  const refreshGalleryBtn = document.querySelector("#refreshGalleryBtn");
  const fixtureFilter = document.querySelector("#galleryFixtureFilter");
  const kindFilter = document.querySelector("#galleryKindFilter");

  if (enterGalleryBtn) enterGalleryBtn.addEventListener("click", enterGallery);
  if (exitGalleryBtn) exitGalleryBtn.addEventListener("click", exitGallery);
  if (refreshGalleryBtn) refreshGalleryBtn.addEventListener("click", loadGallery);
  if (fixtureFilter) fixtureFilter.addEventListener("change", renderGallery);
  if (kindFilter) kindFilter.addEventListener("change", renderGallery);

  // Default composition
  state.composition = defaultWaveComposition();

  // Initialize controls
  renderControlsForMode(getActivePipeline());
  updateControlLabels();
  updateVisibility();
  renderLayersUI();
  await loadPresetsList();
  await loadRecipeList();
  updateViewer();
}

init();
