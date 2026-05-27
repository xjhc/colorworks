const state = {
  asset: null,
  output: null,
  tone_map: null,
  edge_mask: null,
  renderTimer: null,
  sourceObjectUrl: null,
  activeTab: "source", // "source" | "tone_map" | "edge_mask" | "final"
  schemas: null,
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
};

function getActivePipeline() {
  return els.pipelineSelect.value;
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
          } else {
            params[param.key] = Number(el.value);
          }
        }
      });
    }
  }
  return params;
}

// Read composition (for tonal_analyzer)
function getComposition() {
  if (!state.schemas) return null;

  // Find pattern params dynamically
  const wave = state.schemas.patterns.find((p) => p.kind === "wave");
  const waveParams = {};
  if (wave) {
    wave.parameters.forEach((param) => {
      const el = document.querySelector(`#pat-${param.key}`);
      if (el) {
        waveParams[param.key] = Number(el.value);
      }
    });
  }

  const preserveEl = document.querySelector("#param-preserve_edges");
  const preserve_edges = preserveEl ? preserveEl.checked : true;

  return {
    paper_color: { hex: "#f4ebd9", name: "paper" },
    layers: [
      {
        name: "ink",
        color: { hex: "#1a1a1a", name: "ink" },
        role: "shadow",
        density_source: "tone_map",
        pattern: {
          kind: "wave",
          params: waveParams,
          mask_source: preserve_edges ? "edge_mask" : null,
          coordinates: {
            space: "image_px",
          },
        },
      },
    ],
  };
}

function updateControlLabels() {
  // Updates any visible output indicators
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
  };
  if (mode === "tonal_analyzer") {
    payload.composition = getComposition();
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
    state.tone_map = result.artifacts.tone_map;
    state.edge_mask = result.artifacts.edge_mask;
  } else {
    state.tone_map = null;
    state.edge_mask = null;
  }

  updateViewer();

  els.renderTime.textContent = `${state.output.render_ms} ms`;
  els.outputChecksum.textContent = state.output.checksum.slice(0, 16);
  els.exportLink.href = state.output.url;
  els.exportLink.download = `colorworks-${state.output.checksum.slice(0, 12)}.png`;
  els.exportLink.classList.remove("disabled");
  setStatus(`${state.output.width} x ${state.output.height}`);
}

function updateViewer() {
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
  if (mode === "tonal_analyzer") {
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
  // Populate loaded values
  if (recipe.renderer_id !== "ordered_bayer" && state.schemas) {
    const algo = state.schemas.algorithms.find((a) => a.id === recipe.renderer_id);
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
      // Update visibility after toggling checkboxes
      updateVisibility();
    }

    // Load wave pattern parameters
    if (recipe.composition && recipe.composition.layers && recipe.composition.layers.length > 0) {
      const inkLayer = recipe.composition.layers[0];
      if (inkLayer.pattern && inkLayer.pattern.params) {
        const patternKind = inkLayer.pattern.kind;
        const patDef = state.schemas.patterns.find((p) => p.kind === patternKind);
        if (patDef) {
          patDef.parameters.forEach((param) => {
            const el = document.querySelector(`#pat-${param.key}`);
            if (el) {
              el.value = inkLayer.pattern.params[param.key];
            }
          });
        }
      }
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

// Generate dynamic parameter controls from schema
function renderControlsForMode(mode) {
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
  } else if (mode === "tonal_analyzer" && state.schemas) {
    // Retrieve tonal_analyzer and wave pattern schemas
    const algo = state.schemas.algorithms.find((a) => a.id === "tonal_analyzer");
    const wave = state.schemas.patterns.find((p) => p.kind === "wave");

    if (algo) {
      const heading = document.createElement("h3");
      heading.textContent = "Analyzer Params";
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

    if (wave) {
      const heading = document.createElement("h3");
      heading.textContent = "Wave Pattern Params";
      heading.style.margin = "16px 0 4px 0";
      heading.style.fontSize = "13px";
      els.dynamicControls.appendChild(heading);

      wave.parameters.forEach((param) => {
        const row = document.createElement("div");
        row.className = "control";
        row.style.gridTemplateColumns = "1fr auto";

        const span = document.createElement("span");
        span.textContent = param.label;

        const output = document.createElement("output");
        output.textContent = Number(param.default).toFixed(2);

        const slider = document.createElement("input");
        slider.id = `pat-${param.key}`;
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

        els.dynamicControls.appendChild(row);
      });
    }
    updateVisibility();
  }
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
      els.tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeTab = btn.getAttribute("data-tab");
      updateViewer();
    });
  });

  // Handle pipeline mode change
  els.pipelineSelect.addEventListener("change", (e) => {
    renderControlsForMode(e.target.value);
    scheduleRender();
  });

  els.assetInput.addEventListener("change", (event) => {
    uploadAsset(event.target.files[0]);
  });

  els.saveRecipe.addEventListener("click", saveRecipe);
  els.reloadRecipe.addEventListener("click", reloadRecipe);

  // Initialize controls
  renderControlsForMode(getActivePipeline());
  updateControlLabels();
  updateVisibility();
  await loadRecipeList();
  updateViewer();
}

init();
