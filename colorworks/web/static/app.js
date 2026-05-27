const state = {
  asset: null,
  output: null,
  renderTimer: null,
  sourceObjectUrl: null,
};

const els = {
  assetInput: document.querySelector("#assetInput"),
  assetName: document.querySelector("#assetName"),
  assetSize: document.querySelector("#assetSize"),
  assetChecksum: document.querySelector("#assetChecksum"),
  sourcePreview: document.querySelector("#sourcePreview"),
  outputPreview: document.querySelector("#outputPreview"),
  matrixSize: document.querySelector("#matrixSize"),
  threshold: document.querySelector("#threshold"),
  thresholdValue: document.querySelector("#thresholdValue"),
  contrast: document.querySelector("#contrast"),
  contrastValue: document.querySelector("#contrastValue"),
  sourceStatus: document.querySelector("#sourceStatus"),
  renderStatus: document.querySelector("#renderStatus"),
  renderTime: document.querySelector("#renderTime"),
  outputChecksum: document.querySelector("#outputChecksum"),
  exportLink: document.querySelector("#exportLink"),
  recipeName: document.querySelector("#recipeName"),
  recipeSelect: document.querySelector("#recipeSelect"),
  saveRecipe: document.querySelector("#saveRecipe"),
  reloadRecipe: document.querySelector("#reloadRecipe"),
};

function params() {
  return {
    matrix_size: Number(els.matrixSize.value),
    threshold: Number(els.threshold.value),
    contrast: Number(els.contrast.value),
  };
}

function setStatus(message) {
  els.renderStatus.textContent = message;
}

function updateControlLabels() {
  els.thresholdValue.textContent = Number(els.threshold.value).toFixed(2);
  els.contrastValue.textContent = Number(els.contrast.value).toFixed(2);
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
  const response = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      asset_id: state.asset.id,
      params: params(),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "Render failed");
    return;
  }
  state.output = payload.output;
  els.outputPreview.src = `${state.output.url}?v=${state.output.checksum}`;
  els.renderTime.textContent = `${state.output.render_ms} ms`;
  els.outputChecksum.textContent = state.output.checksum.slice(0, 16);
  els.exportLink.href = state.output.url;
  els.exportLink.download = `colorworks-${state.output.checksum.slice(0, 12)}.png`;
  els.exportLink.classList.remove("disabled");
  setStatus(`${state.output.width} x ${state.output.height}`);
}

async function uploadAsset(file) {
  if (!file) {
    return;
  }
  if (state.sourceObjectUrl) {
    URL.revokeObjectURL(state.sourceObjectUrl);
  }
  state.sourceObjectUrl = URL.createObjectURL(file);
  els.sourcePreview.src = state.sourceObjectUrl;
  els.sourceStatus.textContent = "Uploading";
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
    els.sourceStatus.textContent = payload.error || "Upload failed";
    return;
  }
  state.asset = payload.asset;
  els.assetName.textContent = state.asset.original_filename;
  els.assetSize.textContent = `${state.asset.width} x ${state.asset.height}`;
  els.assetChecksum.textContent = state.asset.checksum.slice(0, 16);
  els.sourceStatus.textContent = `${state.asset.width} x ${state.asset.height}`;
  scheduleRender();
}

async function saveRecipe() {
  if (!state.asset) {
    setStatus("Load a raster first");
    return;
  }
  const response = await fetch("/api/recipes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: els.recipeName.value,
      asset_id: state.asset.id,
      params: params(),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.error || "Save failed");
    return;
  }
  await loadRecipeList();
  els.recipeSelect.value = payload.id;
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
  els.matrixSize.value = String(recipe.params.matrix_size);
  els.threshold.value = String(recipe.params.threshold);
  els.contrast.value = String(recipe.params.contrast);
  updateControlLabels();

  if (!state.asset || state.asset.id !== recipe.asset.id) {
    const assetResponse = await fetch(`/api/assets/${recipe.asset.id}`);
    const assetPayload = await assetResponse.json();
    if (!assetResponse.ok) {
      setStatus(assetPayload.error || "Asset missing");
      return;
    }
    state.asset = assetPayload.asset;
    els.sourcePreview.src = `/api/assets/${state.asset.id}/image`;
    els.assetName.textContent = state.asset.original_filename;
    els.assetSize.textContent = `${state.asset.width} x ${state.asset.height}`;
    els.assetChecksum.textContent = state.asset.checksum.slice(0, 16);
    els.sourceStatus.textContent = `${state.asset.width} x ${state.asset.height}`;
  }
  scheduleRender();
}

for (const control of [els.matrixSize, els.threshold, els.contrast]) {
  control.addEventListener("input", scheduleRender);
  control.addEventListener("change", scheduleRender);
}

els.assetInput.addEventListener("change", (event) => {
  uploadAsset(event.target.files[0]);
});
els.saveRecipe.addEventListener("click", saveRecipe);
els.reloadRecipe.addEventListener("click", reloadRecipe);

updateControlLabels();
loadRecipeList();
