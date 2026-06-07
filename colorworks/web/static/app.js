/* ============================================================================
   Colorworks — studio (direct editor)
   Upload → live preview → pick a style → turn the knobs → recolour → export.
   Every style renders synchronously, so the preview updates instantly.
   ========================================================================== */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// The curated looks. `fixed` params are set by the Style selector and hidden
// from the knob panel; everything else in the algorithm's schema is a live knob.
const STYLES = {
  flow:            { label: "Flow (waves)",     algorithm: "tone_dither",      fixed: { method: "flow" } },
  bayer:           { label: "Ordered (Bayer)",  algorithm: "tone_dither",      fixed: { method: "bayer" } },
  blue_noise:      { label: "Blue Noise",       algorithm: "tone_dither",      fixed: { method: "blue_noise" } },
  floyd_steinberg: { label: "Floyd–Steinberg",  algorithm: "tone_dither",      fixed: { method: "floyd_steinberg" } },
  flat:            { label: "Flat poster",      algorithm: "palette_quantize", fixed: { dither: false } },
};

const state = {
  asset: null,
  schemas: null,
  setup: { maxW: 360, maxH: null, fit: "fit" },
  styleId: "flow",
  knobValues: {},        // persisted param values by key (carry across styles)
  renderToken: 0,
  renderStart: 0,
  knobDebounce: null,
  render: null,          // { w, h, idx, palette } of the current preview
  colorMap: {},          // originalHex -> replacementHex
  hoverIdx: null,
  pendingMeta: null,
  zoomMode: "fit",
  zoomPct: 100,
};

// ── helpers ─────────────────────────────────────────────────────────────────
function algoSchema(algoId) {
  return state.schemas?.algorithms?.find((a) => a.id === algoId) || null;
}
function outputSpec() {
  const { maxW, maxH, fit } = state.setup;
  return {
    max_width: Number.isFinite(maxW) && maxW > 0 ? maxW : null,
    max_height: Number.isFinite(maxH) && maxH > 0 ? maxH : null,
    fit,
  };
}
let _toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  requestAnimationFrame(() => t.classList.add("show"));
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { t.classList.remove("show"); setTimeout(() => (t.hidden = true), 250); }, 2600);
}
function showStudio(on) {
  $("#emptyState").hidden = on;
  $("#editorView").hidden = !on;
  $("#inspectorEmpty").hidden = on;
  $("#inspectorBody").hidden = !on;
}

// ── init ──────────────────────────────────────────────────────────────────
async function init() {
  try {
    state.schemas = await (await fetch("/api/schemas")).json();
  } catch (e) {
    toast("Could not load algorithm schemas");
  }
  bindSource();
  bindOutput();
  bindZoom();
  bindPan();
  $("#styleSelect").value = state.styleId;
  $("#styleSelect").addEventListener("change", (e) => applyStyle(e.target.value));
  $("#resetColors").addEventListener("click", () => {
    state.colorMap = {}; state.hoverIdx = null;
    redrawCanvas(); buildSwatches(); updateExport();
  });
  showStudio(false);

  let resizeRAF = null;
  window.addEventListener("resize", () => {
    if (state.zoomMode !== "fit") return;
    cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(applyZoom);
  });
}

// ── source upload ───────────────────────────────────────────────────────────
function bindSource() {
  const dz = $("#dropzone");
  const input = $("#fileInput");
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { if (input.files && input.files[0]) uploadAsset(input.files[0]); });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) uploadAsset(f); });
}

async function uploadAsset(file) {
  $("#srcName").textContent = "Uploading…";

  const resp = await fetch("/api/assets", {
    method: "POST",
    headers: { "Content-Type": file.type || "image/png", "X-Filename": encodeURIComponent(file.name) },
    body: file,
  });
  if (!resp.ok) { $("#srcName").textContent = "Upload failed"; toast("Upload failed"); return; }
  state.asset = (await resp.json()).asset;

  const dz = $("#dropzone");
  dz.classList.add("loaded");
  const thumb = $("#srcThumb");
  thumb.classList.add("has-img");
  thumb.style.backgroundImage = `url(/api/assets/${state.asset.id}/image)`;
  $("#srcName").textContent = state.asset.original_filename || "source";
  $("#srcMeta").textContent = `${state.asset.width} × ${state.asset.height}`;
  $("#topbarMeta").textContent = `${state.asset.original_filename || "source"} · ${state.asset.width}×${state.asset.height}`;

  showStudio(true);
  applyStyle(state.styleId);   // open straight into the studio with a live render
}

// ── output controls ─────────────────────────────────────────────────────────
function bindOutput() {
  $("#maxW").addEventListener("input", (e) => { state.setup.maxW = parseInt(e.target.value, 10); scheduleFocusRender(); });
  $("#maxH").addEventListener("input", (e) => { state.setup.maxH = parseInt(e.target.value, 10); scheduleFocusRender(); });
  $$("#fitGroup button").forEach((b) =>
    b.addEventListener("click", () => {
      $$("#fitGroup button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.setup.fit = b.dataset.fit;
      scheduleFocusRender();
    }));
}

// ── style + knobs ───────────────────────────────────────────────────────────
function applyStyle(styleId) {
  if (!STYLES[styleId]) return;
  state.styleId = styleId;
  $("#styleSelect").value = styleId;
  state.colorMap = {};      // a new look starts from its own palette
  state.hoverIdx = null;
  setZoomFit();
  buildKnobs(STYLES[styleId]);
  renderFocus();
}

function paramValue(p) {
  // persisted value if present and (for selects) still valid, else schema default
  let v = state.knobValues[p.key];
  if (v === undefined) v = p.default;
  if (p.options && p.options.length && !p.options.some((o) => String(o.value) === String(v))) v = p.default;
  return v;
}

function buildKnobs(style) {
  const wrap = $("#knobs");
  wrap.innerHTML = "";
  const algo = algoSchema(style.algorithm);
  if (!algo) return;

  algo.parameters.forEach((p) => {
    if (p.key in style.fixed) return;   // controlled by the Style selector — hidden
    const initial = paramValue(p);
    const row = document.createElement("div");
    row.className = "knob";
    row.id = `knob-${p.key}`;

    const persist = (val) => { state.knobValues[p.key] = val; };

    if (p.type === "bool") {
      row.innerHTML = `<div class="knob-check"><span class="knob-label">${p.label}</span></div>`;
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.id = `param-${p.key}`; cb.checked = !!initial;
      row.querySelector(".knob-check").appendChild(cb);
      cb.addEventListener("change", () => { persist(cb.checked); updateKnobVisibility(); scheduleFocusRender(); });
    } else if ((p.type === "str" && p.ui_hint === "color") ||
               (p.type === "str" && (!p.options || !p.options.length) && /^#[0-9a-fA-F]{3,6}$/.test(String(initial)))) {
      row.innerHTML = `<div class="knob-color"><span class="knob-label">${p.label}</span></div>`;
      const ci = document.createElement("input");
      ci.type = "color"; ci.id = `param-${p.key}`; ci.value = initial;
      row.querySelector(".knob-color").appendChild(ci);
      ci.addEventListener("input", () => { persist(ci.value); scheduleFocusRender(); });
    } else if (p.type === "str" && (!p.options || !p.options.length)) {
      row.innerHTML = `<div class="knob-top"><span class="knob-label">${p.label}</span></div>`;
      const inp = document.createElement("input");
      inp.id = `param-${p.key}`; inp.type = "text"; inp.className = "text-input"; inp.value = initial;
      row.appendChild(inp);
      inp.addEventListener("input", () => { persist(inp.value); scheduleFocusRender(); });
    } else if (p.options && p.options.length) {
      row.innerHTML = `<div class="knob-top"><span class="knob-label">${p.label}</span></div>`;
      const sel = document.createElement("select"); sel.id = `param-${p.key}`;
      p.options.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = String(o.value); opt.textContent = o.label;
        if (String(o.value) === String(initial)) opt.selected = true;
        sel.appendChild(opt);
      });
      row.appendChild(sel);
      sel.addEventListener("change", () => {
        persist(p.type === "int" ? parseInt(sel.value, 10) : sel.value);
        updateKnobVisibility(); scheduleFocusRender();
      });
    } else {
      const fixedDigits = p.type === "int" ? 0 : 2;
      row.innerHTML = `<div class="knob-top"><span class="knob-label">${p.label}</span>
        <output class="knob-val" id="val-${p.key}">${Number(initial).toFixed(fixedDigits)}</output></div>`;
      const sl = document.createElement("input");
      sl.type = "range"; sl.id = `param-${p.key}`;
      sl.min = String(p.min ?? 0); sl.max = String(p.max ?? 1);
      sl.step = String(p.step ?? (p.type === "int" ? 1 : 0.01));
      sl.value = String(initial);
      row.appendChild(sl);
      sl.addEventListener("input", () => {
        $(`#val-${p.key}`).textContent = Number(sl.value).toFixed(fixedDigits);
        persist(p.type === "int" ? parseInt(sl.value, 10) : parseFloat(sl.value));
        scheduleFocusRender();
      });
    }
    wrap.appendChild(row);
  });
  updateKnobVisibility();
}

function updateKnobVisibility() {
  const style = STYLES[state.styleId];
  const algo = algoSchema(style.algorithm);
  if (!algo) return;
  algo.parameters.forEach((p) => {
    const row = $(`#knob-${p.key}`);
    if (!row || !p.visible_when) return;
    const dep = p.visible_when.param_key;
    let val;
    if (dep in style.fixed) val = style.fixed[dep];
    else { const inp = $(`#param-${dep}`); val = inp ? (inp.type === "checkbox" ? inp.checked : inp.value) : null; }
    row.classList.toggle("hidden", String(val) !== String(p.visible_when.value));
  });
}

function knobParams() {
  const style = STYLES[state.styleId];
  const algo = algoSchema(style.algorithm);
  const params = {};
  algo.parameters.forEach((p) => {
    if (p.key in style.fixed) { params[p.key] = style.fixed[p.key]; return; }
    const inp = $(`#param-${p.key}`);
    if (!inp) { params[p.key] = paramValue(p); return; }
    if (p.type === "bool") params[p.key] = inp.checked;
    else if (p.type === "int") params[p.key] = parseInt(inp.value, 10);
    else if (p.type === "float") params[p.key] = parseFloat(inp.value);
    else params[p.key] = inp.value;
  });
  return params;
}

// ── render the preview ──────────────────────────────────────────────────────
function scheduleFocusRender() {
  clearTimeout(state.knobDebounce);
  state.knobDebounce = setTimeout(renderFocus, 110);
}

async function renderFocus() {
  if (!state.asset) return;
  const style = STYLES[state.styleId];
  const params = knobParams();
  const token = ++state.renderToken;
  state.renderStart = performance.now();
  $("#plateLoading").hidden = false;

  const payload = {
    asset_id: state.asset.id,
    renderer_id: style.algorithm,
    params,
    seed: 42,
    output: outputSpec(),
  };
  try {
    const resp = await fetch("/api/render", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await resp.json();
    if (token !== state.renderToken) return;
    if (!resp.ok) { $("#plateLoading").hidden = true; toast(result.error || "Render failed"); return; }
    const o = result.output;
    paintPlate(`${o.url}?v=${o.checksum}`, o.url, o);
  } catch (e) {
    if (token === state.renderToken) { $("#plateLoading").hidden = true; toast("Render error"); }
  }
}

function paintPlate(displayUrl, exportUrl, meta) {
  state.pendingMeta = meta || {};
  loadRender(displayUrl);
}

// Load the rendered PNG into the canvas, extract its palette, wire features.
function loadRender(url) {
  const img = new Image();
  img.onload = () => {
    const cw = img.naturalWidth, ch = img.naturalHeight;
    const base = state._baseCanvas || (state._baseCanvas = document.createElement("canvas"));
    base.width = cw; base.height = ch;
    const bctx = base.getContext("2d", { willReadFrequently: true });
    bctx.imageSmoothingEnabled = false;
    bctx.drawImage(img, 0, 0);
    const data = bctx.getImageData(0, 0, cw, ch);

    const { palette, idx } = extractPaletteAndIndex(data);
    state.render = { w: cw, h: ch, idx, palette };
    state.hoverIdx = null;

    const cv = $("#bigCanvas");
    cv.width = cw; cv.height = ch;
    redrawCanvas();
    buildSwatches();

    const meta = state.pendingMeta || {};
    const elapsed = state.renderStart ? Math.round(performance.now() - state.renderStart) : null;
    $("#bigCaption").innerHTML = `<b>${STYLES[state.styleId].label}</b> · ${cw} × ${ch} px`;
    $("#specDims").textContent = `${cw} × ${ch}`;
    $("#specRender").textContent = elapsed != null ? `${elapsed} ms` : "—";
    $("#specHash").textContent = meta.checksum ? meta.checksum.slice(0, 12) : "—";
    $("#plateLoading").hidden = true;

    applyZoom();
    updateExport();
  };
  img.onerror = () => { $("#plateLoading").hidden = true; toast("Could not load render"); };
  img.src = url;
}

function rgbToHex(r, g, b) { return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join(""); }
function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
}
const _lumOf = (c) => 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;

function extractPaletteAndIndex(data) {
  const px = data.data, n = data.width * data.height;
  const seen = new Map();
  const tmp = [];
  const idx = new Uint16Array(n);
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    const r = px[o], g = px[o + 1], b = px[o + 2];
    const key = (r << 16) | (g << 8) | b;
    let i = seen.get(key);
    if (i === undefined) { i = tmp.length; seen.set(key, i); tmp.push({ r, g, b, hex: rgbToHex(r, g, b) }); }
    idx[p] = i;
    if (tmp.length > 256) break;
  }
  const order = tmp.map((_, i) => i).sort((a, b) => _lumOf(tmp[a]) - _lumOf(tmp[b]));
  const remap = new Uint16Array(tmp.length);
  order.forEach((oldI, newI) => { remap[oldI] = newI; });
  for (let p = 0; p < n; p++) idx[p] = remap[idx[p]];
  return { palette: order.map((i) => tmp[i]), idx };
}

// Draw render to canvas, applying recolours; on hover, isolate one colour.
function redrawCanvas() {
  const R = state.render;
  if (!R) return;
  const W = R.w, H = R.h, idx = R.idx, n = idx.length;
  const out = new Uint8ClampedArray(n * 4);
  const oc = R.palette.map((c) => (state.colorMap[c.hex] ? hexToRgb(state.colorMap[c.hex]) : c));
  const hi = state.hoverIdx;
  const paper = { r: 247, g: 240, b: 225 };
  const outline = { r: 232, g: 72, b: 47 };
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    const i = idx[p];
    let c;
    if (hi == null || i === hi) {
      c = oc[i];
      if (hi != null) {
        const x = p % W, y = (p / W) | 0;
        const edge =
          (x > 0 && idx[p - 1] !== hi) || (x < W - 1 && idx[p + 1] !== hi) ||
          (y > 0 && idx[p - W] !== hi) || (y < H - 1 && idx[p + W] !== hi);
        if (edge) c = outline;
      }
    } else {
      const g = oc[i];
      c = { r: (g.r * 0.10 + paper.r * 0.90) | 0, g: (g.g * 0.10 + paper.g * 0.90) | 0, b: (g.b * 0.10 + paper.b * 0.90) | 0 };
    }
    out[o] = c.r; out[o + 1] = c.g; out[o + 2] = c.b; out[o + 3] = 255;
  }
  $("#bigCanvas").getContext("2d").putImageData(new ImageData(out, W, H), 0, 0);
}

function buildSwatches() {
  const R = state.render;
  const block = $("#paletteBlock");
  const wrap = $("#swatches");
  wrap.innerHTML = "";
  if (!R || R.palette.length > 32) { block.hidden = true; return; }
  block.hidden = false;
  $("#paletteCount").textContent = String(R.palette.length);

  R.palette.forEach((c, i) => {
    const shown = state.colorMap[c.hex] || c.hex;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "swatch" + (state.colorMap[c.hex] ? " remapped" : "");
    btn.innerHTML =
      `<span class="swatch-chip" style="background:${shown}"></span>` +
      `<span class="swatch-hex">${shown.toUpperCase()}</span>`;
    btn.addEventListener("mouseenter", () => {
      state.hoverIdx = i;
      $$(".swatch").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      redrawCanvas();
    });
    btn.addEventListener("mouseleave", () => { state.hoverIdx = null; btn.classList.remove("active"); redrawCanvas(); });
    btn.addEventListener("click", () => openRecolor(c));
    wrap.appendChild(btn);
  });
  $("#resetColors").hidden = Object.keys(state.colorMap).length === 0;
}

function openRecolor(c) {
  const picker = $("#swatchPicker");
  picker.value = state.colorMap[c.hex] || c.hex;
  picker.oninput = () => { state.colorMap[c.hex] = picker.value; redrawCanvas(); buildSwatches(); updateExport(); };
  picker.click();
}

function updateExport() {
  $("#bigCanvas").toBlob((blob) => {
    if (!blob) return;
    if (state._exportUrl) URL.revokeObjectURL(state._exportUrl);
    state._exportUrl = URL.createObjectURL(blob);
    const link = $("#exportPng");
    link.href = state._exportUrl;
    link.download = exportFilename();
    link.classList.remove("disabled");
  }, "image/png");
}

function exportFilename() {
  const base = (state.asset?.original_filename || "image").replace(/\.[^.]+$/, "");
  const style = state.styleId;
  const dims = state.render ? `_${state.render.w}x${state.render.h}` : "";
  return `${base}_${style}${dims}.png`;
}

// ── zoom + pan ──────────────────────────────────────────────────────────────
function bindZoom() {
  $("#zoomFit").addEventListener("click", () => setZoomFit());
  $("#zoomSlider").addEventListener("input", (e) => {
    state.zoomMode = "manual"; state.zoomPct = parseInt(e.target.value, 10);
    $("#zoomFit").classList.remove("active"); applyZoom();
  });
  $$(".zoom-step").forEach((b) =>
    b.addEventListener("click", () => {
      if (state.zoomMode === "fit") state.zoomPct = 100;
      state.zoomMode = "manual";
      state.zoomPct = Math.max(25, Math.min(400, state.zoomPct + parseInt(b.dataset.step, 10) * 25));
      $("#zoomSlider").value = String(state.zoomPct);
      $("#zoomFit").classList.remove("active"); applyZoom();
    }));
}
function setZoomFit() {
  state.zoomMode = "fit"; state.zoomPct = 100;
  $("#zoomSlider").value = "100"; $("#zoomFit").classList.add("active");
  applyZoom();
}
function applyZoom() {
  const cv = $("#bigCanvas");
  const nw = cv.width, nh = cv.height;
  if (!nw || !nh) return;
  const lb = document.querySelector(".lightbox");
  const availW = Math.max(80, lb.clientWidth - 40);
  const availH = Math.max(80, lb.clientHeight - 48);
  const fitScale = Math.min(availW / nw, availH / nh);
  const scale = state.zoomMode === "fit" ? fitScale : fitScale * (state.zoomPct / 100);
  cv.style.width = Math.max(1, Math.round(nw * scale)) + "px";
  cv.style.height = "auto";
  $("#zoomVal").textContent = state.zoomMode === "fit" ? "fit" : `${state.zoomPct}%`;
  updatePannable();
}
function updatePannable() {
  const lb = document.querySelector(".lightbox");
  const pan = lb.scrollWidth > lb.clientWidth + 1 || lb.scrollHeight > lb.clientHeight + 1;
  lb.classList.toggle("pannable", pan);
}
function bindPan() {
  const lb = document.querySelector(".lightbox");
  let down = false, sx = 0, sy = 0, sl = 0, st = 0;
  lb.addEventListener("mousedown", (e) => {
    if (!e.target.closest(".plate")) return;
    if (lb.scrollWidth <= lb.clientWidth && lb.scrollHeight <= lb.clientHeight) return;
    down = true; sx = e.clientX; sy = e.clientY; sl = lb.scrollLeft; st = lb.scrollTop;
    lb.classList.add("panning"); e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!down) return;
    lb.scrollLeft = sl - (e.clientX - sx);
    lb.scrollTop = st - (e.clientY - sy);
  });
  window.addEventListener("mouseup", () => { if (down) { down = false; lb.classList.remove("panning"); } });
}

document.addEventListener("DOMContentLoaded", init);
