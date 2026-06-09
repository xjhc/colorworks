/* ============================================================================
   Colorworks — studio (browser-only).
   Upload → live preview → pick a style → turn the knobs → recolour → export.
   Everything runs client-side: local decode, local resize, renderToneDither,
   canvas paint, and PNG export. No backend.
   ========================================================================== */
import "./styles.css";
import {
  STYLES,
  styleParams,
  DEFAULT_STYLE_ID,
  type ParamDef,
  type StyleDef,
} from "./schema";
import {
  renderToneDither,
  rgbToHex,
  parseColor,
  type RGB,
  type Raster,
  type RenderOptions,
} from "./colorworks";
import { renderDepixelate, type DepixelateOptions } from "./depixelate";
import { renderRepixel, detectCandidates, toGlyphText, type RepixelOptions } from "./repixel";
import { boxFit, conformIndexed, type FitMode } from "./output_size";

// Fixed seed — mirrors the studio's seed=42 so blue-noise/flow are stable.
const SEED = 42;

// ── tiny DOM helpers ──────────────────────────────────────────────────────────
function $<T extends HTMLElement = HTMLElement>(sel: string): T {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`missing element: ${sel}`);
  return el as T;
}
function $$(sel: string): HTMLElement[] {
  return Array.from(document.querySelectorAll(sel));
}

type ParamValue = string | number | boolean;

interface RenderState {
  w: number;
  h: number;
  idx: Uint16Array;
  palette: RGB[];
}

const state = {
  source: null as ImageBitmap | null,
  sourceName: "image",
  sourceW: 0,
  sourceH: 0,
  setup: { maxW: 360 as number | null, maxH: null as number | null, fit: "fit" as FitMode },
  styleId: DEFAULT_STYLE_ID,
  // Knob state bucketed by renderer so each mode keeps its own defaults (e.g.
  // Repixel's palette:"original"); the six tone-dither styles share one renderer
  // and thus keep carrying colors/contrast across each other.
  knobValues: {} as Record<string, Record<string, ParamValue>>,
  render: null as RenderState | null,
  colorMap: {} as Record<string, string>, // originalHex -> replacementHex
  hoverIdx: null as number | null,
  renderStart: 0,
  renderDebounce: 0 as ReturnType<typeof setTimeout> | 0,
  zoomMode: "fit" as "fit" | "manual",
  zoomPct: 100,
  exportUrl: "",
  thumbUrl: "",
  offscreen: null as HTMLCanvasElement | null,
  glyphText: "" as string, // braille+block text of the native repixel render (repixel only)
};

function styleById(id: string): StyleDef {
  return STYLES.find((s) => s.id === id) ?? STYLES[0];
}

/** The current style's knob bucket (created on first access), keyed by renderer. */
function knobBucket(): Record<string, ParamValue> {
  const renderer = styleById(state.styleId).renderer ?? "tone_dither";
  return (state.knobValues[renderer] ??= {});
}

// ── toast ─────────────────────────────────────────────────────────────────────
let toastTimer: ReturnType<typeof setTimeout> | 0 = 0;
function toast(msg: string): void {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  requestAnimationFrame(() => t.classList.add("show"));
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => (t.hidden = true), 250);
  }, 2600);
}

function showStudio(on: boolean): void {
  $("#emptyState").hidden = on;
  $("#editorView").hidden = !on;
  $("#inspectorEmpty").hidden = on;
  $("#inspectorBody").hidden = !on;
}

// ── source decode (local) ─────────────────────────────────────────────────────
function bindSource(): void {
  const dz = $("#dropzone");
  const input = $<HTMLInputElement>("#fileInput");
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) loadFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    }),
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
    }),
  );
  dz.addEventListener("drop", (e) => {
    const f = (e as DragEvent).dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });
}

async function loadFile(file: File): Promise<void> {
  $("#srcName").textContent = "Decoding…";
  let bitmap: ImageBitmap;
  try {
    // `from-image` applies EXIF orientation so portrait photos aren't sideways.
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    $("#srcName").textContent = "Unsupported";
    toast("Couldn't decode that image — try PNG, JPEG, WebP, or GIF (HEIC isn't supported in-browser)");
    return;
  }

  state.source = bitmap;
  state.sourceW = bitmap.width;
  state.sourceH = bitmap.height;
  state.sourceName = file.name || "image";

  if (state.thumbUrl) URL.revokeObjectURL(state.thumbUrl);
  state.thumbUrl = URL.createObjectURL(file);
  $("#dropzone").classList.add("loaded");
  const thumb = $("#srcThumb");
  thumb.classList.add("has-img");
  thumb.style.backgroundImage = `url(${state.thumbUrl})`;
  $("#srcName").textContent = state.sourceName;
  $("#srcMeta").textContent = `${bitmap.width} × ${bitmap.height}`;
  $("#topbarMeta").textContent = `${state.sourceName} · ${bitmap.width}×${bitmap.height}`;

  showStudio(true);
  applyStyle(state.styleId); // open straight into a live render
}

// ── output controls ───────────────────────────────────────────────────────────
function bindOutput(): void {
  $<HTMLInputElement>("#maxW").addEventListener("input", (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    state.setup.maxW = Number.isFinite(v) && v > 0 ? v : null;
    scheduleRender();
  });
  $<HTMLInputElement>("#maxH").addEventListener("input", (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    state.setup.maxH = Number.isFinite(v) && v > 0 ? v : null;
    scheduleRender();
  });
  $$("#fitGroup button").forEach((b) =>
    b.addEventListener("click", () => {
      $$("#fitGroup button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.setup.fit = (b.dataset.fit as FitMode) || "fit";
      scheduleRender();
    }),
  );
}

// ── style + knobs ─────────────────────────────────────────────────────────────
function buildStyleSelect(): void {
  const sel = $<HTMLSelectElement>("#styleSelect");
  sel.innerHTML = "";
  STYLES.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.label;
    sel.appendChild(opt);
  });
  sel.value = state.styleId;
  sel.addEventListener("change", () => applyStyle(sel.value));
}

function applyStyle(styleId: string): void {
  if (!STYLES.some((s) => s.id === styleId)) return;
  state.styleId = styleId;
  $<HTMLSelectElement>("#styleSelect").value = styleId;
  state.colorMap = {}; // a new look starts from its own palette
  state.hoverIdx = null;
  setZoomFit();
  buildKnobs();
  renderFocus();
}

function paramValue(p: ParamDef): ParamValue {
  let v = knobBucket()[p.key];
  if (v === undefined) v = p.default;
  if (p.options && p.options.length && !p.options.some((o) => String(o.value) === String(v))) {
    v = p.default;
  }
  return v;
}

function buildKnobs(): void {
  const wrap = $("#knobs");
  wrap.innerHTML = "";
  const style = styleById(state.styleId);
  const bucket = knobBucket();

  styleParams(style).forEach((p) => {
    if (p.key in style.fixed) return; // controlled by the Style selector — hidden
    const initial = paramValue(p);
    const row = document.createElement("div");
    row.className = "knob";
    row.id = `knob-${p.key}`;

    const persist = (val: ParamValue) => {
      bucket[p.key] = val;
    };

    if (p.type === "bool") {
      row.innerHTML = `<div class="knob-check"><span class="knob-label">${p.label}</span></div>`;
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = `param-${p.key}`;
      cb.checked = !!initial;
      row.querySelector(".knob-check")!.appendChild(cb);
      cb.addEventListener("change", () => {
        persist(cb.checked);
        updateKnobVisibility();
        scheduleRender();
      });
    } else if (p.type === "str" && p.uiHint === "color") {
      row.innerHTML = `<div class="knob-color"><span class="knob-label">${p.label}</span></div>`;
      const ci = document.createElement("input");
      ci.type = "color";
      ci.id = `param-${p.key}`;
      ci.value = String(initial);
      row.querySelector(".knob-color")!.appendChild(ci);
      ci.addEventListener("input", () => {
        persist(ci.value);
        scheduleRender();
      });
    } else if (p.options && p.options.length) {
      row.innerHTML = `<div class="knob-top"><span class="knob-label">${p.label}</span></div>`;
      const sel = document.createElement("select");
      sel.id = `param-${p.key}`;
      p.options.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = String(o.value);
        opt.textContent = o.label;
        if (String(o.value) === String(initial)) opt.selected = true;
        sel.appendChild(opt);
      });
      row.appendChild(sel);
      sel.addEventListener("change", () => {
        persist(p.type === "int" ? parseInt(sel.value, 10) : sel.value);
        updateKnobVisibility();
        scheduleRender();
      });
    } else {
      const digits = p.type === "int" ? 0 : 2;
      row.innerHTML = `<div class="knob-top"><span class="knob-label">${p.label}</span>
        <output class="knob-val" id="val-${p.key}">${Number(initial).toFixed(digits)}</output></div>`;
      const sl = document.createElement("input");
      sl.type = "range";
      sl.id = `param-${p.key}`;
      sl.min = String(p.min ?? 0);
      sl.max = String(p.max ?? 1);
      sl.step = String(p.step ?? (p.type === "int" ? 1 : 0.01));
      sl.value = String(initial);
      row.appendChild(sl);
      sl.addEventListener("input", () => {
        $(`#val-${p.key}`).textContent = Number(sl.value).toFixed(digits);
        persist(p.type === "int" ? parseInt(sl.value, 10) : parseFloat(sl.value));
        scheduleRender();
      });
    }
    wrap.appendChild(row);
  });
  updateKnobVisibility();
}

function updateKnobVisibility(): void {
  const style = styleById(state.styleId);
  styleParams(style).forEach((p) => {
    const row = document.getElementById(`knob-${p.key}`);
    if (!row || !p.visibleWhen) return;
    const dep = p.visibleWhen.param;
    let val: ParamValue | null;
    if (dep in style.fixed) {
      val = (style.fixed as Record<string, ParamValue>)[dep];
    } else {
      const inp = document.getElementById(`param-${dep}`) as HTMLInputElement | HTMLSelectElement | null;
      val = inp ? (inp instanceof HTMLInputElement && inp.type === "checkbox" ? inp.checked : inp.value) : null;
    }
    const visible = p.visibleWhen.equals.some((e) => String(e) === String(val));
    row.classList.toggle("hidden", !visible);
  });
}

function gatherValues(): Record<string, ParamValue> {
  const style = styleById(state.styleId);
  const out: Record<string, ParamValue> = {};
  styleParams(style).forEach((p) => {
    if (p.key in style.fixed) {
      out[p.key] = (style.fixed as Record<string, ParamValue>)[p.key];
      return;
    }
    const inp = document.getElementById(`param-${p.key}`) as HTMLInputElement | HTMLSelectElement | null;
    if (!inp) {
      out[p.key] = paramValue(p);
      return;
    }
    if (p.type === "bool") out[p.key] = (inp as HTMLInputElement).checked;
    else if (p.type === "int") out[p.key] = parseInt(inp.value, 10);
    else if (p.type === "float") out[p.key] = parseFloat(inp.value);
    else out[p.key] = inp.value;
  });
  return out;
}

function numParam(v: Record<string, ParamValue>, key: string, d: number): number {
  return typeof v[key] === "number" ? (v[key] as number) : d;
}

function toRenderOptions(v: Record<string, ParamValue>): RenderOptions {
  const num = (k: string, d: number) => (typeof v[k] === "number" ? (v[k] as number) : d);
  return {
    colors: num("colors", 4),
    palette: v.palette as RenderOptions["palette"],
    method: v.method as RenderOptions["method"],
    contrast: num("contrast", 1),
    midpoint: num("midpoint", 0.5),
    inkColor: String(v.ink_color ?? "#161616"),
    paperColor: String(v.paper_color ?? "#f4ebd9"),
    seed: SEED,
    params: {
      matrixSize: num("matrix_size", 8),
      noiseSize: num("noise_size", 64),
      frequency: num("frequency", 5),
      warp: num("warp", 7),
      angleDeg: num("angle_deg", 45),
      detail: num("detail", 2.5),
    },
  };
}

// ── local rasterisation (replaces server resize + asset fetch) ─────────────────
/** Draw the source into an offscreen canvas at the output size and read pixels.
 *  `fullRes` ignores the output-size setting (depixelate needs the native grid). */
function rasterizeSource(fullRes = false): Raster {
  const { maxW, maxH, fit } = state.setup;
  const sw = state.sourceW;
  const sh = state.sourceH;
  const mw = fullRes || !(maxW && maxW > 0) ? null : maxW;
  const mh = fullRes || !(maxH && maxH > 0) ? null : maxH;
  const { canvasW, canvasH, drawW, drawH, dx, dy } = boxFit(sw, sh, mw, mh, fit, false);

  const cv = state.offscreen ?? (state.offscreen = document.createElement("canvas"));
  cv.width = canvasW;
  cv.height = canvasH;
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, canvasW, canvasH);
  ctx.drawImage(state.source!, dx, dy, drawW, drawH);
  const data = ctx.getImageData(0, 0, canvasW, canvasH);
  return { width: canvasW, height: canvasH, data: data.data };
}

// ── render the preview (local) ────────────────────────────────────────────────
function scheduleRender(): void {
  if (state.renderDebounce) clearTimeout(state.renderDebounce);
  state.renderDebounce = setTimeout(renderFocus, 90);
}

function renderFocus(): void {
  if (!state.source) return;
  $("#plateLoading").hidden = false;
  // Defer one frame so the spinner can paint before a heavy synchronous render.
  requestAnimationFrame(() => {
    if (!state.source) return;
    state.renderStart = performance.now();
    const style = styleById(state.styleId);
    const vals = gatherValues();
    let repixelInfo = ""; // candidate pixel-size readout (repixel only)
    state.glyphText = ""; // reset; set below only for the repixel renderer
    // Depixelate and Repixel derive their own resolution from the full-res grid,
    // so they must see the source at native size (not the output-downscaled raster).
    const gridRenderer = style.renderer === "depixelate" || style.renderer === "repixel";
    const raster = rasterizeSource(gridRenderer);
    let res: ReturnType<typeof renderToneDither>;
    if (style.renderer === "depixelate") {
      res = renderDepixelate(raster, {
        block: numParam(vals, "block", 2),
        tau: numParam(vals, "tau", 45),
        pitch: numParam(vals, "pitch", 0),
        palette: vals.palette as DepixelateOptions["palette"],
        colors: numParam(vals, "colors", 4),
        inkColor: String(vals.ink_color ?? "#161616"),
        paperColor: String(vals.paper_color ?? "#f4ebd9"),
        keepMarks: vals.keep_marks === true,
        fillMult: numParam(vals, "fill_mult", 1),
      });
    } else if (style.renderer === "repixel") {
      const repixelOpts: RepixelOptions = {
        target: vals.target as RepixelOptions["target"],
        pitch: numParam(vals, "pitch", 0),
        tau: numParam(vals, "tau", 45),
        minLit: numParam(vals, "min_lit", 2),
        shade: vals.shade !== false,
        palette: vals.palette as RepixelOptions["palette"],
        colors: numParam(vals, "colors", 6),
        inkColor: String(vals.ink_color ?? "#161616"),
        paperColor: String(vals.paper_color ?? "#f4ebd9"),
        bgMode: vals.bg_mode as RepixelOptions["bgMode"],
        bgColor: String(vals.bg_color ?? "#181818"),
        spriteSat: numParam(vals, "sprite_sat", 0.3),
        eyeLuma: numParam(vals, "eye_luma", 45),
      };
      res = renderRepixel(raster, repixelOpts);
      // Glyph text needs a NATIVE 1px/cell grid (the 2x4 braille grouping). Composite
      // renders at source resolution (gapped dots), so derive its glyph text from a
      // separate fine pass rather than from `res`.
      state.glyphText = toGlyphText(
        vals.target === "composite"
          ? renderRepixel(raster, { ...repixelOpts, target: "fine" })
          : res,
      );
      // Candidate pixel sizes for the readout — the image may carry two scales.
      const cand = detectCandidates(raster);
      const star = (t: string) => (vals.target === t ? "●" : "○");
      repixelInfo =
        ` · ${star("fine")} fine ${cand.fine.toFixed(1)}px · ` +
        `${star("subject")} subject ${cand.subject.toFixed(1)}px`;
    } else {
      res = renderToneDither(raster, toRenderOptions(vals));
    }
    // The grid renderers detect + tile on the native grid, so their raw output
    // size is decoupled from the output-size control; scale to the requested size
    // so the preview and exported PNG honour it (other renderers already do).
    // EXCEPT composite: it draws gapped braille dots at source resolution, and
    // nearest-neighbour downscaling would alias the dots away — so it keeps native.
    if (gridRenderer && vals.target !== "composite") {
      const { maxW, maxH, fit } = state.setup;
      res = conformIndexed(res, maxW && maxW > 0 ? maxW : null, maxH && maxH > 0 ? maxH : null, fit);
    }
    state.render = { w: res.width, h: res.height, idx: res.indices, palette: res.palette };
    state.hoverIdx = null;

    const cv = $<HTMLCanvasElement>("#bigCanvas");
    cv.width = res.width;
    cv.height = res.height;
    redrawCanvas();
    buildSwatches();

    const elapsed = Math.round(performance.now() - state.renderStart);
    const name = styleById(state.styleId).label.split(" — ")[0];
    // repixelInfo is built only from numbers + bullet glyphs (no user input).
    $("#bigCaption").innerHTML = `<b>${name}</b> · ${res.width} × ${res.height} px${repixelInfo}`;
    $("#specDims").textContent = `${res.width} × ${res.height}`;
    $("#specRender").textContent = `${elapsed} ms`;
    $("#copyGlyphs").hidden = style.renderer !== "repixel";
    $("#plateLoading").hidden = true;

    applyZoom();
    updateExport();
  });
}

// ── canvas paint (recolour + hover isolation) ─────────────────────────────────
function redrawCanvas(): void {
  const R = state.render;
  if (!R) return;
  const { w: W, h: H, idx, palette } = R;
  const n = idx.length;
  const out = new Uint8ClampedArray(n * 4);
  const oc: RGB[] = palette.map((c) => {
    const m = state.colorMap[rgbToHex(c)];
    return m ? parseColor(m) : c;
  });
  const hi = state.hoverIdx;
  const paper: RGB = [247, 240, 225];
  const outline: RGB = [232, 72, 47];

  for (let p = 0, o = 0; p < n; p++, o += 4) {
    const i = idx[p];
    let c: RGB;
    if (hi === null || i === hi) {
      c = oc[i];
      if (hi !== null) {
        const x = p % W;
        const y = (p / W) | 0;
        const edge =
          (x > 0 && idx[p - 1] !== hi) ||
          (x < W - 1 && idx[p + 1] !== hi) ||
          (y > 0 && idx[p - W] !== hi) ||
          (y < H - 1 && idx[p + W] !== hi);
        if (edge) c = outline;
      }
    } else {
      const g = oc[i];
      c = [
        (g[0] * 0.1 + paper[0] * 0.9) | 0,
        (g[1] * 0.1 + paper[1] * 0.9) | 0,
        (g[2] * 0.1 + paper[2] * 0.9) | 0,
      ];
    }
    out[o] = c[0];
    out[o + 1] = c[1];
    out[o + 2] = c[2];
    out[o + 3] = 255;
  }
  $<HTMLCanvasElement>("#bigCanvas").getContext("2d")!.putImageData(new ImageData(out, W, H), 0, 0);
}

// ── palette swatches ──────────────────────────────────────────────────────────
function buildSwatches(): void {
  const R = state.render;
  const block = $("#paletteBlock");
  const wrap = $("#swatches");
  wrap.innerHTML = "";
  if (!R || R.palette.length > 32) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  $("#paletteCount").textContent = String(R.palette.length);

  R.palette.forEach((c, i) => {
    const hex = rgbToHex(c);
    const shown = state.colorMap[hex] || hex;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "swatch" + (state.colorMap[hex] ? " remapped" : "");
    // Build via DOM (not innerHTML): `shown` is a colour value, kept out of markup.
    const chip = document.createElement("span");
    chip.className = "swatch-chip";
    chip.style.background = shown;
    const label = document.createElement("span");
    label.className = "swatch-hex";
    label.textContent = shown.toUpperCase();
    btn.append(chip, label);
    btn.addEventListener("mouseenter", () => {
      state.hoverIdx = i;
      $$(".swatch").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      redrawCanvas();
    });
    btn.addEventListener("mouseleave", () => {
      state.hoverIdx = null;
      btn.classList.remove("active");
      redrawCanvas();
    });
    btn.addEventListener("click", () => openRecolor(hex));
    wrap.appendChild(btn);
  });
  $("#resetColors").hidden = Object.keys(state.colorMap).length === 0;
}

function openRecolor(hex: string): void {
  const picker = $<HTMLInputElement>("#swatchPicker");
  picker.value = state.colorMap[hex] || hex;
  picker.oninput = () => {
    state.colorMap[hex] = picker.value;
    redrawCanvas();
    buildSwatches();
    updateExport();
  };
  picker.click();
}

// ── export (local PNG + Web Crypto checksum) ──────────────────────────────────
function updateExport(): void {
  const cv = $<HTMLCanvasElement>("#bigCanvas");
  cv.toBlob((blob) => {
    if (!blob) return;
    if (state.exportUrl) URL.revokeObjectURL(state.exportUrl);
    state.exportUrl = URL.createObjectURL(blob);
    const link = $<HTMLAnchorElement>("#exportPng");
    link.href = state.exportUrl;
    link.download = exportFilename();
    link.classList.remove("disabled");
    void setChecksum(blob);
  }, "image/png");
}

async function setChecksum(blob: Blob): Promise<void> {
  try {
    const buf = await blob.arrayBuffer();
    const hash = await crypto.subtle.digest("SHA-256", buf);
    const hex = Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    $("#specHash").textContent = hex.slice(0, 12);
  } catch {
    $("#specHash").textContent = "—";
  }
}

function exportFilename(): string {
  const base = state.sourceName.replace(/\.[^.]+$/, "") || "image";
  const dims = state.render ? `_${state.render.w}x${state.render.h}` : "";
  return `${base}_${state.styleId}${dims}.png`;
}

// ── zoom + pan ────────────────────────────────────────────────────────────────
function bindZoom(): void {
  $("#zoomFit").addEventListener("click", () => setZoomFit());
  $<HTMLInputElement>("#zoomSlider").addEventListener("input", (e) => {
    state.zoomMode = "manual";
    state.zoomPct = parseInt((e.target as HTMLInputElement).value, 10);
    $("#zoomFit").classList.remove("active");
    applyZoom();
  });
  $$(".zoom-step").forEach((b) =>
    b.addEventListener("click", () => {
      if (state.zoomMode === "fit") state.zoomPct = 100;
      state.zoomMode = "manual";
      const step = parseInt(b.dataset.step || "0", 10);
      state.zoomPct = Math.max(25, Math.min(400, state.zoomPct + step * 25));
      $<HTMLInputElement>("#zoomSlider").value = String(state.zoomPct);
      $("#zoomFit").classList.remove("active");
      applyZoom();
    }),
  );
}

function setZoomFit(): void {
  state.zoomMode = "fit";
  state.zoomPct = 100;
  $<HTMLInputElement>("#zoomSlider").value = "100";
  $("#zoomFit").classList.add("active");
  applyZoom();
}

function applyZoom(): void {
  const cv = $<HTMLCanvasElement>("#bigCanvas");
  const nw = cv.width;
  const nh = cv.height;
  if (!nw || !nh) return;
  const lb = $(".lightbox");
  const availW = Math.max(80, lb.clientWidth - 40);
  const availH = Math.max(80, lb.clientHeight - 48);
  const fitScale = Math.min(availW / nw, availH / nh);
  const scale = state.zoomMode === "fit" ? fitScale : fitScale * (state.zoomPct / 100);
  cv.style.width = Math.max(1, Math.round(nw * scale)) + "px";
  cv.style.height = "auto";
  $("#zoomVal").textContent = state.zoomMode === "fit" ? "fit" : `${state.zoomPct}%`;
  updatePannable();
}

function updatePannable(): void {
  const lb = $(".lightbox");
  const pan = lb.scrollWidth > lb.clientWidth + 1 || lb.scrollHeight > lb.clientHeight + 1;
  lb.classList.toggle("pannable", pan);
}

function bindPan(): void {
  const lb = $(".lightbox");
  let down = false;
  let sx = 0;
  let sy = 0;
  let sl = 0;
  let st = 0;
  lb.addEventListener("mousedown", (e) => {
    const me = e as MouseEvent;
    if (!(me.target as HTMLElement).closest(".plate")) return;
    if (lb.scrollWidth <= lb.clientWidth && lb.scrollHeight <= lb.clientHeight) return;
    down = true;
    sx = me.clientX;
    sy = me.clientY;
    sl = lb.scrollLeft;
    st = lb.scrollTop;
    lb.classList.add("panning");
    me.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!down) return;
    lb.scrollLeft = sl - ((e as MouseEvent).clientX - sx);
    lb.scrollTop = st - ((e as MouseEvent).clientY - sy);
  });
  window.addEventListener("mouseup", () => {
    if (down) {
      down = false;
      lb.classList.remove("panning");
    }
  });
}

// ── glyph-text copy (braille + block re-encoding) ─────────────────────────────
async function copyGlyphText(): Promise<void> {
  const text = state.glyphText;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast("Glyph text copied to clipboard");
    return;
  } catch {
    // Clipboard blocked (e.g. insecure context) — fall back to a .txt download.
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.sourceName.replace(/\.[^.]+$/, "") || "image"}_glyphs.txt`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Clipboard unavailable — downloaded glyph text");
  }
}

// ── init ──────────────────────────────────────────────────────────────────────
function init(): void {
  buildStyleSelect();
  bindSource();
  bindOutput();
  bindZoom();
  bindPan();
  $("#resetColors").addEventListener("click", () => {
    state.colorMap = {};
    state.hoverIdx = null;
    redrawCanvas();
    buildSwatches();
    updateExport();
  });
  $("#copyGlyphs").addEventListener("click", () => copyGlyphText());
  showStudio(false);

  let resizeRAF = 0;
  window.addEventListener("resize", () => {
    if (state.zoomMode !== "fit") return;
    cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(applyZoom);
  });
}

init();
