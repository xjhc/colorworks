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
  type StyleIntent,
} from "./schema";
import {
  renderToneDither,
  paintIndices,
  sharpenRaster,
  rgbToHex,
  parseColor,
  type RGB,
  type Raster,
  type RenderOptions,
  type RenderResult,
  type PinnedColor,
  type PinMode,
} from "./colorworks";
import { renderDepixelate, type DepixelateOptions } from "./depixelate";
import { renderBlockMosaic, type BlockMosaicOptions } from "./blockmosaic";
import { renderKnit, type KnitOptions, type KnitSymbolMode } from "./knit";
import { renderRepixel, detectCandidates, toGlyphText, type RepixelOptions } from "./repixel";
import { boxFit, conformIndexed, type FitMode } from "./output_size";
import { decodeToBitmap, classifyFile, decodeLabel } from "./decode";
import { renderHalftone, type HalftoneResult } from "./halftone";
import { computeProofMetrics, formatBytes } from "./proof";
import { canEncodeIndexedPng, encodeIndexedPng, isLosslessWebp } from "./png";

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
type View = "single" | "grid";

interface RenderState {
  w: number;
  h: number;
  idx: Uint16Array;
  palette: RGB[];
  swatchCount?: number;
  knit?: {
    cols: number;
    rows: number;
    cell: number;
    stitchIdx: Uint16Array;
    symbols: string[];
    symbolMode: KnitSymbolMode;
  };
  plates?: HalftoneResult["plates"];
  screen?: HalftoneResult["screen"];
  cell?: number;
}

interface SetupState {
  maxW: number | null;
  maxH: number | null;
  fit: FitMode;
}

const state = {
  source: null as ImageBitmap | null,
  sourceName: "image",
  sourceW: 0,
  sourceH: 0,
  setupValues: {} as Record<string, SetupState>,
  styleId: DEFAULT_STYLE_ID,
  // Knob state bucketed by renderer so each mode keeps its own defaults (e.g.
  // Repixel's palette:"original"); the six tone-dither styles share one renderer
  // and thus keep carrying colors/contrast across each other.
  knobValues: {} as Record<string, Record<string, ParamValue>>,
  render: null as RenderState | null,
  colorMap: {} as Record<string, string>, // originalHex -> replacementHex
  customPalette: [
    [28, 23, 18],
    [232, 72, 47],
    [245, 179, 42],
    [43, 58, 140],
  ] as RGB[],
  pins: [] as PinnedColor[], // colours pinned BEFORE generation (lock/boost/exclude)
  hoverIdx: null as number | null,
  renderStart: 0,
  renderDebounce: 0 as ReturnType<typeof setTimeout> | 0,
  zoomMode: "fit" as "fit" | "manual",
  zoomPct: 100,
  exportUrls: {} as Record<string, string>,
  exportSeq: 0,
  exportTimer: 0 as ReturnType<typeof setTimeout> | 0,
  // Requested nearest-neighbour export multiplier; clamped by EXPORT_MAX_* below.
  // Decouples the dither grain (output size) from the size of the exported file.
  exportScale: 4,
  thumbUrl: "",
  offscreen: null as HTMLCanvasElement | null,
  // The render at native resolution — the source of truth for export. #bigCanvas
  // is only its on-screen presentation (which may be a smoothed reduction).
  native: null as HTMLCanvasElement | null,
  hoverView: null as HTMLCanvasElement | null, // native size, hovered ink isolated
  exportCanvas: null as HTMLCanvasElement | null,
  glyphText: "" as string, // braille+block text of the native repixel render (repixel only)
  // Palette mode before a swatch edit promoted it to exact inks (tone dither);
  // non-null means the palette block offers "revert".
  paletteBeforeEdit: null as string | null,
  view: "single" as View,
  gridSeq: 0,
  gridTimer: 0 as ReturnType<typeof setTimeout> | 0,
  gridDirty: true,
};

function styleById(id: string): StyleDef {
  return STYLES.find((s) => s.id === id) ?? STYLES[0];
}

function rendererKey(style = styleById(state.styleId)): string {
  return style.renderer ?? "tone_dither";
}

/** Output size is shared by every image renderer (switching style must not
 *  silently change the output size); knit keeps its own, because there W/H are
 *  stitch counts rather than pixels. */
function setupKey(style: StyleDef): string {
  return rendererKey(style) === "knit" ? "knit" : "image";
}

function setupFor(style: StyleDef): SetupState {
  const key = setupKey(style);
  return (state.setupValues[key] ??= key === "knit"
    ? { maxW: 200, maxH: null, fit: "fit" }
    : { maxW: 360, maxH: null, fit: "fit" });
}

function setupBucket(): SetupState {
  return setupFor(styleById(state.styleId));
}

/** A style's knob bucket (created on first access), keyed by renderer. */
function knobBucketFor(style: StyleDef): Record<string, ParamValue> {
  return (state.knobValues[rendererKey(style)] ??= {});
}

function knobBucket(): Record<string, ParamValue> {
  return knobBucketFor(styleById(state.styleId));
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
  const input = $<HTMLInputElement>("#fileInput");
  const browse = () => input.click();
  $("#dropzone").addEventListener("click", browse);
  $("#emptyPick").addEventListener("click", browse);
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) loadFile(input.files[0]);
    input.value = ""; // choosing the same file again should still load it
  });

  // The whole window is a drop target: a file dropped anywhere else would make
  // the browser navigate to it and throw the session away.
  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
  let depth = 0; // dragenter/leave fire per element; count to know when it truly left
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    document.body.classList.add("dropping");
  });
  window.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.dataTransfer!.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) document.body.classList.remove("dropping");
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    document.body.classList.remove("dropping");
    const f = e.dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });

  // Paste an image (a screenshot, a copied photo). Text pastes — hex inks into
  // the palette box — carry no image file and pass through untouched.
  window.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items ?? []).find(
      (it) => it.kind === "file" && it.type.startsWith("image/"),
    );
    const file = item?.getAsFile();
    if (!file) return;
    e.preventDefault();
    loadFile(file);
  });
}

async function loadFile(file: File, restored?: { view: View }): Promise<void> {
  $("#srcName").textContent = decodeLabel(classifyFile(file));
  let bitmap: ImageBitmap;
  try {
    // Routes by format: raster → createImageBitmap (EXIF-oriented), SVG →
    // canvas rasterise, HEIC/TIFF → lazily-loaded decoders. See ./decode.
    bitmap = await decodeToBitmap(file);
  } catch {
    $("#srcName").textContent = "Unsupported";
    toast("Couldn't decode that file — try PNG, JPEG, WebP, GIF, AVIF, SVG, TIFF, or HEIC");
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
  buildGrid();
  applyStyle(state.styleId); // render the selected style behind the grid
  // A new image opens on every style side by side; a restored one where it was.
  setView(restored?.view ?? "grid");
  if (!restored) void saveSourceFile(file);
}

// ── session (survives a refresh) ──────────────────────────────────────────────
// Settings live in localStorage; the source image in IndexedDB (too big for
// localStorage). Both are best-effort: private windows or blocked storage just
// mean a refresh starts fresh.
const SESSION_KEY = "colorworks.session.v1";
const SOURCE_DB = "colorworks";
const SOURCE_MAX_BYTES = 64 * 1024 * 1024;

interface SavedSession {
  styleId: string;
  knobValues: Record<string, Record<string, ParamValue>>;
  setupValues: Record<string, SetupState>;
  pins: PinnedColor[];
  customPalette: RGB[];
  exportScale: number;
  paletteBeforeEdit: string | null;
  view: View;
}

let sessionTimer: ReturnType<typeof setTimeout> | 0 = 0;
function saveSessionSoon(): void {
  if (sessionTimer) clearTimeout(sessionTimer);
  sessionTimer = setTimeout(saveSession, 400);
}

function saveSession(): void {
  const saved: SavedSession = {
    styleId: state.styleId,
    knobValues: state.knobValues,
    setupValues: state.setupValues,
    pins: state.pins,
    customPalette: state.customPalette,
    exportScale: state.exportScale,
    paletteBeforeEdit: state.paletteBeforeEdit,
    view: state.view,
  };
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(saved));
  } catch {
    // storage full or blocked — nothing to do
  }
}

/** Read the saved settings back into `state`. Anything malformed is dropped
 *  rather than trusted: the knob code re-validates option values, and an
 *  unknown style falls back to the default. */
function restoreSession(): { view: View } | null {
  let saved: Partial<SavedSession>;
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    saved = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!saved || typeof saved !== "object") return null;
  const obj = (v: unknown) => !!v && typeof v === "object" && !Array.isArray(v);
  if (typeof saved.styleId === "string" && STYLES.some((st) => st.id === saved.styleId)) state.styleId = saved.styleId;
  if (obj(saved.knobValues)) state.knobValues = saved.knobValues!;
  if (obj(saved.setupValues)) state.setupValues = saved.setupValues!;
  if (Array.isArray(saved.pins)) {
    state.pins = saved.pins.filter((p) => p && /^#[0-9a-f]{6}$/i.test(p.hex) && ["lock", "boost", "exclude"].includes(p.mode));
  }
  if (Array.isArray(saved.customPalette) && saved.customPalette.length >= 2 &&
      saved.customPalette.every((c) => Array.isArray(c) && c.length === 3 && c.every((v) => Number.isInteger(v) && v >= 0 && v <= 255))) {
    state.customPalette = saved.customPalette;
  }
  if (EXPORT_SCALES.includes(saved.exportScale as number)) state.exportScale = saved.exportScale!;
  if (typeof saved.paletteBeforeEdit === "string") state.paletteBeforeEdit = saved.paletteBeforeEdit;
  return { view: saved.view === "single" ? "single" : "grid" };
}

function openSourceDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SOURCE_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore("files");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveSourceFile(file: File): Promise<void> {
  if (file.size > SOURCE_MAX_BYTES) return;
  try {
    const db = await openSourceDb();
    const tx = db.transaction("files", "readwrite");
    tx.objectStore("files").put(file, "source");
    tx.oncomplete = () => db.close();
  } catch {
    // storage unavailable — the session just won't survive a refresh
  }
}

async function loadSourceFile(): Promise<File | null> {
  try {
    const db = await openSourceDb();
    return await new Promise<File | null>((resolve) => {
      const req = db.transaction("files").objectStore("files").get("source");
      req.onsuccess = () => {
        db.close();
        resolve(req.result instanceof Blob ? (req.result as File) : null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    });
  } catch {
    return null;
  }
}

// ── output controls ───────────────────────────────────────────────────────────
function bindOutput(): void {
  $<HTMLInputElement>("#maxW").addEventListener("input", (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    setupBucket().maxW = Number.isFinite(v) && v > 0 ? v : null;
    scheduleRender();
  });
  $<HTMLInputElement>("#maxH").addEventListener("input", (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    setupBucket().maxH = Number.isFinite(v) && v > 0 ? v : null;
    scheduleRender();
  });
  $$("#fitGroup button").forEach((b) =>
    b.addEventListener("click", () => {
      $$("#fitGroup button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      setupBucket().fit = (b.dataset.fit as FitMode) || "fit";
      scheduleRender();
    }),
  );
}

function updateOutputPanel(): void {
  const style = styleById(state.styleId);
  const knit = rendererKey(style) === "knit";
  const setup = setupBucket();
  const maxW = $<HTMLInputElement>("#maxW");
  const maxH = $<HTMLInputElement>("#maxH");
  maxW.value = setup.maxW === null ? "" : String(setup.maxW);
  maxH.value = setup.maxH === null ? "" : String(setup.maxH);
  maxW.min = knit ? "40" : "16";
  maxW.max = knit ? "400" : "4096";
  maxH.min = knit ? "1" : "16";
  maxH.max = knit ? "800" : "4096";
  maxH.placeholder = "auto";
  $("#outputTitle").textContent = knit ? "Chart size" : "Output size";
  $("#maxWLabel").textContent = knit ? "stitches W" : "max W";
  $("#maxHLabel").textContent = knit ? "rows H" : "max H";
  $("#outputNote").textContent = knit
    ? "For knit charts, W/H are stitch counts. Leave rows H auto to derive it from row gauge; cell size controls PNG scale."
    : "Low sizes (~150–250px) give the chunky, pixelated dither look. Everything renders in your browser.";
  $("#fitGroup").hidden = knit;
  $$("#fitGroup button").forEach((b) => b.classList.toggle("active", b.dataset.fit === setup.fit));
}

// ── style + knobs ─────────────────────────────────────────────────────────────
const INTENTS: Record<StyleIntent, { label: string; blurb: string }> = {
  faithful: { label: "Faithful palette", blurb: "Exact inks, stable mix" },
  organic: { label: "Organic grain", blurb: "Soft, irregular texture" },
  ordered: { label: "Graphic pattern", blurb: "Crisp, repeating marks" },
  print: { label: "Print screen", blurb: "Spot or CMYK plates" },
  specialty: { label: "More workflows", blurb: "Tools for specific jobs" },
};

function buildStyleSelect(): void {
  const sel = $<HTMLSelectElement>("#styleSelect");
  sel.innerHTML = "";
  (Object.keys(INTENTS) as StyleIntent[]).forEach((intent) => {
    const styles = STYLES.filter((s) => s.intent === intent);
    if (!styles.length) return;
    const group = document.createElement("optgroup");
    group.label = INTENTS[intent].label;
    styles.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label;
      group.appendChild(opt);
    });
    sel.appendChild(group);
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
  updateOutputPanel();
  buildKnobs();
  updateStyleUI();
  markSelectedVariant();
  renderFocus();
}

function updateStyleUI(): void {
  const style = styleById(state.styleId);
  $("#styleDescription").textContent = style.description;
  updateCustomPaletteVisibility();
}

function paramValueFor(style: StyleDef, p: ParamDef): ParamValue {
  let v = knobBucketFor(style)[p.key];
  if (v === undefined) v = p.default;
  if (p.options && p.options.length && !p.options.some((o) => String(o.value) === String(v))) {
    v = p.default;
  }
  return v;
}

function paramValue(p: ParamDef): ParamValue {
  return paramValueFor(styleById(state.styleId), p);
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
        if (p.key === "palette") state.paletteBeforeEdit = null;
        updateKnobVisibility();
        updateCustomPaletteVisibility();
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
  updateCustomPaletteVisibility();
}

function updateKnobVisibility(): void {
  const style = styleById(state.styleId);
  styleParams(style).forEach((p) => {
    const row = document.getElementById(`knob-${p.key}`);
    if (!row || !p.visibleWhen) return;
    const conds = Array.isArray(p.visibleWhen) ? p.visibleWhen : [p.visibleWhen];
    const visible = conds.every((cond) => {
      const dep = cond.param;
      let val: ParamValue | null;
      if (dep in style.fixed) {
        val = (style.fixed as Record<string, ParamValue>)[dep];
      } else {
        const inp = document.getElementById(`param-${dep}`) as HTMLInputElement | HTMLSelectElement | null;
        val = inp ? (inp instanceof HTMLInputElement && inp.type === "checkbox" ? inp.checked : inp.value) : null;
      }
      return cond.equals.some((e) => String(e) === String(val));
    });
    row.classList.toggle("hidden", !visible);
  });
}

/** A style's full parameter set: its fixed values, else the knob bucket (which
 *  every knob writes on input), else the schema default. Reads no DOM, so the
 *  variant grid can evaluate styles that aren't on screen. */
function valuesFor(style: StyleDef): Record<string, ParamValue> {
  const out: Record<string, ParamValue> = {};
  styleParams(style).forEach((p) => {
    out[p.key] = p.key in style.fixed
      ? (style.fixed as Record<string, ParamValue>)[p.key]
      : paramValueFor(style, p);
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
    customPalette: state.customPalette,
    method: v.method as RenderOptions["method"],
    contrast: num("contrast", 1),
    midpoint: num("midpoint", 0.5),
    inkColor: String(v.ink_color ?? "#161616"),
    paperColor: String(v.paper_color ?? "#f4ebd9"),
    seed: SEED,
    pins: state.pins,
    params: {
      matrixSize: num("matrix_size", 8),
      noiseSize: num("noise_size", 64),
      frequency: num("frequency", 5),
      warp: num("warp", 7),
      angleDeg: num("angle_deg", 45),
      detail: num("detail", 2.5),
      candidates: num("candidates", 0) || undefined, // 0 ⇒ let the renderer pick 2×palette
      ditherStrength: num("dither_strength", 0.8),
      serpentine: v.serpentine !== false, // default on
      spread: num("spread", 0),
    },
  };
}

// ── local rasterisation (replaces server resize + asset fetch) ─────────────────
/** Draw the source into an offscreen canvas at the output size and read pixels.
 *  `fullRes` ignores the output-size setting (depixelate needs the native grid).
 *  `sharpen` > 0 returns an unsharp-masked copy of that raster.
 *  `memo`, when given, reuses rasters across calls with the same geometry (the
 *  variant grid renders ~20 styles from the same few rasters; renderers never
 *  write to their input). */
function rasterizeSource(setup: SetupState, fullRes = false, memo?: Map<string, Raster>, sharpen = 0): Raster {
  if (sharpen > 0) {
    const base = rasterizeSource(setup, fullRes, memo);
    const key = `${setup.maxW}|${setup.maxH}|${setup.fit}|${fullRes}|s${sharpen}`;
    const hit = memo?.get(key);
    if (hit) return hit;
    const out = sharpenRaster(base, sharpen);
    memo?.set(key, out);
    return out;
  }
  const { maxW, maxH, fit } = setup;
  const sw = state.sourceW;
  const sh = state.sourceH;
  const mw = fullRes || !(maxW && maxW > 0) ? null : maxW;
  const mh = fullRes || !(maxH && maxH > 0) ? null : maxH;
  const key = `${mw}|${mh}|${fit}`;
  const hit = memo?.get(key);
  if (hit) return hit;
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
  const raster = { width: canvasW, height: canvasH, data: data.data };
  memo?.set(key, raster);
  return raster;
}

// ── render one style (shared by the main preview and the variant grid) ────────
interface StyleRender {
  raster: Raster; // the input, for proof metrics
  res: RenderResult;
  info: string; // caption suffix (repixel candidates / knit stats)
  swatchCount?: number;
  knit?: RenderState["knit"];
  halftone?: Pick<HalftoneResult, "plates" | "screen" | "cell">;
  glyphText?: string;
}

function renderStyle(
  style: StyleDef,
  vals: Record<string, ParamValue>,
  setup: SetupState,
  memo = new Map<string, Raster>(),
): StyleRender {
  // Depixelate, Repixel, and Knit derive their own resolution from the full-res
  // grid, so they must see the source at native size (not the output-downscaled
  // raster). Knit also owns its exact chart size and is never conformed below.
  const nativeRes =
    style.renderer === "depixelate" ||
    style.renderer === "repixel" ||
    style.renderer === "knit";
  // `raster` (the unsharpened source) is what the proof compares against and
  // what palettes are drawn from; `input` is what gets dithered. Only styles
  // with a sharpen knob (the tone dithers) have one.
  const raster = rasterizeSource(setup, nativeRes, memo);
  const input = rasterizeSource(setup, nativeRes, memo, numParam(vals, "sharpen", 0));
  const out: Omit<StyleRender, "res"> = { raster, info: "" };
  let res: RenderResult;
  if (style.renderer === "halftone") {
    const hres = renderHalftone(raster, {
      separation: vals.separation as "spot" | "cmyk",
      screen: vals.screen as "am" | "fm" | "hybrid",
      cell: numParam(vals, "cell", 8),
      angle: numParam(vals, "angle", 45),
      dotShape: vals.dot_shape as "round" | "square" | "line",
      dotGain: numParam(vals, "dot_gain", 0),
      inkColor: String(vals.ink_color ?? "#1c1712"),
      paperColor: String(vals.paper_color ?? "#f4ebd9"),
      seed: SEED,
    });
    res = hres;
    out.halftone = hres;
  } else if (style.renderer === "depixelate") {
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
      pins: state.pins,
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
      pins: state.pins,
    };
    res = renderRepixel(raster, repixelOpts);
    // Glyph text needs a NATIVE 1px/cell grid (the 2x4 braille grouping). Composite
    // renders at source resolution (gapped dots), so derive its glyph text from a
    // separate fine pass rather than from `res`.
    out.glyphText = toGlyphText(
      vals.target === "composite"
        ? renderRepixel(raster, { ...repixelOpts, target: "fine" })
        : res,
    );
    // Candidate pixel sizes for the readout — the image may carry two scales.
    const cand = detectCandidates(raster);
    const star = (t: string) => (vals.target === t ? "●" : "○");
    out.info =
      ` · ${star("fine")} fine ${cand.fine.toFixed(1)}px · ` +
      `${star("subject")} subject ${cand.subject.toFixed(1)}px`;
  } else if (style.renderer === "block_mosaic") {
    res = renderBlockMosaic(raster, {
      block: numParam(vals, "block", 2),
      cell: numParam(vals, "cell", 6),
      colors: numParam(vals, "colors", 4),
      palette: vals.palette as BlockMosaicOptions["palette"],
      inkColor: String(vals.ink_color ?? "#161616"),
      paperColor: String(vals.paper_color ?? "#f4ebd9"),
      library: vals.library as BlockMosaicOptions["library"],
      learn: numParam(vals, "learn", 6),
      libraryBias: numParam(vals, "library_bias", 0.5),
      method: vals.method as BlockMosaicOptions["method"],
      contrast: numParam(vals, "contrast", 1),
      midpoint: numParam(vals, "midpoint", 0.5),
      pins: state.pins,
    });
  } else if (style.renderer === "knit") {
    const kres = renderKnit(raster, {
      stitches: setup.maxW && setup.maxW > 0 ? setup.maxW : 200,
      rows: setup.maxH && setup.maxH > 0 ? setup.maxH : undefined,
      rowGauge: numParam(vals, "row_gauge", 1.4),
      cell: numParam(vals, "cell", 8),
      colors: numParam(vals, "colors", 5),
      palette: vals.palette as KnitOptions["palette"],
      inkColor: String(vals.ink_color ?? "#161616"),
      paperColor: String(vals.paper_color ?? "#f4ebd9"),
      method: vals.method as KnitOptions["method"],
      symbols: vals.symbols as KnitOptions["symbols"],
      grid: vals.grid as KnitOptions["grid"],
      tidy: vals.tidy === true,
      contrast: numParam(vals, "contrast", 1),
      midpoint: numParam(vals, "midpoint", 0.5),
      pins: state.pins,
    });
    res = kres;
    out.swatchCount = kres.yarns.length;
    out.knit = {
      cols: kres.stitchesWide,
      rows: kres.rowsTall,
      cell: kres.cellSize,
      stitchIdx: kres.stitchIndices,
      symbols: kres.symbols,
      symbolMode: (vals.symbols as KnitSymbolMode) ?? "cells",
    };
    const n = kres.yarns.length;
    const stats = kres.stranding;
    out.info =
      ` · ${kres.stitchesWide} × ${kres.rowsTall} sts · ${n} yarn${n === 1 ? "" : "s"}` +
      ` · max float ${stats.maxFloat}` +
      ` · max ${stats.maxYarnsPerRow} yarns/row`;
  } else {
    res = renderToneDither(input, { ...toRenderOptions(vals), paletteSource: raster });
  }
  // The grid renderers detect + tile on the native grid, so their raw output
  // size is decoupled from the output-size control; scale to the requested size
  // so the preview and exported PNG honour it (other renderers already do).
  // EXCEPT composite: it draws gapped braille dots at source resolution, and
  // nearest-neighbour downscaling would alias the dots away — so it keeps native.
  // Knit is excluded too: it owns its exact chart size (cell-px stitches + grid).
  const conformable = style.renderer === "depixelate" || style.renderer === "repixel";
  if (conformable && vals.target !== "composite") {
    const { maxW, maxH, fit } = setup;
    res = conformIndexed(res, maxW && maxW > 0 ? maxW : null, maxH && maxH > 0 ? maxH : null, fit);
  }
  return { ...out, res };
}

// ── render the preview (local) ────────────────────────────────────────────────
function scheduleRender(): void {
  if (state.renderDebounce) clearTimeout(state.renderDebounce);
  state.renderDebounce = setTimeout(renderFocus, 90);
  scheduleGrid();
}

function renderFocus(): void {
  if (!state.source) return;
  $("#plateLoading").hidden = false;
  // Defer one frame so the spinner can paint before a heavy synchronous render.
  requestAnimationFrame(() => {
    if (!state.source) return;
    state.renderStart = performance.now();
    const style = styleById(state.styleId);
    const vals = valuesFor(style);
    const out = renderStyle(style, vals, setupBucket());
    const res = out.res;
    state.glyphText = out.glyphText ?? "";
    state.render = {
      w: res.width, h: res.height, idx: res.indices, palette: res.palette,
      swatchCount: out.swatchCount, knit: out.knit,
      plates: out.halftone?.plates, screen: out.halftone?.screen, cell: out.halftone?.cell,
    };
    state.hoverIdx = null;

    redrawCanvas();
    buildSwatches();
    buildPins();

    const elapsed = Math.round(performance.now() - state.renderStart);
    const name = style.label.split(" — ")[0];
    // out.info is built only from numbers + bullet glyphs (no user input).
    $("#bigCaption").innerHTML = `<b>${name}</b> · ${res.width} × ${res.height} px${out.info}`;
    $("#specDims").textContent = `${res.width} × ${res.height}`;
    updateScaleUI();
    $("#specRender").textContent = `${elapsed} ms`;
    const proof = computeProofMetrics(out.raster, res);
    $("#specColors").textContent = proof ? String(proof.usedColors) : "n/a";
    $("#specError").textContent = proof ? `ΔE ${proof.viewingError.toFixed(1)}` : "n/a";
    $("#specTone").textContent = proof ? `${proof.lowFrequencyError.toFixed(1)}%` : "n/a";
    buildPlateStrip();
    $("#copyGlyphs").hidden = style.renderer !== "repixel";
    $("#plateLoading").hidden = true;

    applyZoom();
    scheduleExport();
    saveSessionSoon();
  });
}

// ── variant grid ("All styles") ───────────────────────────────────────────────
// Every style rendered for the current image through renderStyle — the same path
// as the main preview — so a card shows what clicking it will open. Outputs
// larger than GRID_MAX_SIDE are rendered smaller (and the hint says so), which
// keeps a full pass to roughly a second.
const GRID_MAX_SIDE = 600;

/** The image output box, shrunk to GRID_MAX_SIDE. Knit keeps its stitch counts. */
function gridSetup(style: StyleDef): { setup: SetupState; reduced: boolean } {
  const setup = setupFor(style);
  if (setupKey(style) === "knit") return { setup, reduced: false };
  const mw = setup.maxW && setup.maxW > 0 ? setup.maxW : null;
  const mh = setup.maxH && setup.maxH > 0 ? setup.maxH : null;
  const { canvasW, canvasH } = boxFit(state.sourceW, state.sourceH, mw, mh, setup.fit, false);
  const long = Math.max(canvasW, canvasH);
  if (long <= GRID_MAX_SIDE) return { setup, reduced: false };
  // Shrink the box the output actually occupies (fit never upscales, so the
  // requested max W can be far larger than the real output).
  const f = GRID_MAX_SIDE / long;
  return {
    setup: {
      maxW: Math.max(16, Math.round(canvasW * f)),
      maxH: Math.max(16, Math.round(canvasH * f)),
      fit: setup.fit,
    },
    reduced: true,
  };
}

function setView(view: View): void {
  state.view = view;
  $$("#viewGroup button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#variantView").hidden = view !== "grid";
  $("#singleView").hidden = view !== "single";
  $("#gridHint").hidden = view !== "grid";
  $("#stageExport").hidden = view !== "single";
  saveSessionSoon();
  if (view === "grid") {
    markSelectedVariant();
    if (state.gridDirty) renderGrid();
  } else {
    applyZoom(); // the lightbox had no size while hidden
  }
}

function bindView(): void {
  $$("#viewGroup button").forEach((b) =>
    b.addEventListener("click", () => setView(b.dataset.view === "grid" ? "grid" : "single")),
  );
}

/** One card per style, grouped by intent. Built once per source image. */
function buildGrid(): void {
  const view = $("#variantView");
  view.innerHTML = "";
  const { setup } = gridSetup(STYLES[0]);
  const box = boxFit(
    state.sourceW, state.sourceH,
    setup.maxW && setup.maxW > 0 ? setup.maxW : null,
    setup.maxH && setup.maxH > 0 ? setup.maxH : null,
    setup.fit, false,
  );
  // Preview boxes follow the output's aspect, clamped so extreme panoramas or
  // phone screenshots don't make absurd cards; the canvas is contained inside.
  const aspect = Math.max(0.6, Math.min(1.8, box.canvasW / box.canvasH));
  (Object.keys(INTENTS) as StyleIntent[]).forEach((intent) => {
    const styles = STYLES.filter((s) => s.intent === intent);
    if (!styles.length) return;
    const group = document.createElement("section");
    group.className = "variant-group";
    const head = document.createElement("div");
    head.className = "variant-group-head";
    const h = document.createElement("h3");
    h.textContent = INTENTS[intent].label;
    const blurb = document.createElement("span");
    blurb.textContent = INTENTS[intent].blurb;
    head.append(h, blurb);
    const grid = document.createElement("div");
    grid.className = "variant-grid";
    styles.forEach((style) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "card stale";
      card.dataset.style = style.id;
      const preview = document.createElement("div");
      preview.className = "card-preview";
      preview.style.aspectRatio = String(aspect);
      const spin = document.createElement("div");
      spin.className = "spinner";
      preview.append(spin);
      const label = document.createElement("div");
      label.className = "card-label";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = style.label.split(" — ")[0];
      const desc = document.createElement("div");
      desc.className = "desc";
      desc.textContent = style.description;
      label.append(name, desc);
      card.append(preview, label);
      card.addEventListener("click", () => {
        applyStyle(style.id);
        setView("single");
      });
      grid.append(card);
    });
    group.append(head, grid);
    view.append(group);
  });
  state.gridDirty = true;
  markSelectedVariant();
}

function markSelectedVariant(): void {
  $$("#variantView .card").forEach((c) => c.classList.toggle("selected", c.dataset.style === state.styleId));
}

/** Knob, palette, pin, or size changes re-render the grid once things settle.
 *  Cards dim immediately so a stale thumbnail never reads as current. */
function scheduleGrid(): void {
  state.gridDirty = true;
  state.gridSeq++; // cancel any pass in flight
  if (state.view !== "grid") return;
  $$("#variantView .card").forEach((c) => c.classList.add("stale"));
  if (state.gridTimer) clearTimeout(state.gridTimer);
  state.gridTimer = setTimeout(renderGrid, 250);
}

let gridNative: HTMLCanvasElement | null = null;

function paintCard(card: HTMLElement, res: RenderResult): void {
  const src = (gridNative ??= document.createElement("canvas"));
  src.width = res.width;
  src.height = res.height;
  src.getContext("2d")!.putImageData(new ImageData(paintIndices(res) as Uint8ClampedArray<ArrayBuffer>, res.width, res.height), 0, 0);
  const box = card.querySelector<HTMLElement>(".card-preview")!;
  let cv = box.querySelector("canvas");
  if (!cv) {
    box.innerHTML = "";
    cv = document.createElement("canvas");
    box.append(cv);
  }
  const scale = Math.min(box.clientWidth / res.width, box.clientHeight / res.height);
  cv.style.width = `${paintScaled(cv, src, scale * (window.devicePixelRatio || 1)).toFixed(3)}px`;
  cv.style.height = "auto";
  card.classList.remove("stale");
}

/** Render every card, one style per task so input stays responsive between
 *  them. A newer pass (or leaving the grid) cancels this one. */
function renderGrid(): void {
  if (!state.source || state.view !== "grid") return;
  if (state.gridTimer) clearTimeout(state.gridTimer);
  state.gridDirty = false;
  const seq = ++state.gridSeq;
  const memo = new Map<string, Raster>();
  // The selected style first, then the rest in on-screen order.
  const order = [styleById(state.styleId), ...STYLES.filter((s) => s.id !== state.styleId)];
  const image = gridSetup(STYLES[0]);
  const ref = boxFit(
    state.sourceW, state.sourceH,
    image.setup.maxW && image.setup.maxW > 0 ? image.setup.maxW : null,
    image.setup.maxH && image.setup.maxH > 0 ? image.setup.maxH : null,
    image.setup.fit, false,
  );
  $("#gridHint").textContent = image.reduced
    ? `Previews at ${ref.canvasW} × ${ref.canvasH} px — your output is larger, so its grain is finer`
    : `${ref.canvasW} × ${ref.canvasH} px · click a style to open it`;
  let i = 0;
  const step = () => {
    if (seq !== state.gridSeq || state.view !== "grid") {
      if (seq === state.gridSeq) state.gridDirty = true; // left mid-pass
      return;
    }
    const style = order[i++];
    if (!style) return;
    const card = document.querySelector<HTMLElement>(`#variantView .card[data-style="${style.id}"]`);
    if (card) {
      try {
        let vals = valuesFor(style);
        // A thumbnail chart: small cells, no grid lines or symbols to drown it.
        if (style.renderer === "knit") vals = { ...vals, cell: 2, grid: "none", symbols: "none" };
        paintCard(card, renderStyle(style, vals, gridSetup(style).setup, memo).res);
      } catch {
        card.classList.remove("stale");
        const box = card.querySelector(".card-preview")!;
        box.innerHTML = '<div class="card-fail">couldn\'t render</div>';
      }
    }
    setTimeout(step, 0);
  };
  setTimeout(step, 0);
}

// ── canvas paint (recolour + hover isolation) ─────────────────────────────────
function nativeCanvas(): HTMLCanvasElement {
  return (state.native ??= document.createElement("canvas"));
}

/** The render's palette with swatch remaps applied — what export writes. */
function shownPalette(R: RenderState): RGB[] {
  return R.palette.map((c) => {
    const m = state.colorMap[rgbToHex(c)];
    return m ? parseColor(m) : c;
  });
}

/** Paint the render into the native canvas (always the clean image — export
 *  reads it), plus a hover-isolated copy while a swatch is hovered, then
 *  present whichever applies. */
function redrawCanvas(): void {
  const R = state.render;
  if (!R) return;
  paintRender(nativeCanvas(), R, shownPalette(R), null);
  showHover();
}

/** Present the render, isolating the hovered ink if there is one. */
function showHover(): void {
  const R = state.render;
  if (R && state.hoverIdx !== null) {
    paintRender((state.hoverView ??= document.createElement("canvas")), R, shownPalette(R), state.hoverIdx);
  }
  presentCanvas();
}

function paintRender(cv: HTMLCanvasElement, R: RenderState, oc: RGB[], hi: number | null): void {
  const { w: W, h: H, idx } = R;
  if (cv.width !== W || cv.height !== H) {
    cv.width = W;
    cv.height = H;
  }
  const n = idx.length;
  const out = new Uint8ClampedArray(n * 4);
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
  const ctx = cv.getContext("2d")!;
  ctx.putImageData(new ImageData(out, W, H), 0, 0);
  drawKnitSymbols(ctx, R, oc, hi);
}

/** Draw `src` into `dst` at `dev` device pixels per source pixel, faithfully:
 *   - whole-number enlargement → `dst` is a 1:1 copy, enlarged by CSS (pixelated);
 *   - fractional enlargement → "sharp bilinear": nearest-neighbour up to the next
 *     whole multiple, then a smooth reduction to the exact size. Plain fractional
 *     nearest-neighbour makes some dots 1px and some 2px, which reads as banding;
 *   - reduction → high-quality smoothing, i.e. the tone the eye sees at that
 *     size. Nearest-neighbour would drop pixels and show moiré that isn't in the
 *     exported file.
 *  Resizes `dst` and returns its CSS width. */
let scaleTmp: HTMLCanvasElement | null = null;
function paintScaled(dst: HTMLCanvasElement, src: HTMLCanvasElement, dev: number): number {
  const dpr = window.devicePixelRatio || 1;
  const whole = dev >= 1 - 1e-3 && Math.abs(dev - Math.round(dev)) < 1e-3;
  const w = whole ? src.width : Math.max(1, Math.round(src.width * dev));
  const h = whole ? src.height : Math.max(1, Math.round(src.height * dev));
  if (dst.width !== w || dst.height !== h) {
    dst.width = w;
    dst.height = h;
  }
  const ctx = dst.getContext("2d")!;
  ctx.clearRect(0, 0, w, h);
  if (whole) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(src, 0, 0);
    return (src.width * Math.round(dev)) / dpr;
  }
  let from: HTMLCanvasElement = src;
  if (dev > 1) {
    const m = Math.ceil(dev);
    const tmp = (scaleTmp ??= document.createElement("canvas"));
    tmp.width = src.width * m;
    tmp.height = src.height * m;
    const t = tmp.getContext("2d")!;
    t.imageSmoothingEnabled = false;
    t.drawImage(src, 0, 0, tmp.width, tmp.height);
    from = tmp;
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(from, 0, 0, w, h);
  return w / dpr;
}

/** Show the native render on #bigCanvas at the current zoom. */
function presentCanvas(): void {
  const src = state.hoverIdx !== null && state.hoverView ? state.hoverView : state.native;
  if (!src || !src.width || !src.height) return;
  const cv = $<HTMLCanvasElement>("#bigCanvas");
  const scale = displayScale(src.width, src.height);
  const cssW = paintScaled(cv, src, scale * (window.devicePixelRatio || 1));
  cv.style.width = `${cssW.toFixed(3)}px`;
  cv.style.height = "auto";
  $("#zoomVal").textContent = `${Math.round(scale * 100)}%`;
}

/** Whether the chart has symbols drawn over its cells (anti-aliased text, so
 *  the image is no longer a pure index map). */
function hasCellSymbols(R: RenderState): boolean {
  const k = R.knit;
  return !!k && k.symbolMode === "cells" && k.symbols.length > 0 && k.cell >= 7;
}

function drawKnitSymbols(ctx: CanvasRenderingContext2D, R: RenderState, palette: RGB[], hi: number | null): void {
  const k = R.knit;
  if (!k || !hasCellSymbols(R)) return;
  const fontSize = Math.max(6, Math.floor(k.cell * 0.68));
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.font = `700 ${fontSize}px "DM Mono", monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let y = 0; y < k.rows; y++) {
    const cy = y * k.cell + k.cell / 2;
    for (let x = 0; x < k.cols; x++) {
      const v = k.stitchIdx[y * k.cols + x];
      const symbol = k.symbols[v];
      if (!symbol || (hi !== null && v !== hi)) continue;
      const c = palette[v] ?? [0, 0, 0];
      const lum = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
      ctx.fillStyle = lum > 150 ? "#1c1712" : "#f8f1e1";
      ctx.fillText(symbol, x * k.cell + k.cell / 2, cy + 0.25);
    }
  }
  ctx.restore();
}

function buildPlateStrip(): void {
  const wrap = $("#plateStrip");
  const plates = state.render?.plates;
  wrap.innerHTML = "";
  wrap.hidden = !plates?.length;
  if (!plates?.length) return;
  plates.forEach((plate) => {
    const item = document.createElement("span");
    item.className = "plate-chip";
    const color = document.createElement("i");
    color.style.background = rgbToHex(plate.color);
    const label = document.createElement("b");
    label.textContent = `${plate.name} ${plate.angle}°`;
    item.append(color, label);
    wrap.append(item);
  });
  const meta = document.createElement("span");
  meta.className = "plate-meta";
  meta.textContent = `${state.render?.screen?.toUpperCase()} · ${state.render?.cell}px cell`;
  wrap.append(meta);
}

// ── palette swatches ──────────────────────────────────────────────────────────
function buildSwatches(): void {
  const R = state.render;
  const block = $("#paletteBlock");
  const wrap = $("#swatches");
  wrap.innerHTML = "";
  const swatchCount = R?.swatchCount ?? R?.palette.length ?? 0;
  if (!R || swatchCount > 32) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  $("#paletteCount").textContent = String(swatchCount);

  R.palette.slice(0, swatchCount).forEach((c, i) => {
    const hex = rgbToHex(c);
    const shown = state.colorMap[hex] || hex;
    const symbol = R.knit?.symbols[i];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "swatch" + (state.colorMap[hex] ? " remapped" : "");
    // Build via DOM (not innerHTML): `shown` is a colour value, kept out of markup.
    const chip = document.createElement("span");
    chip.className = "swatch-chip";
    chip.style.background = shown;
    if (symbol) {
      const sym = document.createElement("span");
      sym.className = "swatch-symbol";
      sym.textContent = symbol;
      chip.append(sym);
    }
    const label = document.createElement("span");
    label.className = "swatch-hex";
    label.textContent = symbol ? `${symbol} ${shown.toUpperCase()}` : shown.toUpperCase();
    btn.append(chip, label);
    btn.addEventListener("mouseenter", () => {
      state.hoverIdx = i;
      $$(".swatch").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      showHover();
    });
    btn.addEventListener("mouseleave", () => {
      state.hoverIdx = null;
      btn.classList.remove("active");
      showHover();
    });
    btn.addEventListener("click", () => openRecolor(hex, i));
    wrap.appendChild(btn);
  });
  updateResetButton();
}

/** The palette block's reset undoes whichever kind of edit is active: a
 *  remap of the result (non-palette-aware renderers) or a swatch edit that
 *  promoted the palette to exact inks (tone dither). */
function canRevertPalette(): boolean {
  return rendererKey() === "tone_dither" && state.paletteBeforeEdit !== null;
}

function updateResetButton(): void {
  const btn = $("#resetColors");
  const revert = canRevertPalette();
  btn.hidden = Object.keys(state.colorMap).length === 0 && !revert;
  btn.textContent = revert ? "revert" : "reset";
  btn.title = revert ? `Back to the ${state.paletteBeforeEdit} palette` : "Undo colour remaps";
}

function revertPalette(): void {
  const mode = state.paletteBeforeEdit;
  if (!mode) return;
  state.paletteBeforeEdit = null;
  knobBucket().palette = mode;
  const sel = document.getElementById("param-palette") as HTMLSelectElement | null;
  if (sel) sel.value = mode;
  updateKnobVisibility();
  updateCustomPaletteVisibility();
  scheduleRender();
}

function openRecolor(hex: string, index: number): void {
  const picker = $<HTMLInputElement>("#swatchPicker");
  picker.value = state.colorMap[hex] || hex;
  // The inks as they were when the picker opened: every input edits slot
  // `index` of this, not of the latest render (which may have merged inks).
  const base = state.render?.palette.map((c) => [...c] as RGB) ?? [];
  picker.oninput = () => {
    if (rendererKey() === "tone_dither" && state.render) {
      // Palette-aware renderers derive their mixing weights from the actual ink
      // colours. Promote the visible palette to an exact palette and recalculate;
      // repainting the old index map would make Knoll/Yliluoma/EMA untruthful.
      const from = String(valuesFor(styleById(state.styleId)).palette);
      if (from !== "custom") {
        state.paletteBeforeEdit = from;
        toast(`Now using exact inks — "revert" returns to ${from}`);
      }
      state.customPalette = base.map((c) => [...c] as RGB);
      state.customPalette[index] = parseColor(picker.value);
      knobBucket().palette = "custom";
      const paletteSelect = document.getElementById("param-palette") as HTMLSelectElement | null;
      if (paletteSelect) paletteSelect.value = "custom";
      syncCustomPaletteText();
      updateKnobVisibility();
      updateCustomPaletteVisibility();
      scheduleRender();
      return;
    }
    state.colorMap[hex] = picker.value;
    redrawCanvas();
    buildSwatches();
    scheduleExport();
  };
  picker.click();
}

function parseCustomPalette(text: string): RGB[] {
  const matches = text.match(/#[0-9a-fA-F]{6}\b/g) ?? [];
  const seen = new Set<string>();
  const colors: RGB[] = [];
  for (const value of matches.slice(0, 16)) {
    const key = value.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      colors.push(parseColor(key));
    }
  }
  return colors;
}

function syncCustomPaletteText(): void {
  $<HTMLTextAreaElement>("#customPaletteText").value = state.customPalette.map(rgbToHex).join(", ");
  $("#customPaletteStatus").textContent = `${state.customPalette.length} ink${state.customPalette.length === 1 ? "" : "s"}`;
}

function updateCustomPaletteVisibility(): void {
  const style = styleById(state.styleId);
  const palette = style.renderer ? null : (document.getElementById("param-palette") as HTMLSelectElement | null)?.value ?? knobBucket().palette;
  $("#customPaletteBlock").hidden = palette !== "custom";
  document.getElementById("knob-colors")?.classList.toggle("hidden", palette === "custom");
  $("#pinBlock").hidden = palette === "custom" || style.renderer === "halftone";
  $("#paletteHint").textContent = rendererKey() === "tone_dither"
    ? "Hover to isolate · click to edit and recalculate"
    : "Hover to isolate · click to remap the result";
}

function bindCustomPalette(): void {
  syncCustomPaletteText();
  $("#applyCustomPalette").addEventListener("click", () => {
    const colors = parseCustomPalette($<HTMLTextAreaElement>("#customPaletteText").value);
    if (colors.length < 2) {
      toast("Enter at least two unique six-digit hex colours");
      return;
    }
    state.customPalette = colors;
    knobBucket().palette = "custom";
    syncCustomPaletteText();
    scheduleRender();
  });
}

// ── colour pins (steer which colours generation picks) ─────────────────────────
// Unlike the swatch recolour above (which edits the *result*), pins are read
// BEFORE generation and constrain the adaptive palette: lock forces a colour in,
// boost favours a hue, exclude keeps one out. Applies to adaptive-palette modes.
const PIN_CYCLE: PinMode[] = ["lock", "boost", "exclude"];

function buildPins(): void {
  const tray = $("#pinTray");
  tray.innerHTML = "";
  state.pins.forEach((pin, i) => {
    const chip = document.createElement("span");
    chip.className = `pin pin-${pin.mode}`;
    const sw = document.createElement("span");
    sw.className = "pin-chip";
    sw.style.background = pin.hex;
    const hex = document.createElement("span");
    hex.className = "pin-hex";
    hex.textContent = pin.hex.toUpperCase();
    const mode = document.createElement("button");
    mode.type = "button";
    mode.className = "pin-mode";
    mode.textContent = pin.mode;
    mode.title = "Change: lock → boost → exclude";
    mode.addEventListener("click", () => cyclePin(i));
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pin-remove";
    rm.textContent = "✕";
    rm.title = "Remove pin";
    rm.addEventListener("click", () => removePin(i));
    chip.append(sw, hex, mode, rm);
    tray.appendChild(chip);
  });
  $("#pinClear").hidden = state.pins.length === 0;
  $("#pinEmpty").hidden = state.pins.length !== 0;
}

function addPin(hex: string): void {
  const norm = hex.toLowerCase();
  if (state.pins.some((p) => p.hex.toLowerCase() === norm)) {
    toast("That colour is already pinned");
    return;
  }
  state.pins.push({ hex: norm, mode: "lock" });
  buildPins();
  scheduleRender();
}

function cyclePin(i: number): void {
  const pin = state.pins[i];
  if (!pin) return;
  pin.mode = PIN_CYCLE[(PIN_CYCLE.indexOf(pin.mode) + 1) % PIN_CYCLE.length];
  buildPins();
  scheduleRender();
}

function removePin(i: number): void {
  state.pins.splice(i, 1);
  buildPins();
  scheduleRender();
}

function bindPins(): void {
  const picker = $<HTMLInputElement>("#pinPicker");
  $("#pinAdd").addEventListener("click", () => {
    // Seed with a current palette colour so the native eyedropper / tweak is handy.
    const seed = state.render?.palette[0];
    picker.value = seed ? rgbToHex(seed) : "#3a6ea5";
    picker.onchange = () => addPin(picker.value);
    picker.click();
  });
  $("#pinClear").addEventListener("click", () => {
    if (state.pins.length === 0) return;
    state.pins = [];
    buildPins();
    scheduleRender();
  });
}

// ── export (local, lossless, + Web Crypto checksum) ───────────────────────────
// Upscaled exports are capped so a big render × 8 can't allocate a canvas the
// browser refuses (or a multi-hundred-MB PNG); larger scales are disabled.
const EXPORT_SCALES = [1, 2, 4, 8];
const EXPORT_MAX_SIDE = 8192;
const EXPORT_MAX_PIXELS = 4096 * 4096;
// Both formats are lossless: a dither is exact dots, and lossy codecs (JPEG,
// lossy WebP) smear them into mud while rarely being smaller.
const EXPORT_FORMATS = [
  { id: "Png", ext: "png" },
  { id: "Webp", ext: "webp" },
] as const;
type ExportFormatId = (typeof EXPORT_FORMATS)[number]["id"];

interface Encoded {
  blob: Blob | null;
  /** Card subtitle when encoded, or why the format is unavailable. */
  note: string;
}

function canvasBlob(canvas: HTMLCanvasElement, mime: string, quality?: number): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, mime, quality));
}

function scaleFits(w: number, h: number, s: number): boolean {
  return Math.max(w, h) * s <= EXPORT_MAX_SIDE && w * h * s * s <= EXPORT_MAX_PIXELS;
}

/** The multiplier actually applied: the requested one, stepped down until it
 *  fits the caps. Knit charts already bake their scale into the cell size. */
function effectiveExportScale(): number {
  const R = state.render;
  if (!R || R.knit) return 1;
  let s = 1;
  for (const c of EXPORT_SCALES) if (c <= state.exportScale && scaleFits(R.w, R.h, c)) s = c;
  return s;
}

function bindExportScale(): void {
  $$("#scaleGroup button").forEach((b) =>
    b.addEventListener("click", () => {
      state.exportScale = parseInt(b.dataset.scale || "1", 10);
      updateScaleUI();
      scheduleExport();
      saveSessionSoon();
    }),
  );
}

function updateScaleUI(): void {
  const R = state.render;
  $("#scaleRow").hidden = !!R?.knit;
  const eff = effectiveExportScale();
  $$("#scaleGroup button").forEach((b) => {
    const s = parseInt(b.dataset.scale || "1", 10);
    const fits = !R || scaleFits(R.w, R.h, s);
    (b as HTMLButtonElement).disabled = !fits;
    b.title = fits ? "" : `Too large — over ${EXPORT_MAX_SIDE}px or ${EXPORT_MAX_PIXELS / 1e6 | 0} MP`;
    b.classList.toggle("active", s === eff);
  });
  $("#specExport").textContent = R ? `${R.w * eff} × ${R.h * eff}${eff > 1 ? ` · ${eff}×` : ""}` : "—";
}

/** The native render, nearest-neighbour upscaled by the export scale. */
function exportSource(): HTMLCanvasElement | null {
  const src = state.native;
  if (!src || !src.width) return null;
  const s = effectiveExportScale();
  if (s === 1) return src;
  const cv = (state.exportCanvas ??= document.createElement("canvas"));
  cv.width = src.width * s;
  cv.height = src.height * s;
  const ctx = cv.getContext("2d")!;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  return cv;
}

/** PNG straight from the index map at 1–8 bits per pixel — several times
 *  smaller than the canvas's 32-bit PNG. Falls back to the canvas when the
 *  image isn't a pure index map (knit symbols are anti-aliased text). */
async function encodePng(R: RenderState): Promise<Encoded> {
  const palette = shownPalette(R);
  let maxIdx = 0;
  for (let i = 0; i < R.idx.length; i++) if (R.idx[i] > maxIdx) maxIdx = R.idx[i];
  if (canEncodeIndexedPng() && palette.length <= 256 && maxIdx < palette.length && !hasCellSymbols(R)) {
    try {
      const blob = await encodeIndexedPng(R.idx, R.w, R.h, palette, effectiveExportScale());
      return { blob, note: "lossless · indexed" };
    } catch {
      // fall through to the canvas encoder
    }
  }
  const cv = exportSource();
  return { blob: cv ? await canvasBlob(cv, "image/png") : null, note: "lossless" };
}

/** Lossless WebP, where the browser's encoder offers it (Chrome does at
 *  quality 1; others may silently encode lossy or fall back to PNG). */
async function encodeWebp(): Promise<Encoded> {
  const cv = exportSource();
  const blob = cv ? await canvasBlob(cv, "image/webp", 1) : null;
  if (!blob || blob.type !== "image/webp") return { blob: null, note: "not in this browser" };
  const head = new Uint8Array(await blob.slice(0, 4096).arrayBuffer());
  if (!isLosslessWebp(head)) return { blob: null, note: "lossy only here" };
  return { blob, note: "lossless" };
}

/** Encoding is the slow part of a knob tweak at large export sizes, so wait
 *  for the knobs to settle. Until then the download links are disabled — a
 *  click must never download the previous render. */
function scheduleExport(): void {
  state.exportSeq++;
  EXPORT_FORMATS.forEach((f) => {
    $(`#export${f.id}`).classList.add("disabled", "pending");
    $(`#size${f.id}`).textContent = "…";
  });
  $("#stageExport").classList.add("disabled");
  $("#stageExportMeta").textContent = "…";
  if (state.exportTimer) clearTimeout(state.exportTimer);
  state.exportTimer = setTimeout(() => void updateExport(), 220);
}

async function updateExport(): Promise<void> {
  const R = state.render;
  if (!R || !state.native?.width) return;
  const seq = ++state.exportSeq;
  const [png, webp] = await Promise.all([encodePng(R), encodeWebp()]);
  if (seq !== state.exportSeq) return;
  const results: Record<ExportFormatId, Encoded> = { Png: png, Webp: webp };
  EXPORT_FORMATS.forEach((format) => {
    const { blob, note } = results[format.id];
    const link = $<HTMLAnchorElement>(`#export${format.id}`);
    const old = state.exportUrls[format.ext];
    if (old) URL.revokeObjectURL(old);
    delete state.exportUrls[format.ext];
    link.classList.remove("pending");
    $(`#note${format.id}`).textContent = note;
    if (!blob) {
      link.classList.add("disabled");
      link.removeAttribute("href");
      $(`#size${format.id}`).textContent = "—";
      return;
    }
    const url = URL.createObjectURL(blob);
    state.exportUrls[format.ext] = url;
    link.href = url;
    link.download = exportFilename(format.ext);
    link.classList.remove("disabled");
    $(`#size${format.id}`).textContent = formatBytes(blob.size);
  });
  // The stage's download button is the PNG — the one format every app opens.
  const stage = $<HTMLAnchorElement>("#stageExport");
  const s = effectiveExportScale();
  if (png.blob) {
    stage.href = state.exportUrls.png;
    stage.download = exportFilename("png");
    stage.classList.remove("disabled");
    $("#stageExportMeta").textContent = `${R.w * s} × ${R.h * s} · ${formatBytes(png.blob.size)}`;
    void setChecksum(png.blob);
  } else {
    stage.removeAttribute("href");
    $("#stageExportMeta").textContent = "unavailable";
  }
}

async function setChecksum(blob: Blob): Promise<void> {
  try {
    const buf = await blob.arrayBuffer();
    const hash = await crypto.subtle.digest("SHA-256", buf);
    const hex = Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    $("#exportPng").title = `SHA-256 ${hex}`;
  } catch {
    $("#exportPng").title = "";
  }
}

function exportFilename(ext = "png"): string {
  const base = state.sourceName.replace(/\.[^.]+$/, "") || "image";
  const s = effectiveExportScale();
  const dims = state.render ? `_${state.render.w}x${state.render.h}${s > 1 ? `@${s}x` : ""}` : "";
  return `${base}_${state.styleId}${dims}.${ext}`;
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
      const src = state.native;
      if (!src || !src.width) return;
      // Enlarged, step to the next whole multiple (crisp, uniform dots);
      // reduced, step by ×1.25 / ×0.8.
      const dpr = window.devicePixelRatio || 1;
      const fit = fitScale(src.width, src.height);
      const raw = displayScale(src.width, src.height) * dpr;
      const dev = Math.abs(raw - Math.round(raw)) < 1e-6 ? Math.round(raw) : raw;
      const up = parseInt(b.dataset.step || "0", 10) > 0;
      let next: number;
      if (up) next = dev >= 1 ? Math.floor(dev) + 1 : Math.min(1, dev * 1.25);
      else next = dev > 1 ? Math.ceil(dev) - 1 : dev * 0.8;
      state.zoomMode = "manual";
      // Beyond the slider's range so large renders can still reach 2×, 3× …
      state.zoomPct = Math.max(10, Math.min(3200, (next / dpr / fit) * 100));
      $<HTMLInputElement>("#zoomSlider").value = String(Math.round(state.zoomPct));
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

/** CSS px per image px for the current zoom. */
function fitScale(nw: number, nh: number): number {
  const lb = $(".lightbox");
  const availW = Math.max(80, lb.clientWidth - 40);
  const availH = Math.max(80, lb.clientHeight - 48);
  return Math.min(availW / nw, availH / nh);
}

function displayScale(nw: number, nh: number): number {
  const fit = fitScale(nw, nh);
  return state.zoomMode === "fit" ? fit : fit * (state.zoomPct / 100);
}

function applyZoom(): void {
  presentCanvas();
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
  const restored = restoreSession();
  buildStyleSelect();
  bindSource();
  bindOutput();
  updateOutputPanel();
  bindZoom();
  bindPan();
  bindView();
  $("#resetColors").addEventListener("click", () => {
    if (canRevertPalette()) {
      revertPalette();
      return;
    }
    state.colorMap = {};
    state.hoverIdx = null;
    redrawCanvas();
    buildSwatches();
    scheduleExport();
  });
  bindExportScale();
  bindPins();
  bindCustomPalette();
  buildPins();
  updateStyleUI();
  $("#copyGlyphs").addEventListener("click", () => copyGlyphText());
  showStudio(false);
  if (restored) {
    void loadSourceFile().then((file) => {
      if (file && !state.source) loadFile(file, restored);
    });
  }

  let resizeRAF = 0;
  window.addEventListener("resize", () => {
    if (state.view === "grid") scheduleGrid(); // thumbnails are painted at card size
    if (state.zoomMode !== "fit") return;
    cancelAnimationFrame(resizeRAF);
    resizeRAF = requestAnimationFrame(applyZoom);
  });
}

init();
