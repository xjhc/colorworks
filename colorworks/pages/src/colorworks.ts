/**
 * Colorworks — browser-safe N-colour dither core.
 *
 * A dependency-free TypeScript port of the Python `render_tone_dither` pipeline
 * (`colorworks/algorithms/dither.py` + `image_ops.py` + `renderers/bayer.py`).
 *
 * Contract (see COLORWORKS_PLAN.md):
 *   - No DOM, no fetch, no Node-only APIs — safe to import in vitest and workers.
 *   - Input is an RGBA raster (canvas ImageData shape) + options.
 *   - Output is `{ indices, palette }`: a per-pixel palette index map plus the
 *     deduped, luma-sorted palette. The UI paints indices → palette, which makes
 *     live palette-recolour a repaint (no re-dither).
 *
 * Fidelity notes vs. Python:
 *   - Adaptive palette is deterministic k-means++ on a ≤96px subsample, sorted by
 *     luma. The PRNG is a seeded mulberry32, so results are deterministic but NOT
 *     bit-identical to NumPy's PCG64 (visual parity, by design).
 *   - Floyd–Steinberg is a fresh colour-space implementation (no PIL): in-palette
 *     output that tracks mean tone, not PIL median-cut parity.
 *   - Flow uses the RAW grayscale for its threshold mask and the tone-remapped RGB
 *     for palette assignment — matching the Python split.
 */
import { BLUE_NOISE_TILES_B64 } from "./blue-noise-tiles";

export type RGB = [number, number, number];
export type PaletteMode = "grayscale" | "adaptive" | "duotone";
export type DitherMethod =
  | "bayer"
  | "blue_noise"
  | "floyd_steinberg"
  | "flow"
  | "flat";

/** An RGBA raster, exactly the shape of `CanvasRenderingContext2D.getImageData`. */
export interface Raster {
  width: number;
  height: number;
  /** RGBA, length = width * height * 4. */
  data: Uint8ClampedArray;
}

export interface RenderParams {
  matrixSize?: number; // bayer: 2 | 4 | 8 | 16
  noiseSize?: number; // blue_noise: 16 | 32 | 64 | 128
  frequency?: number; // flow: wave density
  warp?: number; // flow: flow strength
  angleDeg?: number; // flow: base angle
  detail?: number; // flow: warp-field blur radius
}

export interface RenderOptions {
  colors?: number; // 2..8
  palette?: PaletteMode;
  method?: DitherMethod;
  contrast?: number;
  midpoint?: number;
  inkColor?: string;
  paperColor?: string;
  params?: RenderParams;
  seed?: number;
}

export interface RenderResult {
  /** Per-pixel palette index, row-major, length = width * height. */
  indices: Uint16Array;
  palette: RGB[];
  width: number;
  height: number;
}

// ── Colour helpers ──────────────────────────────────────────────────────────
const clampByte = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

export function rgbToHex([r, g, b]: RGB): string {
  return (
    "#" +
    [r, g, b].map((v) => clampByte(v).toString(16).padStart(2, "0")).join("")
  );
}

export function rgbToCss([r, g, b]: RGB): string {
  return `rgb(${clampByte(r)}, ${clampByte(g)}, ${clampByte(b)})`;
}

export function parseColor(hex: string): RGB {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const luma = (r: number, g: number, b: number): number =>
  0.299 * r + 0.587 * g + 0.114 * b;

function lerpRgb(a: RGB, b: RGB, t: number): RGB {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

// ── Deterministic PRNG (mulberry32) ───────────────────────────────────────────
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Raster → float buffers ────────────────────────────────────────────────────
export function rasterToRgb01(r: Raster): Float32Array {
  const n = r.width * r.height;
  const out = new Float32Array(n * 3);
  const d = r.data;
  for (let i = 0; i < n; i++) {
    out[i * 3] = d[i * 4] / 255;
    out[i * 3 + 1] = d[i * 4 + 1] / 255;
    out[i * 3 + 2] = d[i * 4 + 2] / 255;
  }
  return out;
}

export function rasterToGray(r: Raster): Float32Array {
  const n = r.width * r.height;
  const out = new Float32Array(n);
  const d = r.data;
  for (let i = 0; i < n; i++) {
    out[i] = luma(d[i * 4], d[i * 4 + 1], d[i * 4 + 2]) / 255;
  }
  return out;
}

// ── Tone control ──────────────────────────────────────────────────────────────
/** Hue-preserving tone remap: rescale each pixel's luminance, keep its chroma. */
export function applyTone(
  rgb01: Float32Array,
  gray: Float32Array,
  contrast: number,
  midpoint: number,
): void {
  if (Math.abs(contrast - 1) <= 1e-6 && Math.abs(midpoint - 0.5) <= 1e-6) return;
  for (let i = 0; i < gray.length; i++) {
    const g = gray[i];
    let remapped = (g - midpoint) * contrast + 0.5;
    remapped = remapped < 0 ? 0 : remapped > 1 ? 1 : remapped;
    const scale = remapped / Math.max(g, 1e-4);
    const j = i * 3;
    rgb01[j] = Math.min(1, rgb01[j] * scale);
    rgb01[j + 1] = Math.min(1, rgb01[j + 1] * scale);
    rgb01[j + 2] = Math.min(1, rgb01[j + 2] * scale);
  }
}

// ── Separable Gaussian blur (edge-replicated), mirrors image_ops.gaussian_blur ──
function gaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const radius = Math.trunc(Math.max(1, 3 * sigma));
  const kernel = new Float32Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    ksum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const tmp = new Float32Array(w * h);
  // horizontal
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let xx = x + k;
        xx = xx < 0 ? 0 : xx >= w ? w - 1 : xx;
        acc += src[row + xx] * kernel[k + radius];
      }
      tmp[row + x] = acc;
    }
  }
  // vertical
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let yy = y + k;
        yy = yy < 0 ? 0 : yy >= h ? h - 1 : yy;
        acc += tmp[yy * w + x] * kernel[k + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// ── Palettes ──────────────────────────────────────────────────────────────────
/**
 * k-means++ palette extraction on a ≤96px subsample (mirrors kmeans_palette).
 * Works in 0..255 space like the Python original. Deterministic given `seed`.
 */
export function kmeansPalette(raster: Raster, colors: number, seed = 0, iters = 16): RGB[] {
  colors = Math.max(2, Math.floor(colors));
  const { width: w, height: h, data } = raster;

  // Subsample to a max dimension of 96 (nearest-neighbour, DOM-free).
  const maxDim = Math.max(w, h);
  const s = maxDim > 96 ? 96 / maxDim : 1;
  const sw = Math.max(1, Math.round(w * s));
  const sh = Math.max(1, Math.round(h * s));
  const m = sw * sh;
  const X = new Float32Array(m * 3);
  for (let ty = 0; ty < sh; ty++) {
    const syRow = Math.min(h - 1, Math.floor(ty / s)) * w;
    for (let tx = 0; tx < sw; tx++) {
      const sx = Math.min(w - 1, Math.floor(tx / s));
      const src = (syRow + sx) * 4;
      const dst = (ty * sw + tx) * 3;
      X[dst] = data[src];
      X[dst + 1] = data[src + 1];
      X[dst + 2] = data[src + 2];
    }
  }
  if (m === 0) return buildTonePalette(raster, colors, "grayscale");

  const rng = mulberry32(seed);
  const randInt = (n: number) => Math.min(n - 1, Math.floor(rng() * n));

  const sqDist = (i: number, cr: number, cg: number, cb: number): number => {
    const dr = X[i * 3] - cr;
    const dg = X[i * 3 + 1] - cg;
    const db = X[i * 3 + 2] - cb;
    return dr * dr + dg * dg + db * db;
  };

  // k-means++ seeding.
  const C = new Float32Array(colors * 3);
  {
    const f = randInt(m);
    C[0] = X[f * 3];
    C[1] = X[f * 3 + 1];
    C[2] = X[f * 3 + 2];
  }
  const d2 = new Float32Array(m);
  for (let c = 1; c < colors; c++) {
    let total = 0;
    for (let i = 0; i < m; i++) {
      let best = Infinity;
      for (let j = 0; j < c; j++) {
        const dd = sqDist(i, C[j * 3], C[j * 3 + 1], C[j * 3 + 2]);
        if (dd < best) best = dd;
      }
      d2[i] = best;
      total += best;
    }
    let pick = m - 1;
    if (total > 1e-9) {
      let target = rng() * total;
      for (let i = 0; i < m; i++) {
        target -= d2[i];
        if (target <= 0) {
          pick = i;
          break;
        }
      }
    } else {
      pick = randInt(m);
    }
    C[c * 3] = X[pick * 3];
    C[c * 3 + 1] = X[pick * 3 + 1];
    C[c * 3 + 2] = X[pick * 3 + 2];
  }

  // Lloyd iterations.
  const labels = new Int32Array(m);
  const sums = new Float64Array(colors * 3);
  const counts = new Int32Array(colors);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < m; i++) {
      let best = Infinity;
      let bj = 0;
      for (let j = 0; j < colors; j++) {
        const dd = sqDist(i, C[j * 3], C[j * 3 + 1], C[j * 3 + 2]);
        if (dd < best) {
          best = dd;
          bj = j;
        }
      }
      labels[i] = bj;
    }
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < m; i++) {
      const j = labels[i];
      counts[j]++;
      sums[j * 3] += X[i * 3];
      sums[j * 3 + 1] += X[i * 3 + 1];
      sums[j * 3 + 2] += X[i * 3 + 2];
    }
    for (let j = 0; j < colors; j++) {
      if (counts[j] > 0) {
        C[j * 3] = sums[j * 3] / counts[j];
        C[j * 3 + 1] = sums[j * 3 + 1] / counts[j];
        C[j * 3 + 2] = sums[j * 3 + 2] / counts[j];
      } else {
        // Re-seed a dead cluster on the worst-fit pixel.
        let far = 0;
        let worst = -1;
        for (let i = 0; i < m; i++) {
          let best = Infinity;
          for (let k = 0; k < colors; k++) {
            const dd = sqDist(i, C[k * 3], C[k * 3 + 1], C[k * 3 + 2]);
            if (dd < best) best = dd;
          }
          if (best > worst) {
            worst = best;
            far = i;
          }
        }
        C[j * 3] = X[far * 3];
        C[j * 3 + 1] = X[far * 3 + 1];
        C[j * 3 + 2] = X[far * 3 + 2];
      }
    }
  }

  const swatches: RGB[] = [];
  for (let j = 0; j < colors; j++) {
    swatches.push([
      Math.round(C[j * 3]),
      Math.round(C[j * 3 + 1]),
      Math.round(C[j * 3 + 2]),
    ]);
  }
  swatches.sort((a, b) => luma(a[0], a[1], a[2]) - luma(b[0], b[1], b[2]));
  return swatches.slice(0, colors);
}

export function buildTonePalette(
  raster: Raster,
  colors: number,
  mode: PaletteMode = "grayscale",
  inkColor = "#161616",
  paperColor = "#f4ebd9",
  seed = 0,
): RGB[] {
  colors = Math.max(2, Math.floor(colors));
  if (mode === "grayscale") {
    return Array.from({ length: colors }, (_, i) =>
      lerpRgb([0, 0, 0], [255, 255, 255], i / (colors - 1)),
    );
  }
  if (mode === "duotone") {
    const ink = parseColor(inkColor);
    const paper = parseColor(paperColor);
    return Array.from({ length: colors }, (_, i) =>
      lerpRgb(ink, paper, i / (colors - 1)),
    );
  }
  return kmeansPalette(raster, colors, seed);
}

// ── Threshold masks ───────────────────────────────────────────────────────────
/** Normalized Bayer threshold matrix in [0,1), size ∈ {2,4,8,16}. */
export function bayerMatrix(size: number): number[][] {
  if (![2, 4, 8, 16].includes(size)) size = 8;
  let m = [
    [0, 2],
    [3, 1],
  ];
  let cur = 2;
  while (cur < size) {
    const n = m.length;
    const next: number[][] = Array.from({ length: n * 2 }, () => new Array(n * 2).fill(0));
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const v = m[y][x];
        next[y][x] = 4 * v + 0;
        next[y][x + n] = 4 * v + 2;
        next[y + n][x] = 4 * v + 3;
        next[y + n][x + n] = 4 * v + 1;
      }
    }
    m = next;
    cur *= 2;
  }
  const denom = size * size;
  return m.map((row) => row.map((v) => (v + 0.5) / denom));
}

function bayerThresholdMap(w: number, h: number, matrixSize: number): Float32Array {
  const matrix = bayerMatrix(matrixSize);
  const n = matrix.length;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const mrow = matrix[y % n];
    for (let x = 0; x < w; x++) out[y * w + x] = mrow[x % n];
  }
  return out;
}

const _blueNoiseCache = new Map<number, Float32Array>();

function blueNoiseTile(size: number): { tile: Float32Array; size: number } {
  if (![16, 32, 64, 128].includes(size)) size = 64;
  let tile = _blueNoiseCache.get(size);
  if (!tile) {
    const b64 = BLUE_NOISE_TILES_B64[size];
    if (!b64) throw new Error(`no embedded blue-noise tile for size ${size}`);
    const bin = atob(b64);
    const count = bin.length >> 1;
    tile = new Float32Array(count);
    const denom = size * size;
    for (let i = 0; i < count; i++) {
      const rank = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); // little-endian
      tile[i] = (rank + 0.5) / denom;
    }
    _blueNoiseCache.set(size, tile);
  }
  return { tile, size };
}

function blueNoiseThresholdMap(w: number, h: number, size: number): Float32Array {
  const { tile, size: n } = blueNoiseTile(size);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ty = y % n;
    for (let x = 0; x < w; x++) out[y * w + x] = tile[ty * n + (x % n)];
  }
  return out;
}

/**
 * Structure-aware "flow" field (mirrors flow_threshold_map). Uses RAW gray,
 * domain-warped by its own blurred luminance.
 */
export function flowThresholdMap(
  gray: Float32Array,
  w: number,
  h: number,
  frequency = 6,
  warp = 5,
  angleDeg = 45,
  detail = 2.5,
): Float32Array {
  const g = gaussianBlur(gray, w, h, Math.max(0.5, detail));
  const theta = (angleDeg * Math.PI) / 180;
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const f = frequency / 100;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const carrier = (x * ct + y * st) * f;
      const phase = carrier + warp * g[i];
      out[i] = (Math.sin(2 * Math.PI * phase) + 1) / 2;
    }
  }
  return out;
}

// ── Dither to palette (colour space) ──────────────────────────────────────────
/**
 * Dither `rgb01` to `palette` in colour space (mirrors dither_to_palette).
 * For each pixel: find the two nearest palette colours; with a mask, blend
 * between them by how far the pixel lies toward the second. With `mask=null`
 * this is plain nearest-colour assignment (the flat-poster path). Returns
 * per-pixel palette indices.
 */
export function ditherToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  mask: Float32Array | null,
): Uint16Array {
  const n = palette.length;
  const pal = new Float32Array(n * 3);
  for (let j = 0; j < n; j++) {
    pal[j * 3] = palette[j][0] / 255;
    pal[j * 3 + 1] = palette[j][1] / 255;
    pal[j * 3 + 2] = palette[j][2] / 255;
  }
  const out = new Uint16Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgb01[i * 3];
    const g = rgb01[i * 3 + 1];
    const b = rgb01[i * 3 + 2];
    // two nearest palette colours
    let i0 = 0;
    let i1 = 0;
    let d0 = Infinity;
    let d1 = Infinity;
    for (let j = 0; j < n; j++) {
      const dr = r - pal[j * 3];
      const dg = g - pal[j * 3 + 1];
      const db = b - pal[j * 3 + 2];
      const dd = dr * dr + dg * dg + db * db;
      if (dd < d0) {
        d1 = d0;
        i1 = i0;
        d0 = dd;
        i0 = j;
      } else if (dd < d1) {
        d1 = dd;
        i1 = j;
      }
    }
    if (mask === null) {
      out[i] = i0;
      continue;
    }
    const p0r = pal[i0 * 3];
    const p0g = pal[i0 * 3 + 1];
    const p0b = pal[i0 * 3 + 2];
    const dirr = pal[i1 * 3] - p0r;
    const dirg = pal[i1 * 3 + 1] - p0g;
    const dirb = pal[i1 * 3 + 2] - p0b;
    const denom = dirr * dirr + dirg * dirg + dirb * dirb + 1e-6;
    let t = ((r - p0r) * dirr + (g - p0g) * dirg + (b - p0b) * dirb) / denom;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    out[i] = t > mask[i] ? i1 : i0;
  }
  return out;
}

/**
 * Floyd–Steinberg error diffusion onto `palette` in colour space (a fresh port,
 * since PIL is unavailable). Output is guaranteed in-palette. Returns indices.
 */
export function fsToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
): Uint16Array {
  const n = palette.length;
  const pal = new Float32Array(n * 3);
  for (let j = 0; j < n; j++) {
    pal[j * 3] = palette[j][0] / 255;
    pal[j * 3 + 1] = palette[j][1] / 255;
    pal[j * 3 + 2] = palette[j][2] / 255;
  }
  const buf = Float32Array.from(rgb01); // mutable working copy
  const out = new Uint16Array(w * h);

  const diffuse = (idx: number, er: number, eg: number, eb: number, f: number) => {
    buf[idx * 3] += er * f;
    buf[idx * 3 + 1] += eg * f;
    buf[idx * 3 + 2] += eb * f;
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = buf[i * 3];
      const g = buf[i * 3 + 1];
      const b = buf[i * 3 + 2];
      let bj = 0;
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        const dr = r - pal[j * 3];
        const dg = g - pal[j * 3 + 1];
        const db = b - pal[j * 3 + 2];
        const dd = dr * dr + dg * dg + db * db;
        if (dd < best) {
          best = dd;
          bj = j;
        }
      }
      out[i] = bj;
      const er = r - pal[bj * 3];
      const eg = g - pal[bj * 3 + 1];
      const eb = b - pal[bj * 3 + 2];
      if (x + 1 < w) diffuse(i + 1, er, eg, eb, 7 / 16);
      if (y + 1 < h) {
        if (x > 0) diffuse(i + w - 1, er, eg, eb, 3 / 16);
        diffuse(i + w, er, eg, eb, 5 / 16);
        if (x + 1 < w) diffuse(i + w + 1, er, eg, eb, 1 / 16);
      }
    }
  }
  return out;
}

// ── Dedupe + reindex ──────────────────────────────────────────────────────────
/**
 * Collapse duplicate palette swatches and reindex the cell map accordingly.
 * Preserves first-occurrence order (palettes arrive luma-sorted, so this keeps
 * them sorted). Returns a new palette + a remapped index buffer.
 */
export function dedupePalette(
  indices: Uint16Array,
  palette: RGB[],
): { indices: Uint16Array; palette: RGB[] } {
  const seen = new Map<string, number>();
  const remap = new Int32Array(palette.length);
  const out: RGB[] = [];
  for (let i = 0; i < palette.length; i++) {
    const key = rgbToHex(palette[i]);
    let mapped = seen.get(key);
    if (mapped === undefined) {
      mapped = out.length;
      seen.set(key, mapped);
      out.push(palette[i]);
    }
    remap[i] = mapped;
  }
  if (out.length === palette.length) return { indices, palette };
  const reindexed = new Uint16Array(indices.length);
  for (let i = 0; i < indices.length; i++) reindexed[i] = remap[indices[i]];
  return { indices: reindexed, palette: out };
}

// ── Top-level render ──────────────────────────────────────────────────────────
const METHODS: DitherMethod[] = ["bayer", "blue_noise", "floyd_steinberg", "flow", "flat"];
const MODES: PaletteMode[] = ["grayscale", "adaptive", "duotone"];

/**
 * Render a raster as an N-colour dither — the heart of the target aesthetic.
 * Returns `{ indices, palette }` (deduped, luma-sorted); paint with
 * `palette[indices[i]]`.
 */
export function renderToneDither(raster: Raster, opts: RenderOptions = {}): RenderResult {
  const { width: w, height: h } = raster;
  const colors = Math.max(2, Math.min(8, Math.floor(opts.colors ?? 4)));
  const method: DitherMethod = METHODS.includes(opts.method as DitherMethod)
    ? (opts.method as DitherMethod)
    : "floyd_steinberg";
  const mode: PaletteMode = MODES.includes(opts.palette as PaletteMode)
    ? (opts.palette as PaletteMode)
    : "grayscale";
  const contrast = opts.contrast ?? 1;
  const midpoint = opts.midpoint ?? 0.5;
  const seed = opts.seed ?? 0;
  const p = opts.params ?? {};

  const palette = buildTonePalette(
    raster,
    colors,
    mode,
    opts.inkColor ?? "#161616",
    opts.paperColor ?? "#f4ebd9",
    seed,
  );

  const rgb01 = rasterToRgb01(raster);
  const gray = rasterToGray(raster);
  applyTone(rgb01, gray, contrast, midpoint);

  let indices: Uint16Array;
  if (method === "floyd_steinberg") {
    indices = fsToPalette(rgb01, w, h, palette);
  } else {
    let mask: Float32Array | null;
    if (method === "flat") {
      mask = null;
    } else if (method === "flow") {
      mask = flowThresholdMap(
        gray,
        w,
        h,
        p.frequency ?? 6,
        p.warp ?? 5,
        p.angleDeg ?? 45,
        p.detail ?? 2.5,
      );
    } else if (method === "blue_noise") {
      mask = blueNoiseThresholdMap(w, h, p.noiseSize ?? 64);
    } else {
      mask = bayerThresholdMap(w, h, p.matrixSize ?? 8);
    }
    indices = ditherToPalette(rgb01, w, h, palette, mask);
  }

  const deduped = dedupePalette(indices, palette);
  return { indices: deduped.indices, palette: deduped.palette, width: w, height: h };
}

/** Paint a render result into an RGBA buffer (e.g. for an ImageData). */
export function paintIndices(result: RenderResult, palette?: RGB[]): Uint8ClampedArray {
  const pal = palette ?? result.palette;
  const { indices, width, height } = result;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < indices.length; i++) {
    const c = pal[indices[i]] ?? [0, 0, 0];
    out[i * 4] = c[0];
    out[i * 4 + 1] = c[1];
    out[i * 4 + 2] = c[2];
    out[i * 4 + 3] = 255;
  }
  return out;
}
