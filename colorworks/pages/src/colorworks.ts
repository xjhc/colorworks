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
 * Colour fidelity:
 *   - All palette generation and colour matching happen in OKLab, a near-
 *     perceptually-uniform space. "Nearest colour" therefore means *perceptually*
 *     nearest, not sRGB-Euclidean nearest (which over-weights green and mis-ranks
 *     dark colours) — the single biggest lever on colour quality with a small
 *     palette.
 *   - Adaptive palette is deterministic median-cut (perceptual variance splitting)
 *     refined by k-means, both in OKLab, on a ≤96px subsample, sorted by luma.
 *   - Error diffusion (Floyd–Steinberg & the kernel pack) chooses the perceptually
 *     nearest swatch in OKLab but diffuses the residual in sRGB, so total tone is
 *     conserved like the Python original (no PIL median-cut parity).
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
  | "atkinson"
  | "jarvis"
  | "stucki"
  | "burkes"
  | "sierra"
  | "yliluoma"
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

// ── OKLab perceptual colour space ──────────────────────────────────────────────
// sRGB↔OKLab (Björn Ottosson). Matching and clustering happen here so distances
// are perceptual. sRGB-Euclidean over-weights green and mis-ranks dark colours;
// OKLab is near-uniform, so the nearest swatch is the one the eye agrees with.
const srgbToLinearChannel = (c: number): number =>
  c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const linearToSrgbChannel = (c: number): number =>
  c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;

/** sRGB (0..1) → OKLab [L, a, b] written into `out` at offset `o`. */
function srgbToOklabInto(r: number, g: number, b: number, out: Float32Array, o: number): void {
  const lr = srgbToLinearChannel(r);
  const lg = srgbToLinearChannel(g);
  const lb = srgbToLinearChannel(b);
  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;
  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);
  out[o] = 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_;
  out[o + 1] = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_;
  out[o + 2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_;
}

/** sRGB RGB (0..255) → OKLab [L, a, b]. */
export function srgbToOklab([r, g, b]: RGB): [number, number, number] {
  const out = new Float32Array(3);
  srgbToOklabInto(r / 255, g / 255, b / 255, out, 0);
  return [out[0], out[1], out[2]];
}

/** OKLab [L, a, b] → sRGB RGB (0..255, gamut-clamped). */
export function oklabToRgb(L: number, A: number, B: number): RGB {
  const l_ = L + 0.3963377774 * A + 0.2158037573 * B;
  const m_ = L - 0.1055613458 * A - 0.0638541728 * B;
  const s_ = L - 0.0894841775 * A - 1.291485548 * B;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const b = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
  return [
    clampByte(linearToSrgbChannel(r) * 255),
    clampByte(linearToSrgbChannel(g) * 255),
    clampByte(linearToSrgbChannel(b) * 255),
  ];
}

/** Pack a palette (RGB 0..255) into a flat OKLab Float32Array (length n*3). */
function paletteToOklab(palette: RGB[]): Float32Array {
  const out = new Float32Array(palette.length * 3);
  for (let j = 0; j < palette.length; j++) {
    srgbToOklabInto(palette[j][0] / 255, palette[j][1] / 255, palette[j][2] / 255, out, j * 3);
  }
  return out;
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
 * Adaptive palette extraction in OKLab on a ≤96px subsample. Seeds with
 * perceptual median-cut (split the box with the largest sum-of-squared-error
 * along its highest-variance axis at the median), then refines with k-means —
 * the ubitux-recommended pipeline. Working in OKLab keeps the swatches
 * perceptually balanced (median-cut alone in RGB spends slots on near-identical
 * dominant colours and desaturates), and the k-means pass closes median-cut's
 * residual gap. Deterministic; `seed` is retained for API compatibility but no
 * longer affects the result (the method is RNG-free).
 *
 * Name kept as `kmeansPalette` for callers (`buildTonePalette`, `repixel`).
 */
export function kmeansPalette(raster: Raster, colors: number, seed = 0, iters = 16): RGB[] {
  void seed;
  colors = Math.max(2, Math.floor(colors));
  const { width: w, height: h, data } = raster;

  // Subsample to a max dimension of 96 (nearest-neighbour, DOM-free), → OKLab.
  const maxDim = Math.max(w, h);
  const s = maxDim > 96 ? 96 / maxDim : 1;
  const sw = Math.max(1, Math.round(w * s));
  const sh = Math.max(1, Math.round(h * s));
  const m = sw * sh;
  if (m === 0) return buildTonePalette(raster, colors, "grayscale");

  const P = new Float32Array(m * 3); // OKLab points
  for (let ty = 0; ty < sh; ty++) {
    const syRow = Math.min(h - 1, Math.floor(ty / s)) * w;
    for (let tx = 0; tx < sw; tx++) {
      const sx = Math.min(w - 1, Math.floor(tx / s));
      const src = (syRow + sx) * 4;
      srgbToOklabInto(data[src] / 255, data[src + 1] / 255, data[src + 2] / 255, P, (ty * sw + tx) * 3);
    }
  }

  // Mean + per-axis variance (×count = SSE) of the point slice idx[lo, hi).
  const idx = new Int32Array(m);
  for (let i = 0; i < m; i++) idx[i] = i;
  const stats = (lo: number, hi: number) => {
    let mL = 0;
    let mA = 0;
    let mB = 0;
    for (let i = lo; i < hi; i++) {
      const p = idx[i] * 3;
      mL += P[p];
      mA += P[p + 1];
      mB += P[p + 2];
    }
    const cnt = hi - lo;
    mL /= cnt;
    mA /= cnt;
    mB /= cnt;
    let vL = 0;
    let vA = 0;
    let vB = 0;
    for (let i = lo; i < hi; i++) {
      const p = idx[i] * 3;
      const dL = P[p] - mL;
      const dA = P[p + 1] - mA;
      const dB = P[p + 2] - mB;
      vL += dL * dL;
      vA += dA * dA;
      vB += dB * dB;
    }
    return { mL, mA, mB, vL, vA, vB, sse: vL + vA + vB };
  };

  // ── Median-cut seeding (perceptual variance splitting) ──────────────────────
  const boxes: Array<{ lo: number; hi: number }> = [{ lo: 0, hi: m }];
  while (boxes.length < colors) {
    let bi = -1;
    let bestSse = -1;
    for (let b = 0; b < boxes.length; b++) {
      if (boxes[b].hi - boxes[b].lo <= 1) continue;
      const sse = stats(boxes[b].lo, boxes[b].hi).sse;
      if (sse > bestSse) {
        bestSse = sse;
        bi = b;
      }
    }
    if (bi < 0) break; // every box is a single point — can't split further
    const box = boxes[bi];
    const st = stats(box.lo, box.hi);
    const axis = st.vL >= st.vA && st.vL >= st.vB ? 0 : st.vA >= st.vB ? 1 : 2;
    const slice = Array.from(idx.subarray(box.lo, box.hi));
    slice.sort((p, q) => P[p * 3 + axis] - P[q * 3 + axis]);
    for (let i = 0; i < slice.length; i++) idx[box.lo + i] = slice[i];
    const mid = box.lo + (slice.length >> 1);
    boxes[bi] = { lo: box.lo, hi: mid };
    boxes.push({ lo: mid, hi: box.hi });
  }

  const k = boxes.length;
  const C = new Float32Array(k * 3);
  for (let b = 0; b < k; b++) {
    const st = stats(boxes[b].lo, boxes[b].hi);
    C[b * 3] = st.mL;
    C[b * 3 + 1] = st.mA;
    C[b * 3 + 2] = st.mB;
  }

  // ── k-means refinement (Lloyd iterations in OKLab) ──────────────────────────
  const sq = (i: number, cL: number, cA: number, cB: number): number => {
    const p = i * 3;
    const dL = P[p] - cL;
    const dA = P[p + 1] - cA;
    const dB = P[p + 2] - cB;
    return dL * dL + dA * dA + dB * dB;
  };
  const labels = new Int32Array(m);
  const sums = new Float64Array(k * 3);
  const counts = new Int32Array(k);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < m; i++) {
      let best = Infinity;
      let bj = 0;
      for (let j = 0; j < k; j++) {
        const dd = sq(i, C[j * 3], C[j * 3 + 1], C[j * 3 + 2]);
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
      sums[j * 3] += P[i * 3];
      sums[j * 3 + 1] += P[i * 3 + 1];
      sums[j * 3 + 2] += P[i * 3 + 2];
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        C[j * 3] = sums[j * 3] / counts[j];
        C[j * 3 + 1] = sums[j * 3 + 1] / counts[j];
        C[j * 3 + 2] = sums[j * 3 + 2] / counts[j];
      } else {
        // Re-seed a dead cluster on the worst-fit point.
        let far = 0;
        let worst = -1;
        for (let i = 0; i < m; i++) {
          let best = Infinity;
          for (let q = 0; q < k; q++) {
            const dd = sq(i, C[q * 3], C[q * 3 + 1], C[q * 3 + 2]);
            if (dd < best) best = dd;
          }
          if (best > worst) {
            worst = best;
            far = i;
          }
        }
        C[j * 3] = P[far * 3];
        C[j * 3 + 1] = P[far * 3 + 1];
        C[j * 3 + 2] = P[far * 3 + 2];
      }
    }
  }

  const swatches: RGB[] = [];
  for (let j = 0; j < k; j++) swatches.push(oklabToRgb(C[j * 3], C[j * 3 + 1], C[j * 3 + 2]));
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

// ── Dither to palette (perceptual / OKLab) ─────────────────────────────────────
/**
 * Dither `rgb01` to `palette` in OKLab (mirrors dither_to_palette, now
 * perceptual). For each pixel: find the two perceptually nearest palette
 * colours; with a mask, blend between them by how far the pixel projects toward
 * the second along the OKLab line joining them. With `mask=null` this is plain
 * nearest-colour assignment (the flat-poster path). Returns per-pixel indices.
 */
export function ditherToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  mask: Float32Array | null,
): Uint16Array {
  const n = palette.length;
  const pal = paletteToOklab(palette); // OKLab palette coords
  const lab = new Float32Array(3);
  const out = new Uint16Array(w * h);
  for (let i = 0; i < w * h; i++) {
    srgbToOklabInto(rgb01[i * 3], rgb01[i * 3 + 1], rgb01[i * 3 + 2], lab, 0);
    const r = lab[0]; // L, a, b in OKLab (names kept for diff minimalism)
    const g = lab[1];
    const b = lab[2];
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

// ── Error-diffusion kernels ─────────────────────────────────────────────────
/**
 * A forward-only error-diffusion kernel: a list of [dx, dy, weight] neighbours
 * (dy>0, or dy===0 && dx>0 — error never flows backward over the raster scan)
 * plus the divisor the weights are normalised by. Each named kernel is a
 * different "diffusion character" — the family of looks beyond plain
 * Floyd–Steinberg (smoother, sharper, or sparser grain).
 */
export interface DiffusionKernel {
  divisor: number;
  cells: ReadonlyArray<readonly [number, number, number]>;
}

/**
 * The diffusion-kernel pack. Coefficients are the canonical published matrices.
 * `floyd_steinberg` lives here too so every error-diffused method shares one
 * engine. Note `atkinson` only diffuses 6/8 of the error (divisor 8, weights
 * sum to 6) — that under-diffusion is *the* source of its crisp, high-contrast
 * 1-bit Macintosh look; the others conserve the full error.
 */
export const DIFFUSION_KERNELS: Record<string, DiffusionKernel> = {
  floyd_steinberg: {
    divisor: 16,
    cells: [[1, 0, 7], [-1, 1, 3], [0, 1, 5], [1, 1, 1]],
  },
  atkinson: {
    divisor: 8,
    cells: [[1, 0, 1], [2, 0, 1], [-1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 2, 1]],
  },
  jarvis: {
    divisor: 48,
    cells: [
      [1, 0, 7], [2, 0, 5],
      [-2, 1, 3], [-1, 1, 5], [0, 1, 7], [1, 1, 5], [2, 1, 3],
      [-2, 2, 1], [-1, 2, 3], [0, 2, 5], [1, 2, 3], [2, 2, 1],
    ],
  },
  stucki: {
    divisor: 42,
    cells: [
      [1, 0, 8], [2, 0, 4],
      [-2, 1, 2], [-1, 1, 4], [0, 1, 8], [1, 1, 4], [2, 1, 2],
      [-2, 2, 1], [-1, 2, 2], [0, 2, 4], [1, 2, 2], [2, 2, 1],
    ],
  },
  burkes: {
    divisor: 32,
    cells: [
      [1, 0, 8], [2, 0, 4],
      [-2, 1, 2], [-1, 1, 4], [0, 1, 8], [1, 1, 4], [2, 1, 2],
    ],
  },
  sierra: {
    divisor: 32,
    cells: [
      [1, 0, 5], [2, 0, 3],
      [-2, 1, 2], [-1, 1, 4], [0, 1, 5], [1, 1, 4], [2, 1, 2],
      [-1, 2, 2], [0, 2, 3], [1, 2, 2],
    ],
  },
};

/**
 * Error diffusion onto `palette` using an arbitrary `kernel`. The nearest-colour
 * decision is made *perceptually* (in OKLab), but the residual error is diffused
 * in sRGB — so the choice tracks the eye while total tone stays conserved (the
 * sRGB mean is preserved, matching the Python original). Output is guaranteed
 * in-palette. Returns per-pixel palette indices.
 */
export function errorDiffuseToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  kernel: DiffusionKernel,
): Uint16Array {
  const n = palette.length;
  const pal = new Float32Array(n * 3); // sRGB 0..1 (for the diffused residual)
  for (let j = 0; j < n; j++) {
    pal[j * 3] = palette[j][0] / 255;
    pal[j * 3 + 1] = palette[j][1] / 255;
    pal[j * 3 + 2] = palette[j][2] / 255;
  }
  const palLab = paletteToOklab(palette); // OKLab (for the perceptual decision)
  const buf = Float32Array.from(rgb01); // mutable sRGB working copy
  const out = new Uint16Array(w * h);
  const lab = new Float32Array(3);
  const cells = kernel.cells;
  const inv = 1 / kernel.divisor;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const r = buf[i * 3];
      const g = buf[i * 3 + 1];
      const b = buf[i * 3 + 2];
      // Decide the nearest swatch in OKLab (clamp the error-adjusted value into
      // gamut first so the conversion is well-defined).
      srgbToOklabInto(
        r < 0 ? 0 : r > 1 ? 1 : r,
        g < 0 ? 0 : g > 1 ? 1 : g,
        b < 0 ? 0 : b > 1 ? 1 : b,
        lab,
        0,
      );
      let bj = 0;
      let best = Infinity;
      for (let j = 0; j < n; j++) {
        const dL = lab[0] - palLab[j * 3];
        const dA = lab[1] - palLab[j * 3 + 1];
        const dB = lab[2] - palLab[j * 3 + 2];
        const dd = dL * dL + dA * dA + dB * dB;
        if (dd < best) {
          best = dd;
          bj = j;
        }
      }
      out[i] = bj;
      // Diffuse the residual in sRGB (tone-conserving).
      const er = r - pal[bj * 3];
      const eg = g - pal[bj * 3 + 1];
      const eb = b - pal[bj * 3 + 2];
      for (let c = 0; c < cells.length; c++) {
        const nx = x + cells[c][0];
        const ny = y + cells[c][1];
        if (nx < 0 || nx >= w || ny >= h) continue;
        const f = cells[c][2] * inv;
        const ni = (ny * w + nx) * 3;
        buf[ni] += er * f;
        buf[ni + 1] += eg * f;
        buf[ni + 2] += eb * f;
      }
    }
  }
  return out;
}

/**
 * Floyd–Steinberg error diffusion — kept as a named wrapper over the generic
 * engine (the FS kernel) for callers and tests that target it directly.
 */
export function fsToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
): Uint16Array {
  return errorDiffuseToPalette(rgb01, w, h, palette, DIFFUSION_KERNELS.floyd_steinberg);
}

// ── Yliluoma positional dithering (gamma-correct palette mixing) ──────────────
/**
 * Joel Yliluoma's arbitrary-palette positional dithering (algorithm 2, the
 * Knoll-style variant), gamma-corrected. Unlike ordered/Bayer dithering — which
 * snaps each pixel to one of its two nearest palette colours — this builds a
 * per-colour "mixing plan": the set of palette entries whose *linear-light
 * average* best approximates the true colour. A Bayer index then selects which
 * plan entry shows at each pixel. The payoff: a small *fixed* palette can
 * reproduce colours that aren't in it (it mixes them spatially), so 4–8 colours
 * go a remarkably long way. Mixing happens in linear light (gamma 2.2), which is
 * the physically correct way to blend — so midtones are intentionally darker in
 * gamma space than the sRGB-space error-diffusion methods (that is the "correct
 * colour" the algorithm is named for).
 */
const YLILUOMA_GAMMA = 2.2;

/** sRGB byte (0..255) → linear (0..1), precomputed. */
const SRGB_TO_LINEAR: Float32Array = (() => {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.pow(i / 255, YLILUOMA_GAMMA);
  return lut;
})();

/** linear (0..1) → sRGB value (0..255). */
function linearToSrgb(l: number): number {
  const v = l <= 0 ? 0 : l >= 1 ? 1 : Math.pow(l, 1 / YLILUOMA_GAMMA);
  return v * 255;
}

/**
 * Luminance-weighted perceptual colour distance in gamma space (Yliluoma's
 * `ColorCompare`): the luma term dominates, chroma is down-weighted ×0.75. This
 * is what stops the mixer pairing colours with jarringly different brightness.
 */
function colorCompare(
  r1: number, g1: number, b1: number,
  r2: number, g2: number, b2: number,
): number {
  const luma1 = (r1 * 299 + g1 * 587 + b1 * 114) / 255000;
  const luma2 = (r2 * 299 + g2 * 587 + b2 * 114) / 255000;
  const dl = luma1 - luma2;
  const dr = (r1 - r2) / 255;
  const dg = (g1 - g2) / 255;
  const db = (b1 - b2) / 255;
  return (dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114) * 0.75 + dl * dl;
}

/**
 * Build a mixing plan of `mixSize` palette indices whose linear average best
 * approximates the gamma-space target colour, greedily (add the colour that
 * most improves the running average each step), then sort by luma so the Bayer
 * index walks the plan dark→light.
 */
function deviseMixingPlan(
  tr: number, tg: number, tb: number,
  palGamma: Float32Array,
  palLinear: Float32Array,
  n: number,
  mixSize: number,
): Uint16Array {
  const plan = new Uint16Array(mixSize);
  let sr = 0;
  let sg = 0;
  let sb = 0;
  for (let k = 0; k < mixSize; k++) {
    const t = k + 1;
    let best = 0;
    let bestPenalty = Infinity;
    for (let j = 0; j < n; j++) {
      const ar = linearToSrgb((sr + palLinear[j * 3]) / t);
      const ag = linearToSrgb((sg + palLinear[j * 3 + 1]) / t);
      const ab = linearToSrgb((sb + palLinear[j * 3 + 2]) / t);
      const penalty = colorCompare(tr, tg, tb, ar, ag, ab);
      if (penalty < bestPenalty) {
        bestPenalty = penalty;
        best = j;
      }
    }
    plan[k] = best;
    sr += palLinear[best * 3];
    sg += palLinear[best * 3 + 1];
    sb += palLinear[best * 3 + 2];
  }
  const lumaOf = (j: number) =>
    palGamma[j * 3] * 299 + palGamma[j * 3 + 1] * 587 + palGamma[j * 3 + 2] * 114;
  return Uint16Array.from(Array.from(plan).sort((a, b) => lumaOf(a) - lumaOf(b)));
}

/**
 * Render `rgb01` against `palette` with Yliluoma positional dithering. Mixing
 * plans are cached per quantised target colour (6 bits/channel), so flat
 * regions cost one plan. Returns per-pixel palette indices.
 */
export function yliluomaToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  matrixSize = 8,
): Uint16Array {
  const n = palette.length;
  const palGamma = new Float32Array(n * 3);
  const palLinear = new Float32Array(n * 3);
  for (let j = 0; j < n; j++) {
    const r = clampByte(palette[j][0]);
    const g = clampByte(palette[j][1]);
    const b = clampByte(palette[j][2]);
    palGamma[j * 3] = r;
    palGamma[j * 3 + 1] = g;
    palGamma[j * 3 + 2] = b;
    palLinear[j * 3] = SRGB_TO_LINEAR[r];
    palLinear[j * 3 + 1] = SRGB_TO_LINEAR[g];
    palLinear[j * 3 + 2] = SRGB_TO_LINEAR[b];
  }
  const ms = [2, 4, 8, 16].includes(matrixSize) ? matrixSize : 8;
  const mixSize = ms * ms;
  const bayer = bayerMatrix(ms); // normalized [0,1); ×mixSize ⇒ integer rank
  const out = new Uint16Array(w * h);
  const cache = new Map<number, Uint16Array>();

  for (let y = 0; y < h; y++) {
    const brow = bayer[y % ms];
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const tr = rgb01[i * 3] * 255;
      const tg = rgb01[i * 3 + 1] * 255;
      const tb = rgb01[i * 3 + 2] * 255;
      const qr = clampByte(tr) >> 2;
      const qg = clampByte(tg) >> 2;
      const qb = clampByte(tb) >> 2;
      const key = (qr << 12) | (qg << 6) | qb;
      let plan = cache.get(key);
      if (!plan) {
        plan = deviseMixingPlan(tr, tg, tb, palGamma, palLinear, n, mixSize);
        cache.set(key, plan);
      }
      const rank = Math.min(mixSize - 1, Math.floor(brow[x % ms] * mixSize));
      out[i] = plan[rank];
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
const METHODS: DitherMethod[] = [
  "bayer", "blue_noise", "floyd_steinberg",
  "atkinson", "jarvis", "stucki", "burkes", "sierra",
  "yliluoma", "flow", "flat",
];
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
  if (method === "yliluoma") {
    indices = yliluomaToPalette(rgb01, w, h, palette, p.matrixSize ?? 8);
  } else if (method in DIFFUSION_KERNELS) {
    indices = errorDiffuseToPalette(rgb01, w, h, palette, DIFFUSION_KERNELS[method]);
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
