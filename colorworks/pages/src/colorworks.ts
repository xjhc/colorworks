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
export type PaletteMode = "grayscale" | "adaptive" | "duotone" | "custom";

/**
 * A colour the user pins before generation, to steer which colours the adaptive
 * palette picks (as opposed to recolouring the result after the fact).
 *   - "lock"    — this exact colour is forced into the palette (a frozen k-means
 *                 centroid). It always becomes a candidate; whether it ends up
 *                 *used* still depends on the image.
 *   - "boost"   — points near this hue are up-weighted, so the quantiser spends
 *                 more palette slots (and thus more area) on that colour family.
 *   - "exclude" — points near this hue are removed before clustering, so no
 *                 palette slot lands there.
 * Pins apply to the "adaptive" palette mode only.
 */
export type PinMode = "lock" | "boost" | "exclude";
export interface PinnedColor {
  hex: string;
  mode: PinMode;
  /** boost strength (≥1); ignored for lock/exclude. Default 4. */
  weight?: number;
}
export type DitherMethod =
  | "bayer"
  | "blue_noise"
  | "ign"
  | "r2"
  | "floyd_steinberg"
  | "atkinson"
  | "jarvis"
  | "stucki"
  | "burkes"
  | "sierra"
  | "ostromoukhov"
  | "yliluoma"
  | "knoll"
  | "ema"
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
  candidates?: number; // knoll | ema: N candidate search iterations (default 2×palette, min 16)
  ditherStrength?: number; // knoll: error-compensation scale, 0..1
  serpentine?: boolean; // error diffusion: alternate scan direction per row (default true)
}

export interface RenderOptions {
  colors?: number; // 2..16
  palette?: PaletteMode;
  /** Exact user-supplied palette. Used verbatim when `palette === "custom"`. */
  customPalette?: RGB[];
  method?: DitherMethod;
  contrast?: number;
  midpoint?: number;
  inkColor?: string;
  paperColor?: string;
  params?: RenderParams;
  seed?: number;
  pins?: PinnedColor[];
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
export function srgbToOklabInto(r: number, g: number, b: number, out: Float32Array, o: number): void {
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
export function kmeansPalette(
  raster: Raster,
  colors: number,
  seed = 0,
  iters = 16,
  pins?: PinnedColor[],
): RGB[] {
  void seed;
  colors = Math.max(2, Math.floor(colors));
  if (pins && pins.length) return constrainedKmeansPalette(raster, colors, pins, iters);
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

// ── Constrained adaptive palette (pinned colours) ───────────────────────────────
/** Squared OKLab radius treating two colours as the "same" for pin matching. */
export const PIN_MATCH_R2 = 0.12 * 0.12;
const PIN_BOOST_R2 = 0.16 * 0.16;
const PIN_BOOST_DEFAULT = 4;

interface PinLab {
  L: number;
  A: number;
  B: number;
  rgb: RGB;
  weight: number;
}

function pinLabs(pins: PinnedColor[], mode: PinMode): PinLab[] {
  const out: PinLab[] = [];
  for (const p of pins) {
    if (p.mode !== mode) continue;
    const rgb = parseColor(p.hex);
    const lab = srgbToOklab(rgb);
    out.push({ L: lab[0], A: lab[1], B: lab[2], rgb, weight: Math.max(1, p.weight ?? PIN_BOOST_DEFAULT) });
  }
  return out;
}

/** OKLab labels of every "exclude" pin — for filtering downstream candidates. */
export function excludePinLabs(pins: PinnedColor[] | undefined): Array<[number, number, number]> {
  if (!pins) return [];
  return pinLabs(pins, "exclude").map((p) => [p.L, p.A, p.B] as [number, number, number]);
}

/** True if `rgb` is within PIN_MATCH_R2 of any of the given OKLab labels. */
export function rgbNearAnyLab(rgb: RGB, labs: Array<[number, number, number]>): boolean {
  if (labs.length === 0) return false;
  const lab = srgbToOklab(rgb);
  for (const [L, A, B] of labs) {
    const dL = lab[0] - L;
    const dA = lab[1] - A;
    const dB = lab[2] - B;
    if (dL * dL + dA * dA + dB * dB <= PIN_MATCH_R2) return true;
  }
  return false;
}

/**
 * Adaptive palette with user pins. Locked colours become frozen k-means
 * centroids (always present, exact hue); boosted hues up-weight nearby points so
 * the quantiser spends more slots there; excluded hues drop their points before
 * clustering. Reduces to {@link kmeansPalette} when no pin is effective.
 */
function constrainedKmeansPalette(
  raster: Raster,
  colors: number,
  pins: PinnedColor[],
  iters: number,
): RGB[] {
  colors = Math.max(2, Math.floor(colors));
  const locks = pinLabs(pins, "lock");
  const boosts = pinLabs(pins, "boost");
  const excludes = pinLabs(pins, "exclude");
  if (locks.length === 0 && boosts.length === 0 && excludes.length === 0) {
    return kmeansPalette(raster, colors, 0, iters);
  }

  const lockN = Math.min(locks.length, colors);
  const freeN = colors - lockN;

  // Subsample → OKLab points (same ≤96px lattice as kmeansPalette).
  const { width: w, height: h, data } = raster;
  const maxDim = Math.max(w, h);
  const s = maxDim > 96 ? 96 / maxDim : 1;
  const sw = Math.max(1, Math.round(w * s));
  const sh = Math.max(1, Math.round(h * s));
  const m0 = sw * sh;
  const Praw = new Float32Array(m0 * 3);
  for (let ty = 0; ty < sh; ty++) {
    const syRow = Math.min(h - 1, Math.floor(ty / s)) * w;
    for (let tx = 0; tx < sw; tx++) {
      const sx = Math.min(w - 1, Math.floor(tx / s));
      const src = (syRow + sx) * 4;
      srgbToOklabInto(data[src] / 255, data[src + 1] / 255, data[src + 2] / 255, Praw, (ty * sw + tx) * 3);
    }
  }

  // Exclusion: drop points near any exclude pin (keep all if that empties it).
  const keep: number[] = [];
  for (let i = 0; i < m0; i++) {
    let drop = false;
    for (const e of excludes) {
      const dL = Praw[i * 3] - e.L;
      const dA = Praw[i * 3 + 1] - e.A;
      const dB = Praw[i * 3 + 2] - e.B;
      if (dL * dL + dA * dA + dB * dB <= PIN_MATCH_R2) {
        drop = true;
        break;
      }
    }
    if (!drop) keep.push(i);
  }
  const pick = keep.length >= Math.max(freeN, 1) ? keep : Array.from({ length: m0 }, (_, i) => i);
  const m = pick.length;

  // Fix-palette: no free slots → palette is exactly the locked colours.
  if (freeN <= 0 || m === 0) {
    const sw2 = locks.slice(0, Math.max(colors, 2)).map((l) => l.rgb);
    sw2.sort((a, b) => luma(a[0], a[1], a[2]) - luma(b[0], b[1], b[2]));
    return sw2;
  }

  // Compacted points + per-point boost weights.
  const P = new Float32Array(m * 3);
  const Wt = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const o = pick[i] * 3;
    const L = Praw[o];
    const A = Praw[o + 1];
    const B = Praw[o + 2];
    P[i * 3] = L;
    P[i * 3 + 1] = A;
    P[i * 3 + 2] = B;
    let wt = 1;
    for (const bp of boosts) {
      const dL = L - bp.L;
      const dA = A - bp.A;
      const dB = B - bp.B;
      if (dL * dL + dA * dA + dB * dB <= PIN_BOOST_R2) wt *= bp.weight;
    }
    Wt[i] = wt;
  }

  const sq = (i: number, cL: number, cA: number, cB: number): number => {
    const p = i * 3;
    const dL = P[p] - cL;
    const dA = P[p + 1] - cA;
    const dB = P[p + 2] - cB;
    return dL * dL + dA * dA + dB * dB;
  };

  const k = lockN + freeN; // == colors
  const C = new Float32Array(k * 3);
  for (let l = 0; l < lockN; l++) {
    C[l * 3] = locks[l].L;
    C[l * 3 + 1] = locks[l].A;
    C[l * 3 + 2] = locks[l].B;
  }

  // Weighted median-cut to seed the `freeN` free centroids (boost biases splits).
  const idx = new Int32Array(m);
  for (let i = 0; i < m; i++) idx[i] = i;
  const wstats = (lo: number, hi: number) => {
    let WT = 0;
    let mL = 0;
    let mA = 0;
    let mB = 0;
    for (let i = lo; i < hi; i++) {
      const p = idx[i] * 3;
      const wt = Wt[idx[i]];
      WT += wt;
      mL += wt * P[p];
      mA += wt * P[p + 1];
      mB += wt * P[p + 2];
    }
    if (WT === 0) WT = 1;
    mL /= WT;
    mA /= WT;
    mB /= WT;
    let vL = 0;
    let vA = 0;
    let vB = 0;
    for (let i = lo; i < hi; i++) {
      const p = idx[i] * 3;
      const wt = Wt[idx[i]];
      const dL = P[p] - mL;
      const dA = P[p + 1] - mA;
      const dB = P[p + 2] - mB;
      vL += wt * dL * dL;
      vA += wt * dA * dA;
      vB += wt * dB * dB;
    }
    return { WT, mL, mA, mB, vL, vA, vB, sse: vL + vA + vB };
  };
  const boxes: Array<{ lo: number; hi: number }> = [{ lo: 0, hi: m }];
  while (boxes.length < freeN) {
    let bi = -1;
    let bestSse = -1;
    for (let b = 0; b < boxes.length; b++) {
      if (boxes[b].hi - boxes[b].lo <= 1) continue;
      const sse = wstats(boxes[b].lo, boxes[b].hi).sse;
      if (sse > bestSse) {
        bestSse = sse;
        bi = b;
      }
    }
    if (bi < 0) break;
    const box = boxes[bi];
    const st = wstats(box.lo, box.hi);
    const axis = st.vL >= st.vA && st.vL >= st.vB ? 0 : st.vA >= st.vB ? 1 : 2;
    const slice = Array.from(idx.subarray(box.lo, box.hi));
    slice.sort((p, q) => P[p * 3 + axis] - P[q * 3 + axis]);
    for (let i = 0; i < slice.length; i++) idx[box.lo + i] = slice[i];
    const half = st.WT / 2;
    let acc = 0;
    let mid = box.lo + 1;
    for (let i = box.lo; i < box.hi - 1; i++) {
      acc += Wt[idx[i]];
      mid = i + 1;
      if (acc >= half) break;
    }
    if (mid <= box.lo) mid = box.lo + 1;
    if (mid >= box.hi) mid = box.hi - 1;
    boxes[bi] = { lo: box.lo, hi: mid };
    boxes.push({ lo: mid, hi: box.hi });
  }
  for (let b = 0; b < boxes.length; b++) {
    const st = wstats(boxes[b].lo, boxes[b].hi);
    const j = lockN + b;
    C[j * 3] = st.mL;
    C[j * 3 + 1] = st.mA;
    C[j * 3 + 2] = st.mB;
  }
  // Tiny image: fewer boxes than freeN — fill the rest on far points.
  for (let b = boxes.length; b < freeN; b++) {
    const j = lockN + b;
    let far = 0;
    let worst = -1;
    for (let i = 0; i < m; i++) {
      let best = Infinity;
      for (let q = 0; q < j; q++) {
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

  // Lloyd iterations: assign to all k centroids, update only the free ones.
  const labels = new Int32Array(m);
  const sums = new Float64Array(k * 3);
  const wsum = new Float64Array(k);
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
    wsum.fill(0);
    for (let i = 0; i < m; i++) {
      const j = labels[i];
      const wt = Wt[i];
      wsum[j] += wt;
      sums[j * 3] += wt * P[i * 3];
      sums[j * 3 + 1] += wt * P[i * 3 + 1];
      sums[j * 3 + 2] += wt * P[i * 3 + 2];
    }
    for (let j = lockN; j < k; j++) {
      if (wsum[j] > 0) {
        C[j * 3] = sums[j * 3] / wsum[j];
        C[j * 3 + 1] = sums[j * 3 + 1] / wsum[j];
        C[j * 3 + 2] = sums[j * 3 + 2] / wsum[j];
      } else {
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

  // Locked centroids emit their exact pinned RGB; free centroids round-trip.
  const swatches: RGB[] = [];
  for (let l = 0; l < lockN; l++) swatches.push(locks[l].rgb);
  for (let j = lockN; j < k; j++) swatches.push(oklabToRgb(C[j * 3], C[j * 3 + 1], C[j * 3 + 2]));
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
  pins?: PinnedColor[],
  customPalette?: RGB[],
): RGB[] {
  colors = Math.max(2, Math.min(16, Math.floor(colors)));
  if (mode === "custom" && customPalette && customPalette.length >= 2) {
    const seen = new Set<string>();
    const exact: RGB[] = [];
    for (const color of customPalette.slice(0, 16)) {
      const rgb: RGB = [clampByte(color[0]), clampByte(color[1]), clampByte(color[2])];
      const key = rgbToHex(rgb);
      if (!seen.has(key)) {
        seen.add(key);
        exact.push(rgb);
      }
    }
    if (exact.length >= 2) return exact;
  }
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
  return kmeansPalette(raster, colors, seed, 16, pins);
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

/**
 * Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post Processing
 * in Call of Duty: Advanced Warfare", SIGGRAPH 2014). A single magic-constant
 * formula — no tile, no table — whose spectrum sits between Bayer's rigid grid
 * and true blue noise: fine interleaved diagonal gradients rather than a
 * repeating cell. Cheap enough to be free, and visually distinct from both
 * masks already here.
 */
export function ignThresholdMap(w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const frac = (v: number) => v - Math.floor(v);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = frac(52.9829189 * frac(0.06711056 * x + 0.00583715 * y));
    }
  }
  return out;
}

/**
 * R2 low-discrepancy sequence (Martin Roberts, 2018), the 2-D generalisation of
 * the golden ratio: alpha = 1/phi_2 where phi_2 is the plastic number, the real
 * root of x^3 = x + 1. Sampling it on the pixel lattice gives an even,
 * non-repeating quasirandom mask — no tile period at all, so no visible
 * seams at any output size.
 */
const PLASTIC = 1.32471795724474602596; // phi_2
export function r2ThresholdMap(w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  const a1 = 1 / PLASTIC;
  const a2 = 1 / (PLASTIC * PLASTIC);
  const frac = (v: number) => v - Math.floor(v);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) out[y * w + x] = frac(0.5 + a1 * x + a2 * y);
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
  serpentine = true,
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
    // Serpentine (boustrophedon) scan: alternate direction each row and mirror
    // the kernel's x-offsets to match. A fixed left-to-right raster gives every
    // kernel a directional bias that shows up as correlated diagonal "worms" in
    // highlights and shadows; reversing every other row cancels it.
    const ltr = !serpentine || (y & 1) === 0;
    for (let k = 0; k < w; k++) {
      const x = ltr ? k : w - 1 - k;
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
        const nx = x + (ltr ? cells[c][0] : -cells[c][0]);
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

// ── Ostromoukhov variable-coefficient error diffusion ────────────────────────
/**
 * Victor Ostromoukhov, "A Simple and Efficient Error-Diffusion Algorithm"
 * (SIGGRAPH 2001). Every other kernel here is *fixed*: the same weights at every
 * intensity. That is what makes error diffusion produce correlated diagonal
 * "worms" in highlights and shadows. Ostromoukhov instead varies the three
 * coefficients per input level, using values found by offline optimisation of
 * the output's blue-noise spectrum — so the artifacts cancel where they would
 * otherwise line up. It uses only 3 taps (vs Floyd–Steinberg's 4), on a
 * serpentine path, which the paper notes makes it *faster* than F–S too.
 *
 * Distribution, matching the paper's Figure 1 (right):
 *     A10 → (x+1, y)    A-11 → (x-1, y+1)    A01 → (x, y+1)
 * each divided by M = A10 + A-11 + A01.
 */

/**
 * Appendix I of the paper: `A10, A-11, A01` for input levels 0..127, verbatim.
 * Levels 128..255 come from the paper's stated symmetry D(i) = D(255-i), so
 * only half the table is stored. These are tuned constants — they are not
 * derivable, and interpolating or rounding them degrades the spectrum.
 */
const OSTROMOUKHOV_HALF = Int32Array.from([
  13,0,5, 13,0,5, 21,0,10, 7,0,4, 8,0,5, 47,3,28, 23,3,13, 15,3,8,
  22,6,11, 43,15,20, 7,3,3, 501,224,211, 249,116,103, 165,80,67, 123,62,49, 489,256,191,
  81,44,31, 483,272,181, 60,35,22, 53,32,19, 237,148,83, 471,304,161, 3,2,1, 481,314,185,
  354,226,155, 1389,866,685, 227,138,125, 267,158,163, 327,188,220, 61,34,45, 627,338,505, 1227,638,1075,
  20,10,19, 1937,1000,1767, 977,520,855, 657,360,551, 71,40,57, 2005,1160,1539, 337,200,247, 2039,1240,1425,
  257,160,171, 691,440,437, 1045,680,627, 301,200,171, 177,120,95, 2141,1480,1083, 1079,760,513, 725,520,323,
  137,100,57, 2209,1640,855, 53,40,19, 2243,1720,741, 565,440,171, 759,600,209, 1147,920,285, 2311,1880,513,
  97,80,19, 335,280,57, 1181,1000,171, 793,680,95, 599,520,57, 2413,2120,171, 405,360,19, 2447,2200,57,
  11,10,0, 158,151,3, 178,179,7, 1030,1091,63, 248,277,21, 318,375,35, 458,571,63, 878,1159,147,
  5,7,1, 172,181,37, 97,76,22, 72,41,17, 119,47,29, 4,1,1, 4,1,1, 4,1,1,
  4,1,1, 4,1,1, 4,1,1, 4,1,1, 4,1,1, 4,1,1, 65,18,17, 95,29,26,
  185,62,53, 30,11,9, 35,14,11, 85,37,28, 55,26,19, 80,41,29, 155,86,59, 5,3,2,
  5,3,2, 5,3,2, 5,3,2, 5,3,2, 5,3,2, 5,3,2, 5,3,2, 5,3,2,
  5,3,2, 5,3,2, 5,3,2, 5,3,2, 305,176,119, 155,86,59, 105,56,39, 80,41,29,
  65,32,23, 55,26,19, 335,152,113, 85,37,28, 115,48,37, 35,14,11, 355,136,109, 30,11,9,
  365,128,107, 185,62,53, 25,8,7, 95,29,26, 385,112,103, 65,18,17, 395,104,101, 4,1,1,
]);

/** Full 0..255 table as normalized fractions, mirrored via D(i) = D(255-i). */
const OSTROMOUKHOV_COEFFS: Float32Array = (() => {
  const t = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const src = (i < 128 ? i : 255 - i) * 3;
    const a = OSTROMOUKHOV_HALF[src];
    const b = OSTROMOUKHOV_HALF[src + 1];
    const c = OSTROMOUKHOV_HALF[src + 2];
    const m = a + b + c;
    t[i * 3] = a / m;
    t[i * 3 + 1] = b / m;
    t[i * 3 + 2] = c / m;
  }
  return t;
})();

/**
 * Render `rgb01` against `palette` with Ostromoukhov's variable-coefficient
 * error diffusion.
 *
 * NOTE the published coefficients are indexed by *grayscale* intensity, since
 * the algorithm was designed for bilevel halftoning. For an N-colour palette we
 * index by the luma of the error-adjusted pixel, which preserves the intended
 * per-level behaviour along the tone axis. That is an adaptation, not something
 * the paper specifies.
 */
export function ostromoukhovToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  serpentine = true,
): Uint16Array {
  const n = palette.length;
  const pal = new Float32Array(n * 3);
  for (let j = 0; j < n; j++) {
    pal[j * 3] = palette[j][0] / 255;
    pal[j * 3 + 1] = palette[j][1] / 255;
    pal[j * 3 + 2] = palette[j][2] / 255;
  }
  const palLab = paletteToOklab(palette);
  const buf = Float32Array.from(rgb01);
  const out = new Uint16Array(w * h);
  const lab = new Float32Array(3);

  for (let y = 0; y < h; y++) {
    const ltr = !serpentine || (y & 1) === 0;
    for (let k = 0; k < w; k++) {
      const x = ltr ? k : w - 1 - k;
      const i = y * w + x;
      const r = buf[i * 3];
      const g = buf[i * 3 + 1];
      const b = buf[i * 3 + 2];
      const cr = r < 0 ? 0 : r > 1 ? 1 : r;
      const cg = g < 0 ? 0 : g > 1 ? 1 : g;
      const cb = b < 0 ? 0 : b > 1 ? 1 : b;

      srgbToOklabInto(cr, cg, cb, lab, 0);
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

      // Coefficients follow the tone of the pixel being diffused.
      const level = clampByte(luma(cr, cg, cb) * 255);
      const f10 = OSTROMOUKHOV_COEFFS[level * 3];
      const f11 = OSTROMOUKHOV_COEFFS[level * 3 + 1];
      const f01 = OSTROMOUKHOV_COEFFS[level * 3 + 2];

      const er = r - pal[bj * 3];
      const eg = g - pal[bj * 3 + 1];
      const eb = b - pal[bj * 3 + 2];
      // Taps are inlined rather than pushed through a helper: this is the hot
      // loop, and a per-pixel closure costs more than the tap it saves over F–S.
      const dx = ltr ? 1 : -1;
      const ahead = x + dx; // A10  → next pixel on the scan path
      if (ahead >= 0 && ahead < w) {
        const ni = (y * w + ahead) * 3;
        buf[ni] += er * f10;
        buf[ni + 1] += eg * f10;
        buf[ni + 2] += eb * f10;
      }
      const y1 = y + 1;
      if (y1 < h) {
        const behind = x - dx; // A-11 → "behind" on the row below
        if (behind >= 0 && behind < w) {
          const ni = (y1 * w + behind) * 3;
          buf[ni] += er * f11;
          buf[ni + 1] += eg * f11;
          buf[ni + 2] += eb * f11;
        }
        const ni = (y1 * w + x) * 3; // A01 → directly below
        buf[ni] += er * f01;
        buf[ni + 1] += eg * f01;
        buf[ni + 2] += eb * f01;
      }
    }
  }
  return out;
}

// ── Yliluoma positional dithering (gamma-correct palette mixing) ──────────────
/**
 * Joel Yliluoma's arbitrary-palette positional dithering — algorithm 2 in its
 * simple published form, with algorithm 3's gamma-correct mixing. (It is *not*
 * Knoll's algorithm, which compensates error rather than averaging; see
 * `knollToPalette` below for that one.) Unlike ordered/Bayer dithering — which
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
 *
 * NOTE this adds exactly one copy per step. Yliluoma's own C++ additionally
 * searches a *run length* (how many copies of a candidate to commit at once),
 * which can escape steps where no single addition helps. `emaToPalette` is the
 * modern take on that search; this stays the simple published form.
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

// ── N-candidate ordered dithering: Knoll + Yliluoma-EMA ──────────────────────
/**
 * Two "N-candidate" ordered dithers, following Pekka Väänänen's analysis of
 * Yliluoma's algorithms (https://30fps.net/pages/revisiting-yliluoma-2/).
 *
 * Both share a shape with `yliluoma` above: per pixel, collect N weighted
 * candidate palette colours whose weighted mean approximates the true colour,
 * then let a Bayer threshold pick which candidate shows. They differ in how
 * candidates are found, and in two structural ways:
 *
 *   - They work in a perceptually pre-weighted sRGB space (see `liqWeight`)
 *     where plain squared Euclidean distance is already a good colour metric.
 *     That removes the need for a luma-weighted difference formula — and for
 *     `ema`, it is what makes the mixing factor solvable in closed form.
 *   - Candidates carry *weights* rather than filling plan slots directly, so N
 *     is decoupled from the matrix size. `yliluoma` welds them together
 *     (an 8×8 matrix forces a 64-entry plan); here an 8×8 matrix can run N=16.
 *
 * Note these mix in the (pre-weighted) gamma domain, where the article's
 * constants were tuned — NOT in linear light like `yliluoma`. That is a
 * deliberate difference, and the two look different in the midtones.
 */

/**
 * libimagequant-style perceptual pre-weighting: raise each channel to a fixed
 * exponent, then scale so green dominates and blue is discounted. Applied to
 * BOTH image and palette. Only colour *choice* happens in this space — the
 * original palette is still what gets drawn — so it never shifts output colour.
 *
 * NOTE the source article's prose says "power of 0.8" while its reference
 * script uses 0.57/0.45455 ≈ 1.254; those are reciprocals of each other. We
 * follow the script, being the runnable artifact.
 */
const LIQ_EXPONENT = 0.57 / 0.45455;
const LIQ_WEIGHTS: RGB = [0.5, 1.0, 0.45];

/** Write the pre-weighted form of one [0,1] rgb triple into `out` at `o`. */
function liqWeight(out: Float32Array, o: number, r: number, g: number, b: number): void {
  const cr = r < 0 ? 0 : r > 1 ? 1 : r; // pow() of a negative is NaN — tone can overshoot
  const cg = g < 0 ? 0 : g > 1 ? 1 : g;
  const cb = b < 0 ? 0 : b > 1 ? 1 : b;
  out[o] = Math.pow(cr, LIQ_EXPONENT) * LIQ_WEIGHTS[0];
  out[o + 1] = Math.pow(cg, LIQ_EXPONENT) * LIQ_WEIGHTS[1];
  out[o + 2] = Math.pow(cb, LIQ_EXPONENT) * LIQ_WEIGHTS[2];
}

/**
 * Poynton luma weights, used to order the palette dark→light. The reference
 * prefers these over Yliluoma's Rec.601 as better balanced.
 */
const POYNTON: RGB = [0.309, 0.609, 0.082];

/** Palette prepared for candidate search: pre-weighted colours + luma ordering. */
interface CandidateCtx {
  /** Palette in the pre-weighted space, packed as rgb triples. */
  pw: Float32Array;
  /** Palette indices sorted dark→light. */
  order: Uint16Array;
  n: number;
}

function candidateCtx(palette: RGB[]): CandidateCtx {
  const n = palette.length;
  const pw = new Float32Array(n * 3);
  const lum = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const r = clampByte(palette[j][0]);
    const g = clampByte(palette[j][1]);
    const b = clampByte(palette[j][2]);
    liqWeight(pw, j * 3, r / 255, g / 255, b / 255);
    lum[j] = POYNTON[0] * r + POYNTON[1] * g + POYNTON[2] * b;
  }
  const order = Uint16Array.from(
    Array.from({ length: n }, (_, j) => j).sort((a, b) => lum[a] - lum[b]),
  );
  return { pw, order, n };
}

/** Index of the pre-weighted palette colour closest to (r,g,b) by squared distance. */
function closestWeighted(
  pw: Float32Array, n: number, r: number, g: number, b: number,
): number {
  let best = 0;
  let bestD = Infinity;
  for (let j = 0; j < n; j++) {
    const dr = r - pw[j * 3];
    const dg = g - pw[j * 3 + 1];
    const db = b - pw[j * 3 + 2];
    const d = dr * dr + dg * dg + db * db;
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  return best;
}

/**
 * Collapse per-palette candidate weights (summing to 1) into a `mixSize`-slot
 * plan ordered dark→light: slot `s` holds the palette index at which the
 * cumulative weight first exceeds `(s + 0.5) / mixSize`.
 *
 * This is exactly the reference's per-pixel "walk the luma order until the
 * cumulative sum crosses the Bayer threshold" loop, precomputed once per unique
 * colour — the Bayer values are themselves `(v + 0.5) / mixSize`, so slot
 * `floor(bayer × mixSize)` selects the same entry that loop would. Unused
 * palette entries have weight 0 and are stepped over without ever being picked.
 */
function weightsToPlan(
  weights: Float32Array, order: Uint16Array, n: number, mixSize: number,
): Uint16Array {
  const plan = new Uint16Array(mixSize);
  let cum = 0;
  let oi = 0;
  let idx = order[n - 1];
  for (let s = 0; s < mixSize; s++) {
    // Thresholds ascend, so this walk advances monotonically across all slots.
    while (oi < n) {
      const cand = order[oi];
      if (cum + weights[cand] > (s + 0.5) / mixSize) {
        idx = cand;
        break;
      }
      cum += weights[cand];
      oi++;
    }
    if (oi >= n) idx = order[n - 1]; // float slop only; weights sum to 1
    plan[s] = idx;
  }
  return plan;
}

/** Resolve the shared knobs both N-candidate dithers take. */
function candidateSetup(palette: RGB[], matrixSize: number, candidates: number | undefined) {
  const ctx = candidateCtx(palette);
  const ms = [2, 4, 8, 16].includes(matrixSize) ? matrixSize : 8;
  // The article's rule of thumb is N ≈ 2K; a floor of 16 keeps small palettes
  // (K=2..4, our common case) from looking blocky.
  const fallback = Math.max(16, 2 * ctx.n);
  const N = Math.max(2, Math.min(128, Math.floor(candidates ?? fallback)));
  return { ...ctx, ms, mixSize: ms * ms, bayer: bayerMatrix(ms), N };
}

/**
 * Thomas Knoll's pattern dither (the one in Photoshop), as described by
 * Väänänen — the article's state-of-the-art reference point.
 *
 * Per pixel it walks a goal point around the true colour: take the closest
 * palette entry to the goal, tally it, then push the goal by the error that
 * entry left behind, so the next pick compensates (too blue ⇒ next is
 * yellowish). After N rounds the tally *is* the candidate distribution.
 * `ditherStrength` scales that push: low values keep candidates hugging the
 * true colour (weak dither), high values roam wider (strong mixing).
 */
export function knollToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  matrixSize = 8,
  candidates?: number,
  ditherStrength = 0.8,
): Uint16Array {
  const { pw, order, n, ms, mixSize, bayer, N } = candidateSetup(palette, matrixSize, candidates);
  const strength = ditherStrength < 0 ? 0 : ditherStrength > 1 ? 1 : ditherStrength;
  const out = new Uint16Array(w * h);
  const cache = new Map<number, Uint16Array>();
  const weights = new Float32Array(n);
  const target = new Float32Array(3);

  for (let y = 0; y < h; y++) {
    const brow = bayer[y % ms];
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const qr = clampByte(rgb01[i * 3] * 255);
      const qg = clampByte(rgb01[i * 3 + 1] * 255);
      const qb = clampByte(rgb01[i * 3 + 2] * 255);
      const key = ((qr >> 2) << 12) | ((qg >> 2) << 6) | (qb >> 2);
      let plan = cache.get(key);
      if (!plan) {
        liqWeight(target, 0, qr / 255, qg / 255, qb / 255);
        weights.fill(0);
        let er = 0;
        let eg = 0;
        let eb = 0;
        for (let it = 0; it < N; it++) {
          const j = closestWeighted(
            pw, n,
            target[0] + er * strength,
            target[1] + eg * strength,
            target[2] + eb * strength,
          );
          weights[j] += 1;
          er += target[0] - pw[j * 3];
          eg += target[1] - pw[j * 3 + 1];
          eb += target[2] - pw[j * 3 + 2];
        }
        for (let j = 0; j < n; j++) weights[j] /= N;
        plan = weightsToPlan(weights, order, n, mixSize);
        cache.set(key, plan);
      }
      out[i] = plan[Math.min(mixSize - 1, Math.floor(brow[x % ms] * mixSize))];
    }
  }
  return out;
}

/**
 * Yliluoma-2 recast as Väänänen's "EMA-Exact" variant.
 *
 * The candidate set is built as an exponential moving average: the running mean
 * starts at the closest palette colour, and each round picks the palette entry
 * whose blend with that mean lands nearest the true colour. Because the metric
 * is Euclidean in the pre-weighted space, the best blend factor `t` along each
 * segment has a closed form (project the target onto it) instead of needing the
 * original's parameter sweep — same result, one test per colour instead of 8+.
 *
 * `t` is clamped: 0 would leave the mean untouched, 1 would discard it wholly.
 * The 0.2 floor is the article's empirical pick — the unclamped optimum sits at
 * ~0 about 97% of the time, which stalls the walk.
 */
const EMA_MIN_T = 0.2;
const EMA_MAX_T = 0.975;

export function emaToPalette(
  rgb01: Float32Array,
  w: number,
  h: number,
  palette: RGB[],
  matrixSize = 8,
  candidates?: number,
): Uint16Array {
  const { pw, order, n, ms, mixSize, bayer, N } = candidateSetup(palette, matrixSize, candidates);
  const out = new Uint16Array(w * h);
  const cache = new Map<number, Uint16Array>();
  const weights = new Float32Array(n);
  const target = new Float32Array(3);

  for (let y = 0; y < h; y++) {
    const brow = bayer[y % ms];
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const qr = clampByte(rgb01[i * 3] * 255);
      const qg = clampByte(rgb01[i * 3 + 1] * 255);
      const qb = clampByte(rgb01[i * 3 + 2] * 255);
      const key = ((qr >> 2) << 12) | ((qg >> 2) << 6) | (qb >> 2);
      let plan = cache.get(key);
      if (!plan) {
        liqWeight(target, 0, qr / 255, qg / 255, qb / 255);
        const pr = target[0];
        const pg = target[1];
        const pb = target[2];
        weights.fill(0);
        const seed = closestWeighted(pw, n, pr, pg, pb);
        weights[seed] = 1;
        let mr = pw[seed * 3];
        let mg = pw[seed * 3 + 1];
        let mb = pw[seed * 3 + 2];

        for (let it = 0; it < N; it++) {
          let bestK = 0;
          let bestT = EMA_MIN_T;
          let bestD = Infinity;
          for (let k = 0; k < n; k++) {
            const dr = pw[k * 3] - mr;
            const dg = pw[k * 3 + 1] - mg;
            const db = pw[k * 3 + 2] - mb;
            const d2 = dr * dr + dg * dg + db * db;
            // t = ((p − mean) · delta) / ‖delta‖², clamped to the usable range.
            let t = 0;
            if (d2 >= 1e-9) t = ((pr - mr) * dr + (pg - mg) * dg + (pb - mb) * db) / d2;
            t = t < EMA_MIN_T ? EMA_MIN_T : t > EMA_MAX_T ? EMA_MAX_T : t;
            const xr = pr - (mr + t * dr);
            const xg = pg - (mg + t * dg);
            const xb = pb - (mb + t * db);
            const d = xr * xr + xg * xg + xb * xb;
            if (d < bestD) {
              bestD = d;
              bestK = k;
              bestT = t;
            }
          }
          weights[bestK] += bestT;
          mr += bestT * (pw[bestK * 3] - mr);
          mg += bestT * (pw[bestK * 3 + 1] - mg);
          mb += bestT * (pw[bestK * 3 + 2] - mb);
        }

        let sum = 0;
        for (let j = 0; j < n; j++) sum += weights[j];
        if (sum < 1e-6) sum = 1e-6;
        for (let j = 0; j < n; j++) weights[j] /= sum;
        plan = weightsToPlan(weights, order, n, mixSize);
        cache.set(key, plan);
      }
      out[i] = plan[Math.min(mixSize - 1, Math.floor(brow[x % ms] * mixSize))];
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
  "bayer", "blue_noise", "ign", "r2", "floyd_steinberg",
  "atkinson", "jarvis", "stucki", "burkes", "sierra",
  "ostromoukhov", "yliluoma", "knoll", "ema", "flow", "flat",
];
const MODES: PaletteMode[] = ["grayscale", "adaptive", "duotone", "custom"];

/**
 * Render a raster as an N-colour dither — the heart of the target aesthetic.
 * Returns `{ indices, palette }` (deduped, luma-sorted); paint with
 * `palette[indices[i]]`.
 */
export function renderToneDither(raster: Raster, opts: RenderOptions = {}): RenderResult {
  const { width: w, height: h } = raster;
  const requestedColors = opts.palette === "custom" && opts.customPalette
    ? opts.customPalette.length
    : (opts.colors ?? 4);
  const colors = Math.max(2, Math.min(16, Math.floor(requestedColors)));
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
    opts.pins,
    opts.customPalette,
  );

  const rgb01 = rasterToRgb01(raster);
  const gray = rasterToGray(raster);
  applyTone(rgb01, gray, contrast, midpoint);

  let indices: Uint16Array;
  if (method === "yliluoma") {
    indices = yliluomaToPalette(rgb01, w, h, palette, p.matrixSize ?? 8);
  } else if (method === "ostromoukhov") {
    indices = ostromoukhovToPalette(rgb01, w, h, palette, p.serpentine ?? true);
  } else if (method === "knoll") {
    indices = knollToPalette(
      rgb01, w, h, palette, p.matrixSize ?? 8, p.candidates, p.ditherStrength ?? 0.8,
    );
  } else if (method === "ema") {
    indices = emaToPalette(rgb01, w, h, palette, p.matrixSize ?? 8, p.candidates);
  } else if (method in DIFFUSION_KERNELS) {
    indices = errorDiffuseToPalette(
      rgb01, w, h, palette, DIFFUSION_KERNELS[method], p.serpentine ?? true,
    );
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
    } else if (method === "ign") {
      mask = ignThresholdMap(w, h);
    } else if (method === "r2") {
      mask = r2ThresholdMap(w, h);
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
