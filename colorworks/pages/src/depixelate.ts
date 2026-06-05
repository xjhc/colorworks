/* ============================================================================
   Depixelate — recover the native pixel grid of an upscaled image and re-render
   each cell as a small two-colour ordered-dither tile ("one dot -> one pixel").

   TypeScript port of colorworks/algorithms/depixelate.py (the reference impl),
   running fully client-side. Detection: saturation-weighted edge profiles ->
   autocorrelation fundamental -> sub-pixel comb refine -> square pitch. Reduction:
   per-cell two colours laid out by a Bayer-ordered dither whose density matches
   the cell's colour mix (block=2 midtone -> the o/x checkerboard).
   ========================================================================== */
import type { Raster, RGB, RenderResult } from "./colorworks";

export interface Grid {
  pitchX: number;
  pitchY: number;
  phaseX: number;
  phaseY: number;
  confidence: number;
}

export interface DepixelateOptions {
  block?: number; // tile size (2 = checkerboard)
  tau?: number; // colour distance that registers a second colour
  pitch?: number; // 0 / undefined = auto-detect the upscale grid
}

// ── grid detection ────────────────────────────────────────────────────────────

/** Per-axis edge-energy profiles (profileX length W-1, profileY length H-1).
 *  Edges are saturation-weighted when the image carries a colourful subject on a
 *  grey field, so the subject's grid wins over a competing background grid. */
function edgeProfiles(r: Raster): { px: Float64Array; py: Float64Array } {
  const { width: W, height: H, data } = r;
  const sat = new Float32Array(W * H);
  let colourful = 0;
  for (let p = 0, i = 0; p < W * H; p++, i += 4) {
    const s = Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]);
    sat[p] = s;
    if (s > 60) colourful++;
  }
  const weighted = colourful > 2000;

  const px = new Float64Array(W - 1);
  const py = new Float64Array(H - 1);

  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W - 1; x++) {
      const i0 = (row + x) * 4;
      const i1 = i0 + 4;
      const g =
        Math.abs(data[i0] - data[i1]) +
        Math.abs(data[i0 + 1] - data[i1 + 1]) +
        Math.abs(data[i0 + 2] - data[i1 + 2]);
      px[x] += g * (weighted ? Math.max(sat[row + x], sat[row + x + 1]) : 1);
    }
  }
  for (let y = 0; y < H - 1; y++) {
    const row = y * W;
    const row2 = row + W;
    for (let x = 0; x < W; x++) {
      const i0 = (row + x) * 4;
      const i1 = (row2 + x) * 4;
      const g =
        Math.abs(data[i0] - data[i1]) +
        Math.abs(data[i0 + 1] - data[i1 + 1]) +
        Math.abs(data[i0 + 2] - data[i1 + 2]);
      py[y] += g * (weighted ? Math.max(sat[row + x], sat[row2 + x]) : 1);
    }
  }
  return { px, py };
}

const mean = (a: Float64Array): number => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
};

/** Coarse fundamental period: the smallest autocorrelation peak within half the
 *  strongest, so we land on the fundamental rather than a 2x/3x harmonic. */
function fundamental(sig: Float64Array, minP: number, maxP: number): number {
  const m = mean(sig);
  const s = new Float64Array(sig.length);
  for (let i = 0; i < sig.length; i++) s[i] = sig[i] - m;

  const ac = (lag: number): number => {
    let acc = 0;
    for (let i = 0; i + lag < s.length; i++) acc += s[i] * s[i + lag];
    return acc;
  };
  const ac0 = ac(0) + 1e-12;
  const hi = Math.min(maxP, s.length - 2);
  const val: number[] = [];
  for (let lag = minP - 1; lag <= hi + 1; lag++) val[lag] = ac(lag) / ac0;

  const peaks: Array<[number, number]> = [];
  for (let lag = minP; lag <= hi; lag++) {
    if (val[lag] >= val[lag - 1] && val[lag] >= val[lag + 1]) peaks.push([lag, val[lag]]);
  }
  if (!peaks.length) return minP;
  const strongest = Math.max(...peaks.map((p) => p[1]));
  for (const [lag, v] of peaks) if (v >= 0.5 * strongest) return lag; // ascending-lag order
  return peaks[0][0];
}

function combScore(sig: Float64Array, pitch: number, phase: number): number {
  let sum = 0;
  let count = 0;
  for (let x = phase; x < sig.length; x += pitch) {
    const idx = Math.round(x);
    if (idx < sig.length) {
      sum += sig[idx];
      count++;
    }
  }
  return count ? sum / count : 0;
}

function bestPhase(sig: Float64Array, pitch: number): { phase: number; score: number } {
  let score = -Infinity;
  let phase = 0;
  for (let ph = 0; ph < pitch; ph += 0.5) {
    const sc = combScore(sig, pitch, ph);
    if (sc > score) {
      score = sc;
      phase = ph;
    }
  }
  return { phase, score };
}

/** Sub-pixel pitch maximising the combined comb score within +/-15% of coarse. */
function refineSquare(profiles: Float64Array[], coarse: number): number {
  const lo = coarse * 0.85;
  const hi = coarse * 1.15;
  let best = -Infinity;
  let bestP = coarse;
  for (let p = lo; p < hi; p += 0.05) {
    let s = 0;
    for (const prof of profiles) s += bestPhase(prof, p).score;
    if (s > best) {
      best = s;
      bestP = p;
    }
  }
  return bestP;
}

export function detectGrid(r: Raster, minPitch = 6, maxPitch = 64): Grid {
  const { px, py } = edgeProfiles(r);
  const coarse = Math.min(fundamental(px, minPitch, maxPitch), fundamental(py, minPitch, maxPitch));
  const pitch = refineSquare([px, py], coarse);
  const phx = bestPhase(px, pitch);
  const phy = bestPhase(py, pitch);
  const contrast = 0.5 * (phx.score / (mean(px) + 1e-12) + phy.score / (mean(py) + 1e-12));
  const confidence = Math.min(1, Math.max(0, (contrast - 1) / 4));
  return { pitchX: pitch, pitchY: pitch, phaseX: phx.phase, phaseY: phy.phase, confidence };
}

function gridOrigin(g: Grid): [number, number] {
  // _best_phase locks onto the boundary comb, which can land near a cell's far
  // edge; fold any phase past the half-pitch back so the origin sits near 0.
  const ox = g.phaseX > g.pitchX / 2 ? g.phaseX - g.pitchX : g.phaseX;
  const oy = g.phaseY > g.pitchY / 2 ? g.phaseY - g.pitchY : g.phaseY;
  return [ox, oy];
}

// ── tile reduction ────────────────────────────────────────────────────────────

const isPow2 = (n: number): boolean => (n & (n - 1)) === 0;

function bayerRanks(n: number): number[][] {
  if (n === 1) return [[0]];
  const h = bayerRanks(n >> 1);
  const s = n >> 1;
  const m: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const v = h[y][x];
      m[y][x] = 4 * v;
      m[y][x + s] = 4 * v + 2;
      m[y + s][x] = 4 * v + 3;
      m[y + s][x + s] = 4 * v + 1;
    }
  }
  return m;
}

/** n x n dispersed ordering, ranks 0..n*n-1. Exact Bayer for powers of two;
 *  otherwise a ranked sample of a 16x16 Bayer keeps the dispersed character. */
function ditherOrder(n: number): number[][] {
  if (isPow2(n)) return bayerRanks(n);
  const base = bayerRanks(16);
  const idx = Array.from({ length: n }, (_, k) => Math.round((k * 15) / (n - 1)));
  const vals: number[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) vals.push(base[idx[y]][idx[x]]);
  const order = vals.map((_, i) => i).sort((a, b) => vals[a] - vals[b]);
  const rank = new Array<number>(vals.length);
  order.forEach((orig, r) => (rank[orig] = r));
  const m: number[][] = [];
  for (let y = 0; y < n; y++) {
    m.push([]);
    for (let x = 0; x < n; x++) m[y].push(rank[y * n + x]);
  }
  return m;
}

const colourKey = (r: number, g: number, b: number): number => (r << 16) | (g << 8) | b;
const dist = (r: number, g: number, b: number, c: RGB): number =>
  Math.max(Math.abs(r - c[0]), Math.abs(g - c[1]), Math.abs(b - c[2]));

/** Most common colour in a window of the raster. */
function windowMode(
  data: Uint8ClampedArray,
  W: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  predicate?: (r: number, g: number, b: number) => boolean,
): { colour: RGB; count: number; total: number } {
  const counts = new Map<number, number>();
  let total = 0;
  let kept = 0;
  for (let y = y0; y <= y1; y++) {
    const row = y * W;
    for (let x = x0; x <= x1; x++) {
      const i = (row + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      total++;
      if (predicate && !predicate(r, g, b)) continue;
      kept++;
      const k = colourKey(r, g, b);
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
  }
  let bestKey = 0;
  let bestCount = -1;
  for (const [k, c] of counts) {
    if (c > bestCount) {
      bestCount = c;
      bestKey = k;
    }
  }
  return {
    colour: [(bestKey >> 16) & 255, (bestKey >> 8) & 255, bestKey & 255],
    count: kept,
    total,
  };
}

/** Render each grid cell as a block x block two-colour ordered-dither tile.
 *  Output raster is `block`x the native cell grid. */
export function reduceToTiles(r: Raster, grid: Grid, block: number, tau: number, minFrac = 0.04): Raster {
  const { width: W, height: H, data } = r;
  const order = ditherOrder(block);
  const nSub = block * block;
  const { pitchX: px, pitchY: py } = grid;
  const [ox, oy] = gridOrigin(grid);
  const nCols = Math.round((W - ox) / px);
  const nRows = Math.round((H - oy) / py);
  const outW = nCols * block;
  const outH = nRows * block;
  const out = new Uint8ClampedArray(outW * outH * 4);
  const halfX = Math.max(1, Math.floor((px * 0.9) / 2));
  const halfY = Math.max(1, Math.floor((py * 0.9) / 2));

  for (let rr = 0; rr < nRows; rr++) {
    const cy = Math.round(oy + (rr + 0.5) * py);
    const y0 = Math.max(0, cy - halfY);
    const y1 = Math.min(H - 1, cy + halfY);
    for (let cc = 0; cc < nCols; cc++) {
      const cx = Math.round(ox + (cc + 0.5) * px);
      const x0 = Math.max(0, cx - halfX);
      const x1 = Math.min(W - 1, cx + halfX);

      const m0 = windowMode(data, W, x0, x1, y0, y1);
      const c0 = m0.colour;
      const total = m0.count;
      const minCount = Math.max(2, Math.floor(minFrac * total));
      const far = windowMode(data, W, x0, x1, y0, y1, (r2, g2, b2) => dist(r2, g2, b2, c0) > tau);
      const c1: RGB = far.count >= minCount ? far.colour : c0;

      let on: number;
      if (c0[0] === c1[0] && c0[1] === c1[1] && c0[2] === c1[2]) {
        on = 0; // solid cell -> uniform tile
      } else {
        const frac = far.count / total;
        on = Math.min(nSub - 1, Math.max(1, Math.round(frac * nSub)));
      }

      for (let i = 0; i < block; i++) {
        for (let j = 0; j < block; j++) {
          const c = order[i][j] < on ? c1 : c0;
          const oi = ((rr * block + i) * outW + (cc * block + j)) * 4;
          out[oi] = c[0];
          out[oi + 1] = c[1];
          out[oi + 2] = c[2];
          out[oi + 3] = 255;
        }
      }
    }
  }
  return { width: outW, height: outH, data: out };
}

// ── studio entry point ────────────────────────────────────────────────────────

/** Index a small raster into a palette + per-pixel indices (RenderResult shape),
 *  so the studio's recolour / swatch / export machinery applies unchanged. */
function rasterToIndexed(r: Raster): RenderResult {
  const { width, height, data } = r;
  const n = width * height;
  const indices = new Uint16Array(n);
  const map = new Map<number, number>();
  const palette: RGB[] = [];
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const k = colourKey(data[i], data[i + 1], data[i + 2]);
    let idx = map.get(k);
    if (idx === undefined) {
      idx = palette.length;
      map.set(k, idx);
      palette.push([data[i], data[i + 1], data[i + 2]]);
    }
    indices[p] = idx;
  }
  return { indices, palette, width, height };
}

export function renderDepixelate(r: Raster, opts: DepixelateOptions = {}): RenderResult {
  const block = opts.block ?? 2;
  const tau = opts.tau ?? 45;
  let grid: Grid;
  if (opts.pitch && opts.pitch > 0) {
    const { px, py } = edgeProfiles(r);
    grid = {
      pitchX: opts.pitch,
      pitchY: opts.pitch,
      phaseX: bestPhase(px, opts.pitch).phase,
      phaseY: bestPhase(py, opts.pitch).phase,
      confidence: 1,
    };
  } else {
    grid = detectGrid(r);
  }
  return rasterToIndexed(reduceToTiles(r, grid, block, tau));
}
