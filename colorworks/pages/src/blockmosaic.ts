/* ============================================================================
   Block mosaic — fill an image with multi-colour block tiles.

   Reproduce a target image by tiling it with small fixed colour blocks: a b×b
   grid of colours, e.g. a 2×2 "c1 c2 / c3 c4". For each block-sized SLOT of the
   image, pick the candidate block whose cell colours best match that region (in
   OKLab), then stamp the whole block. Candidates come from a PRESET library
   (solids / checkers / diagonals built from the image's adaptive palette) plus
   K LEARNED blocks discovered from the image by k-means vector quantisation. A
   bias prefers the library but isn't forced to use it — a learned block wins only
   when it's materially better.

   Two fill modes:
     - "match":   pure nearest-block per slot — crisp, repeating structure.
     - "diffuse": Floyd–Steinberg diffusion of each slot's mean residual onto its
                  neighbours — recovers regional tone the fixed vocabulary can't
                  hit (the dithered-photo look).

   Pure (raster, opts) → RenderResult, mirroring the other studio renderers
   (tone_dither / depixelate / repixel). No DOM, no fetch — safe in vitest.
   ========================================================================== */
import {
  buildTonePalette,
  rasterToRgb01,
  rasterToGray,
  applyTone,
  oklabToRgb,
  srgbToOklabInto,
  type PaletteMode,
  type Raster,
  type RGB,
  type RenderResult,
} from "./colorworks";
import { rasterToIndexed } from "./depixelate";

/** A w×h colour tile: `cells` row-major (length w*h), each an sRGB triplet. */
export interface Block {
  w: number;
  h: number;
  cells: RGB[];
}

/** Which structured preset blocks to seed the candidate set with.
 *  "none" = learned blocks only (pure vector-quantisation mosaic). */
export type BlockLibrary = "auto" | "solids" | "checker" | "none";

/** "match" = nearest block (clean); "diffuse" = error-diffused tone. */
export type BlockMosaicMethod = "match" | "diffuse";

export interface BlockMosaicOptions {
  block?: number; // cells per block edge (square), 2..4
  cell?: number; // output px per cell, 2..16
  colors?: number; // base palette size feeding the presets, 2..8
  palette?: PaletteMode; // grayscale | adaptive | duotone (base palette mode)
  inkColor?: string; // duotone dark
  paperColor?: string; // duotone light
  library?: BlockLibrary; // which preset blocks to include
  learn?: number; // K learned blocks (0 = presets only)
  libraryBias?: number; // 0..1 preference for preset blocks at match time
  method?: BlockMosaicMethod; // fill mode (default "match")
  contrast?: number;
  midpoint?: number;
}

// ── candidate construction ─────────────────────────────────────────────────────

/** Pack a block's cells into a flat OKLab Float32Array (length cells*3). */
function blockToOklab(block: Block): Float32Array {
  const out = new Float32Array(block.cells.length * 3);
  for (let i = 0; i < block.cells.length; i++) {
    const c = block.cells[i];
    srgbToOklabInto(c[0] / 255, c[1] / 255, c[2] / 255, out, i * 3);
  }
  return out;
}

/**
 * Build the preset block library from a base palette. Every kind but "none"
 * includes one solid block per colour; "checker" adds two-colour checkerboards
 * over every colour pair; "auto" adds diagonal splits as well. The pair count is
 * O(n²) — bounded (n ≤ 8) so the candidate set stays small.
 */
export function buildPresetLibrary(kind: BlockLibrary, palette: RGB[], b: number): Block[] {
  if (kind === "none") return [];
  const n = palette.length;
  const fromFn = (fn: (i: number, j: number) => RGB): Block => {
    const cells: RGB[] = [];
    for (let i = 0; i < b; i++) for (let j = 0; j < b; j++) cells.push(fn(i, j));
    return { w: b, h: b, cells };
  };
  const blocks: Block[] = [];
  for (let i = 0; i < n; i++) blocks.push(fromFn(() => palette[i])); // solids
  if (kind === "solids") return blocks;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = palette[i];
      const c = palette[j];
      blocks.push(fromFn((r, s) => ((r + s) & 1 ? c : a))); // checkerboard
      if (kind === "auto") blocks.push(fromFn((r, s) => (s >= r ? c : a))); // diagonal split
    }
  }
  return blocks;
}

/**
 * Learn K representative blocks from the image by k-means vector quantisation:
 * each slot is a point in OKLab space of dimension b²·3 (its cells stacked), and
 * the K cluster centroids become blocks. Deterministic — farthest-point seeding
 * (no RNG) then Lloyd iterations — so the same image always yields the same
 * codebook. This is what lets the mosaic represent colours/arrangements the hand
 * authored presets don't cover.
 */
export function learnBlocks(
  targetLab: Float32Array,
  cellCols: number,
  cellRows: number,
  b: number,
  K: number,
): Block[] {
  const cols = Math.floor(cellCols / b);
  const rows = Math.floor(cellRows / b);
  const nSlots = cols * rows;
  const D = b * b * 3;
  if (K <= 0 || nSlots === 0) return [];
  K = Math.min(K, nSlots);

  // Gather slot vectors (OKLab), dim D.
  const V = new Float32Array(nSlots * D);
  for (let sr = 0; sr < rows; sr++) {
    for (let sc = 0; sc < cols; sc++) {
      let o = (sr * cols + sc) * D;
      for (let i = 0; i < b; i++) {
        const cr = sr * b + i;
        for (let j = 0; j < b; j++) {
          const ci = (cr * cellCols + (sc * b + j)) * 3;
          V[o++] = targetLab[ci];
          V[o++] = targetLab[ci + 1];
          V[o++] = targetLab[ci + 2];
        }
      }
    }
  }

  // Deterministic farthest-point seeding: slot 0, then repeatedly the slot
  // farthest from the centroids chosen so far.
  const C = new Float32Array(K * D);
  for (let d = 0; d < D; d++) C[d] = V[d];
  const dist2 = (s: number, co: number): number => {
    let acc = 0;
    const so = s * D;
    for (let d = 0; d < D; d++) {
      const diff = V[so + d] - C[co + d];
      acc += diff * diff;
    }
    return acc;
  };
  const nearest = new Float32Array(nSlots);
  for (let s = 0; s < nSlots; s++) nearest[s] = dist2(s, 0);
  for (let k = 1; k < K; k++) {
    let far = 0;
    let farD = -1;
    for (let s = 0; s < nSlots; s++) {
      if (nearest[s] > farD) {
        farD = nearest[s];
        far = s;
      }
    }
    for (let d = 0; d < D; d++) C[k * D + d] = V[far * D + d];
    for (let s = 0; s < nSlots; s++) {
      const dd = dist2(s, k * D);
      if (dd < nearest[s]) nearest[s] = dd;
    }
  }

  // Lloyd iterations.
  const labels = new Int32Array(nSlots);
  const sums = new Float64Array(K * D);
  const counts = new Int32Array(K);
  for (let it = 0; it < 12; it++) {
    for (let s = 0; s < nSlots; s++) {
      let best = Infinity;
      let bj = 0;
      for (let k = 0; k < K; k++) {
        const dd = dist2(s, k * D);
        if (dd < best) {
          best = dd;
          bj = k;
        }
      }
      labels[s] = bj;
    }
    sums.fill(0);
    counts.fill(0);
    for (let s = 0; s < nSlots; s++) {
      const k = labels[s];
      counts[k]++;
      const so = s * D;
      const ko = k * D;
      for (let d = 0; d < D; d++) sums[ko + d] += V[so + d];
    }
    for (let k = 0; k < K; k++) {
      if (counts[k] === 0) continue; // keep an empty centroid where it is
      const ko = k * D;
      for (let d = 0; d < D; d++) C[ko + d] = sums[ko + d] / counts[k];
    }
  }

  const blocks: Block[] = [];
  for (let k = 0; k < K; k++) {
    const cells: RGB[] = [];
    for (let i = 0; i < b * b; i++) {
      const o = k * D + i * 3;
      cells.push(oklabToRgb(C[o], C[o + 1], C[o + 2]));
    }
    blocks.push({ w: b, h: b, cells });
  }
  return blocks;
}

// ── matching ───────────────────────────────────────────────────────────────────

/**
 * Pick the candidate block index minimising summed OKLab ΔE² over a slot's cells.
 * `libFactor` (≤ 1) is a multiplicative discount applied to library candidates so
 * presets are preferred unless a learned block is materially better — the
 * "tends to use the library, not forced" behaviour. `libFactor === 1` is neutral.
 */
export function chooseCandidate(
  slotLab: Float32Array,
  candLab: Float32Array[],
  candIsLib: boolean[],
  libFactor: number,
): number {
  let bestK = 0;
  let bestScore = Infinity;
  for (let k = 0; k < candLab.length; k++) {
    const cl = candLab[k];
    let score = 0;
    for (let d = 0; d < slotLab.length; d++) {
      const diff = slotLab[d] - cl[d];
      score += diff * diff;
    }
    if (candIsLib[k]) score *= libFactor;
    if (score < bestScore) {
      bestScore = score;
      bestK = k;
    }
  }
  return bestK;
}

// ── downsample ───────────────────────────────────────────────────────────────

/** Box-average a tone-mapped sRGB raster (0..1) into a cellCols×cellRows grid. */
function downsampleToCells(
  rgb01: Float32Array,
  W: number,
  H: number,
  cellCols: number,
  cellRows: number,
): Float32Array {
  const out = new Float32Array(cellCols * cellRows * 3);
  for (let cy = 0; cy < cellRows; cy++) {
    const y0 = Math.floor((cy * H) / cellRows);
    const y1 = Math.max(y0 + 1, Math.floor(((cy + 1) * H) / cellRows));
    for (let cx = 0; cx < cellCols; cx++) {
      const x0 = Math.floor((cx * W) / cellCols);
      const x1 = Math.max(x0 + 1, Math.floor(((cx + 1) * W) / cellCols));
      let r = 0;
      let g = 0;
      let bl = 0;
      let n = 0;
      for (let y = y0; y < y1 && y < H; y++) {
        const row = y * W;
        for (let x = x0; x < x1 && x < W; x++) {
          const i = (row + x) * 3;
          r += rgb01[i];
          g += rgb01[i + 1];
          bl += rgb01[i + 2];
          n++;
        }
      }
      const o = (cy * cellCols + cx) * 3;
      const inv = n > 0 ? 1 / n : 0;
      out[o] = r * inv;
      out[o + 1] = g * inv;
      out[o + 2] = bl * inv;
    }
  }
  return out;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

// ── top-level render ───────────────────────────────────────────────────────────

const PALETTE_MODES: PaletteMode[] = ["grayscale", "adaptive", "duotone"];

/**
 * Render `raster` as a block mosaic. Works on the output-sized raster (like
 * tone_dither), so the result honours the studio's output-size control directly.
 * Returns `{ indices, palette }`; paint with `palette[indices[i]]`.
 */
export function renderBlockMosaic(raster: Raster, opts: BlockMosaicOptions = {}): RenderResult {
  const { width: W, height: H } = raster;
  const b = Math.max(2, Math.min(4, Math.floor(opts.block ?? 2)));
  const cell = Math.max(1, Math.min(16, Math.floor(opts.cell ?? 6)));
  const colors = Math.max(2, Math.min(8, Math.floor(opts.colors ?? 4)));
  const palMode: PaletteMode = PALETTE_MODES.includes(opts.palette as PaletteMode)
    ? (opts.palette as PaletteMode)
    : "adaptive";
  const libraryKind: BlockLibrary = opts.library ?? "auto";
  const learnK = Math.max(0, Math.min(16, Math.floor(opts.learn ?? 6)));
  const libraryBias = Math.max(0, Math.min(1, opts.libraryBias ?? 0.5));
  const method: BlockMosaicMethod = opts.method === "diffuse" ? "diffuse" : "match";
  const contrast = opts.contrast ?? 1;
  const midpoint = opts.midpoint ?? 0.5;

  // Slot lattice: cols×rows blocks, each b×b cells, each cell cell×cell px.
  const slotPx = b * cell;
  const cols = Math.max(1, Math.round(W / slotPx));
  const rows = Math.max(1, Math.round(H / slotPx));
  const cellCols = cols * b;
  const cellRows = rows * b;

  // Tone-mapped source → cell grid (sRGB 0..1) → OKLab cell grid.
  const rgb01 = rasterToRgb01(raster);
  const gray = rasterToGray(raster);
  applyTone(rgb01, gray, contrast, midpoint);
  const target = downsampleToCells(rgb01, W, H, cellCols, cellRows);
  const nCells = cellCols * cellRows;
  const targetLab = new Float32Array(nCells * 3);
  for (let i = 0; i < nCells; i++) {
    srgbToOklabInto(target[i * 3], target[i * 3 + 1], target[i * 3 + 2], targetLab, i * 3);
  }

  // Candidate blocks: presets (library) ∪ learned, with provenance.
  const palette = buildTonePalette(
    raster,
    colors,
    palMode,
    opts.inkColor ?? "#161616",
    opts.paperColor ?? "#f4ebd9",
  );
  const presets = buildPresetLibrary(libraryKind, palette, b);
  const learned = learnBlocks(targetLab, cellCols, cellRows, b, learnK);
  const candBlocks: Block[] = [...presets, ...learned];
  const candIsLib: boolean[] = presets.map(() => true).concat(learned.map(() => false));
  if (candBlocks.length === 0) {
    // library:"none" + learn:0 — fall back to solids so we always have a vocabulary.
    const solids = buildPresetLibrary("solids", palette, b);
    for (const s of solids) {
      candBlocks.push(s);
      candIsLib.push(true);
    }
  }
  const candLab = candBlocks.map(blockToOklab);
  const libFactor = 1 - 0.6 * libraryBias; // ≤1 discount on library candidates

  // Paint.
  const outW = cellCols * cell;
  const outH = cellRows * cell;
  const out = new Uint8ClampedArray(outW * outH * 4);
  const work = method === "diffuse" ? Float32Array.from(target) : target;
  const slotLab = new Float32Array(b * b * 3);

  const spread = (tsc: number, tsr: number, wt: number, er: number, eg: number, eb: number): void => {
    if (tsc < 0 || tsc >= cols || tsr < 0 || tsr >= rows) return;
    for (let i = 0; i < b; i++) {
      const cr = tsr * b + i;
      for (let j = 0; j < b; j++) {
        const ti = (cr * cellCols + (tsc * b + j)) * 3;
        work[ti] += er * wt;
        work[ti + 1] += eg * wt;
        work[ti + 2] += eb * wt;
      }
    }
  };

  for (let sr = 0; sr < rows; sr++) {
    for (let sc = 0; sc < cols; sc++) {
      // This slot's b×b cells → OKLab (gamut-clamped, as diffusion may push out).
      for (let i = 0; i < b; i++) {
        const cr = sr * b + i;
        for (let j = 0; j < b; j++) {
          const ti = (cr * cellCols + (sc * b + j)) * 3;
          srgbToOklabInto(
            clamp01(work[ti]),
            clamp01(work[ti + 1]),
            clamp01(work[ti + 2]),
            slotLab,
            (i * b + j) * 3,
          );
        }
      }

      const k = chooseCandidate(slotLab, candLab, candIsLib, libFactor);
      const chosen = candBlocks[k];

      // Stamp the chosen block (each cell → cell×cell solid px).
      for (let i = 0; i < b; i++) {
        const py0 = (sr * b + i) * cell;
        for (let j = 0; j < b; j++) {
          const c = chosen.cells[i * b + j];
          const px0 = (sc * b + j) * cell;
          for (let yy = 0; yy < cell; yy++) {
            let oi = ((py0 + yy) * outW + px0) * 4;
            for (let xx = 0; xx < cell; xx++) {
              out[oi] = c[0];
              out[oi + 1] = c[1];
              out[oi + 2] = c[2];
              out[oi + 3] = 255;
              oi += 4;
            }
          }
        }
      }

      // Diffuse this slot's mean residual (sRGB) onto its neighbours (FS weights).
      if (method === "diffuse") {
        let er = 0;
        let eg = 0;
        let eb = 0;
        for (let i = 0; i < b; i++) {
          const cr = sr * b + i;
          for (let j = 0; j < b; j++) {
            const ti = (cr * cellCols + (sc * b + j)) * 3;
            const c = chosen.cells[i * b + j];
            er += work[ti] - c[0] / 255;
            eg += work[ti + 1] - c[1] / 255;
            eb += work[ti + 2] - c[2] / 255;
          }
        }
        const inv = 1 / (b * b);
        er *= inv;
        eg *= inv;
        eb *= inv;
        spread(sc + 1, sr, 7 / 16, er, eg, eb);
        spread(sc - 1, sr + 1, 3 / 16, er, eg, eb);
        spread(sc, sr + 1, 5 / 16, er, eg, eb);
        spread(sc + 1, sr + 1, 1 / 16, er, eg, eb);
      }
    }
  }

  return rasterToIndexed({ width: outW, height: outH, data: out });
}
