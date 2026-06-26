/* ============================================================================
   Knit chart — turn an image into a stranded-colorwork chart.

   One pixel = one stitch. The image is resampled onto a stitch grid (the
   shared chart-width setting drives it), quantised to a small fixed YARN palette, and
   dithered — which produces the two colorwork regimes for free: a flat-tone
   region error-diffuses into alternating "salt-and-pepper" stitches (a blended
   mid-tone you knit by carrying two yarns), while a saturated region collapses
   to solid runs of one yarn. The dithering already *is* the knit texture.

   Knit-specific concerns the plain dither renderer doesn't handle:
     - Gauge: a stockinette stitch is wider than it is tall (more rows/inch than
       sts/inch), so the row count is inflated by `rowGauge` (rows÷sts, default
       1.4). Chart cells stay square — the conventional chart look — and the
       knitted fabric comes out matching the source proportions.
     - Tidy: an optional cleanup pass merges fully isolated single stitches into
       a dominant surrounding colour. It is off by default because the faintest
       deliberate background dots are locally indistinguishable from strays.
     - A baked grid (thin lines, bold every 10) so the result reads as a chart,
       plus a yarn legend, per-yarn counts, and row-level stranding stats.

   Pure (raster, opts) → KnitResult (a RenderResult plus chart stats), mirroring
   the other studio renderers. No DOM, no fetch — safe in vitest.
   ========================================================================== */
import {
  buildTonePalette,
  rasterToRgb01,
  rasterToGray,
  applyTone,
  ditherToPalette,
  errorDiffuseToPalette,
  yliluomaToPalette,
  DIFFUSION_KERNELS,
  type PaletteMode,
  type PinnedColor,
  type Raster,
  type RGB,
  type RenderResult,
} from "./colorworks";

/** Dither engine for the stitch grid. "flat" = solid blocks only (no tone). */
export type KnitMethod = "floyd_steinberg" | "atkinson" | "stucki" | "yliluoma" | "flat";

/** Chart grid overlay: none, thin lines, or thin lines + bold every-10 decades. */
export type KnitGrid = "none" | "lines" | "decades";

/** How the studio should annotate yarn symbols. */
export type KnitSymbolMode = "cells" | "legend" | "none";

export interface KnitOptions {
  stitches?: number; // sts across, 40..400 (drives the whole grid)
  rows?: number; // optional exact rows; otherwise derived from source aspect × gauge
  rowGauge?: number; // rows÷sts gauge ratio, 0.8..2.0 (1.0 = square chart)
  cell?: number; // output px per stitch, 4..16
  colors?: number; // yarn count, 2..8
  palette?: PaletteMode; // grayscale | adaptive | duotone
  inkColor?: string; // duotone dark
  paperColor?: string; // duotone light
  method?: KnitMethod; // dither engine (default floyd_steinberg)
  symbols?: KnitSymbolMode; // draw symbols in cells, legend only, or hide them
  contrast?: number;
  midpoint?: number;
  tidy?: boolean; // merge isolated single stitches (default false)
  grid?: KnitGrid; // chart grid overlay (default "decades")
  pins?: PinnedColor[]; // steer the yarn palette (lock / boost / exclude)
}

/** One yarn in the legend: its swatch, palette index, and stitch count. */
export interface Yarn {
  index: number;
  rgb: RGB;
  count: number;
  symbol: string;
}

/** Row-level knitting risk readout, computed from the stitch grid before paint. */
export interface StrandingStats {
  longestSolidRun: number; // longest run of one colour within a row
  maxFloat: number; // longest gap between repeats of a row-carried yarn
  maxYarnsPerRow: number;
  rowsWithThreePlusYarns: number;
  rowsWithLongFloats: number; // rows whose max float exceeds LONG_FLOAT_WARN
}

/** RenderResult (paint with palette[indices[i]]) plus chart stats for the UI. */
export interface KnitResult extends RenderResult {
  stitchesWide: number;
  rowsTall: number;
  cellSize: number;
  stitchIndices: Uint16Array; // cols×rows, before cell expansion and chart grid
  yarns: Yarn[]; // sorted most-used first (main colour leads), grid excluded
  symbols: string[];
  stranding: StrandingStats;
  longestRun: number; // legacy alias for stranding.longestSolidRun
}

const GRID_RGB: RGB = [82, 82, 82]; // neutral grid line colour
const SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H"];
const LONG_FLOAT_WARN = 7;

// ── stitch grid ─────────────────────────────────────────────────────────────

/** Box-average a tone-mapped sRGB raster (0..1) into a cols×rows stitch grid. */
function downsampleToStitches(
  rgb01: Float32Array,
  W: number,
  H: number,
  cols: number,
  rows: number,
): Float32Array {
  const out = new Float32Array(cols * rows * 3);
  for (let sy = 0; sy < rows; sy++) {
    const y0 = Math.floor((sy * H) / rows);
    const y1 = Math.max(y0 + 1, Math.floor(((sy + 1) * H) / rows));
    for (let sx = 0; sx < cols; sx++) {
      const x0 = Math.floor((sx * W) / cols);
      const x1 = Math.max(x0 + 1, Math.floor(((sx + 1) * W) / cols));
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let y = y0; y < y1 && y < H; y++) {
        const row = y * W;
        for (let x = x0; x < x1 && x < W; x++) {
          const i = (row + x) * 3;
          r += rgb01[i];
          g += rgb01[i + 1];
          b += rgb01[i + 2];
          n++;
        }
      }
      const o = (sy * cols + sx) * 3;
      const inv = n > 0 ? 1 / n : 0;
      out[o] = r * inv;
      out[o + 1] = g * inv;
      out[o + 2] = b * inv;
    }
  }
  return out;
}

/** Dither the stitch grid to the yarn palette with the chosen engine. */
export function ditherStitches(
  rgb01: Float32Array,
  cols: number,
  rows: number,
  palette: RGB[],
  method: KnitMethod,
): Uint16Array {
  if (method === "flat") return ditherToPalette(rgb01, cols, rows, palette, null);
  if (method === "yliluoma") return yliluomaToPalette(rgb01, cols, rows, palette, 8);
  const kernel = DIFFUSION_KERNELS[method] ?? DIFFUSION_KERNELS.floyd_steinberg;
  return errorDiffuseToPalette(rgb01, cols, rows, palette, kernel);
}

// ── knit-aware cleanup + stats ──────────────────────────────────────────────

/**
 * Merge fully-isolated single stitches into their surroundings: a stitch whose
 * colour appears in NONE of its 8 neighbours, where one other colour dominates
 * (≥5 of 8), is replaced by that colour. Anything with even one same-colour
 * neighbour — a domino, a checkerboard cell (its diagonals match), ordinary
 * dither texture — is left untouched.
 *
 * NOTE this is opt-in (default off): a deliberate sparse contrast dot on a field
 * is locally identical to a stray, so this also erases the faintest 1-stitch
 * dots. Use it when you want a cleaner, more solid chart, not the dotted look.
 */
export function tidyStrays(idx: Uint16Array, cols: number, rows: number): Uint16Array {
  const out = Uint16Array.from(idx);
  const nb = new Int32Array(8);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const p = y * cols + x;
      const c = idx[p];
      let m = 0; // count of non-self neighbours gathered in nb
      let self = 0; // count of neighbours equal to this stitch
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= rows) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= cols) continue;
          const v = idx[ny * cols + nx];
          if (v === c) self++;
          else nb[m++] = v;
        }
      }
      if (self > 0) continue; // not isolated — leave it
      let best = -1;
      let bestCount = 0;
      for (let a = 0; a < m; a++) {
        let k = 0;
        for (let b = 0; b < m; b++) if (nb[b] === nb[a]) k++;
        if (k > bestCount) {
          bestCount = k;
          best = nb[a];
        }
      }
      if (best >= 0 && bestCount >= 5) out[p] = best;
    }
  }
  return out;
}

/** Longest run of one colour within any single row — a stranding float proxy. */
export function longestRun(idx: Uint16Array, cols: number, rows: number): number {
  let mx = 0;
  for (let y = 0; y < rows; y++) {
    let run = 0;
    let prev = -1;
    for (let x = 0; x < cols; x++) {
      const v = idx[y * cols + x];
      if (v === prev) run++;
      else {
        prev = v;
        run = 1;
      }
      if (run > mx) mx = run;
    }
  }
  return mx;
}

/** Row-level risk stats for stranded knitting. Edges are not counted as floats:
 *  a float is only the gap between two uses of a yarn in the same row. */
export function analyzeStranding(
  idx: Uint16Array,
  cols: number,
  rows: number,
  yarnCount: number,
): StrandingStats {
  let longestSolidRun = 0;
  let maxFloat = 0;
  let maxYarnsPerRow = 0;
  let rowsWithThreePlusYarns = 0;
  let rowsWithLongFloats = 0;
  const lastSeen = new Int32Array(yarnCount);
  const seen = new Uint8Array(yarnCount);

  for (let y = 0; y < rows; y++) {
    lastSeen.fill(-1);
    seen.fill(0);
    let rowYarns = 0;
    let rowMaxFloat = 0;
    let run = 0;
    let prev = -1;

    for (let x = 0; x < cols; x++) {
      const v = idx[y * cols + x];
      if (v === prev) run++;
      else {
        prev = v;
        run = 1;
      }
      if (run > longestSolidRun) longestSolidRun = run;

      if (v < yarnCount) {
        if (seen[v] === 0) {
          seen[v] = 1;
          rowYarns++;
        } else {
          const gap = x - lastSeen[v] - 1;
          if (gap > rowMaxFloat) rowMaxFloat = gap;
        }
        lastSeen[v] = x;
      }
    }

    if (rowYarns > maxYarnsPerRow) maxYarnsPerRow = rowYarns;
    if (rowYarns >= 3) rowsWithThreePlusYarns++;
    if (rowMaxFloat > maxFloat) maxFloat = rowMaxFloat;
    if (rowMaxFloat > LONG_FLOAT_WARN) rowsWithLongFloats++;
  }

  return { longestSolidRun, maxFloat, maxYarnsPerRow, rowsWithThreePlusYarns, rowsWithLongFloats };
}

/**
 * Drop unused yarns and reorder by descending stitch count (so palette[0] is the
 * main colour), remapping `idx`. Returns the remapped indices, the pruned/sorted
 * palette, and the matching counts.
 */
export function pruneSortYarns(
  idx: Uint16Array,
  palette: RGB[],
): { idx: Uint16Array; palette: RGB[]; counts: number[] } {
  const counts = new Array(palette.length).fill(0);
  for (let i = 0; i < idx.length; i++) counts[idx[i]]++;
  const used = palette
    .map((rgb, i) => ({ rgb, i, count: counts[i] }))
    .filter((o) => o.count > 0)
    .sort((a, b) => b.count - a.count || a.i - b.i);
  const remap = new Int32Array(palette.length).fill(-1);
  used.forEach((o, newIdx) => (remap[o.i] = newIdx));
  const outIdx = new Uint16Array(idx.length);
  for (let i = 0; i < idx.length; i++) outIdx[i] = remap[idx[i]];
  return { idx: outIdx, palette: used.map((o) => o.rgb), counts: used.map((o) => o.count) };
}

// ── chart raster ────────────────────────────────────────────────────────────

/** Paint grid lines into the chart; decade boundaries (every 10) get bold (3px). */
function paintGrid(
  out: Uint16Array,
  outW: number,
  outH: number,
  cols: number,
  rows: number,
  cell: number,
  gridIdx: number,
  decades: boolean,
): void {
  const setCol = (x: number) => {
    if (x < 0 || x >= outW) return;
    for (let y = 0; y < outH; y++) out[y * outW + x] = gridIdx;
  };
  const setRow = (y: number) => {
    if (y < 0 || y >= outH) return;
    for (let x = 0; x < outW; x++) out[x + y * outW] = gridIdx;
  };
  for (let sc = 0; sc <= cols; sc++) {
    const x = Math.min(sc * cell, outW - 1);
    setCol(x);
    if (decades && sc % 10 === 0) {
      setCol(x - 1);
      setCol(x + 1 < outW ? x + 1 : outW - 1);
    }
  }
  for (let sr = 0; sr <= rows; sr++) {
    const y = Math.min(sr * cell, outH - 1);
    setRow(y);
    if (decades && sr % 10 === 0) {
      setRow(y - 1);
      setRow(y + 1 < outH ? y + 1 : outH - 1);
    }
  }
}

// ── top-level render ────────────────────────────────────────────────────────

const PALETTE_MODES: PaletteMode[] = ["grayscale", "adaptive", "duotone"];
const KNIT_METHODS: KnitMethod[] = ["floyd_steinberg", "atkinson", "stucki", "yliluoma", "flat"];
const SYMBOL_MODES: KnitSymbolMode[] = ["cells", "legend", "none"];

/**
 * Render `raster` as a knitting chart. Runs on the full-resolution raster (it
 * owns its own stitch resolution, driven by the studio chart-size controls) and
 * returns a `KnitResult`: a `{ indices, palette }` you paint normally, plus the
 * stitch dimensions, yarn legend, symbols, and stranding stats.
 */
export function renderKnit(raster: Raster, opts: KnitOptions = {}): KnitResult {
  const { width: W, height: H } = raster;
  const stsW = Math.max(8, Math.min(400, Math.floor(opts.stitches ?? 200)));
  const exactRows =
    typeof opts.rows === "number" && Number.isFinite(opts.rows)
      ? Math.max(1, Math.min(800, Math.floor(opts.rows)))
      : null;
  const rowGauge = Math.max(0.5, Math.min(2.5, opts.rowGauge ?? 1.4));
  const cell = Math.max(2, Math.min(16, Math.floor(opts.cell ?? 8)));
  const colors = Math.max(2, Math.min(8, Math.floor(opts.colors ?? 5)));
  const palMode: PaletteMode = PALETTE_MODES.includes(opts.palette as PaletteMode)
    ? (opts.palette as PaletteMode)
    : "adaptive";
  const method: KnitMethod = KNIT_METHODS.includes(opts.method as KnitMethod)
    ? (opts.method as KnitMethod)
    : "floyd_steinberg";
  const symbols: KnitSymbolMode = SYMBOL_MODES.includes(opts.symbols as KnitSymbolMode)
    ? (opts.symbols as KnitSymbolMode)
    : "cells";
  const contrast = opts.contrast ?? 1;
  const midpoint = opts.midpoint ?? 0.5;
  const tidy = opts.tidy === true; // opt-in: also erases deliberate sparse dots
  const grid: KnitGrid = opts.grid ?? "decades";

  // Gauge-corrected row count: more rows than the pixel aspect implies, so the
  // knitted (taller-than-wide-stitch) fabric matches the source proportions.
  const cols = stsW;
  const rows = exactRows ?? Math.max(1, Math.min(800, Math.round((stsW * H * rowGauge) / W)));

  // Tone-map the source, then box-average down to the stitch grid.
  const rgb01 = rasterToRgb01(raster);
  const gray = rasterToGray(raster);
  applyTone(rgb01, gray, contrast, midpoint);
  const stitchRgb = downsampleToStitches(rgb01, W, H, cols, rows);

  // Yarn palette from the full-res image (best adaptive quality), then dither.
  const yarnPalette = buildTonePalette(
    raster,
    colors,
    palMode,
    opts.inkColor ?? "#161616",
    opts.paperColor ?? "#f4ebd9",
    0,
    opts.pins,
  );
  let stitchIdx = ditherStitches(stitchRgb, cols, rows, yarnPalette, method);
  if (tidy) stitchIdx = tidyStrays(stitchIdx, cols, rows);

  // Prune unused yarns, sort main-colour-first, gather stats.
  const pruned = pruneSortYarns(stitchIdx, yarnPalette);
  stitchIdx = pruned.idx;
  const yarnSymbols = pruned.palette.map((_, i) => SYMBOLS[i] ?? String(i + 1));
  const yarns: Yarn[] = pruned.palette.map((rgb, i) => ({
    index: i,
    rgb,
    count: pruned.counts[i],
    symbol: yarnSymbols[i],
  }));
  const stranding = analyzeStranding(stitchIdx, cols, rows, pruned.palette.length);

  // Expand each stitch to a cell×cell block; overlay the chart grid.
  const drawGrid = grid !== "none";
  const palette: RGB[] = drawGrid ? [...pruned.palette, GRID_RGB] : pruned.palette.slice();
  const gridIdx = pruned.palette.length;
  const outW = cols * cell;
  const outH = rows * cell;
  const out = new Uint16Array(outW * outH);
  for (let sr = 0; sr < rows; sr++) {
    const py0 = sr * cell;
    for (let sc = 0; sc < cols; sc++) {
      const v = stitchIdx[sr * cols + sc];
      const px0 = sc * cell;
      for (let yy = 0; yy < cell; yy++) {
        let oi = (py0 + yy) * outW + px0;
        for (let xx = 0; xx < cell; xx++) out[oi++] = v;
      }
    }
  }
  if (drawGrid) paintGrid(out, outW, outH, cols, rows, cell, gridIdx, grid === "decades");

  return {
    indices: out,
    palette,
    width: outW,
    height: outH,
    stitchesWide: cols,
    rowsTall: rows,
    cellSize: cell,
    stitchIndices: stitchIdx,
    yarns,
    symbols: symbols === "none" ? [] : yarnSymbols,
    stranding,
    longestRun: stranding.longestSolidRun,
  };
}
