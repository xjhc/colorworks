/* ============================================================================
   glyphfit — represent an image as the best block/braille GLYPH DOCUMENT, then
   render that document back. Glyph-FIRST (unlike repixel's post-hoc toGlyphText):
   the alphabet participates in the fit.

   Per cell of a character GRID (cellW × cellH, phaseX/phaseY): sample the canonical
   2×4 subcells, choose ONE allowed glyph + colour(s), and store the result. Render
   back by expanding every cell to its 2×4 logical pixels.

   Colour models (GLYPHFIT_PLAN §3):
     - "fg_bg" (default, RECOVERY): one fg + one bg per cell — the terminal's own
       model; 2-means the subcell colours, the binary split IS the glyph mask.
     - "per_subpixel" (EXPRESSIVE): each lit subcell keeps its own colour.

   The fit objective (`score`) and the reported metric (`reconError`, render-back
   MSE) are different numbers — see GLYPHFIT_PLAN §0.
   ========================================================================== */
import { type RGB, type Raster, type RenderResult, parseColor } from "./colorworks";
import { rasterToIndexed, gridOrigin } from "./depixelate";
import { detectGrid, globalBg } from "./repixel";
import { type Alphabet, type GlyphKind, maskToBraille, maskToBlock, snapToBlock } from "./glyph_alphabet";

export type ColorModel = "fg_bg" | "per_subpixel";

/** Where the character cells sit on the source: cell size AND phase/origin. */
export interface GlyphGrid {
  cellW: number;
  cellH: number;
  phaseX: number;
  phaseY: number;
}

export interface GlyphCell {
  glyph: string;
  kind: GlyphKind;
  /** Canonical 2×4 mask, length 8, index = row*2 + col. */
  mask: boolean[];
  bg: RGB;
  /** Colour per canonical subpixel (length 8). */
  colors: RGB[];
  /** Single foreground (fg_bg model only). */
  fg?: RGB;
  /** Measured foreground coverage 0..1 per subcell (diagnostic / shape signal). */
  coverage: number[];
  /** Fit loss that chose this glyph (cluster split cost; lower = cleaner). */
  score: number;
  /** Render-back RGB MSE of this cell vs source — the honest per-cell metric. */
  reconError: number;
}

export interface GlyphDocument {
  sourceWidth: number;
  sourceHeight: number;
  grid: GlyphGrid;
  cellSubW: 2;
  cellSubH: 4;
  cols: number;
  rows: number;
  cells: GlyphCell[];
  /** Mean reconError over all cells — drives the readout + failure signal. */
  meanError: number;
}

export interface GlyphfitOptions {
  cellMode?: "auto" | "manual"; // default "auto"
  cellW?: number; // manual cell width  (default 16)
  cellH?: number; // manual cell height (default 32)
  phaseX?: number; // manual x offset (default detected/0)
  phaseY?: number; // manual y offset (default detected/0)
  alphabet?: Alphabet; // default "blocks_braille"
  colorModel?: ColorModel; // default "fg_bg" (recovery)
  tau?: number; // per_subpixel: bg-distance that counts as lit (default 45)
  bgMode?: "auto" | "custom"; // background for per_subpixel / fill (default "auto")
  bgColor?: string; // when bgMode = "custom"
}

const SUB_W = 2;
const SUB_H = 4;
const UNIFORM_TOL = 26; // colour spread below which a cell is treated as solid

const dist3 = (a: number[], b: number[]): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const roundRGB = (c: number[]): RGB => [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])];

interface CellStats {
  means: number[][]; // 8 × [r,g,b]
  count: number[]; // 8
  total: number;
}

/** One pass over a cell window → per-subcell mean colour + pixel count. */
function cellStats(data: Uint8ClampedArray, W: number, H: number, x0: number, y0: number, cellW: number, cellH: number): CellStats {
  const sum = Array.from({ length: 8 }, () => [0, 0, 0]);
  const count = new Array(8).fill(0);
  const sw = cellW / SUB_W;
  const sh = cellH / SUB_H;
  for (let r = 0; r < SUB_H; r++) {
    const ya = Math.floor(y0 + r * sh);
    const yb = Math.min(H, Math.floor(y0 + (r + 1) * sh));
    for (let c = 0; c < SUB_W; c++) {
      const xa = Math.floor(x0 + c * sw);
      const xb = Math.min(W, Math.floor(x0 + (c + 1) * sw));
      const si = r * 2 + c;
      const s = sum[si];
      let n = 0;
      for (let y = Math.max(0, ya); y < yb; y++) {
        const row = y * W;
        for (let x = Math.max(0, xa); x < xb; x++) {
          const i = (row + x) * 4;
          s[0] += data[i]; s[1] += data[i + 1]; s[2] += data[i + 2];
          n++;
        }
      }
      count[si] = n;
    }
  }
  const means = sum.map((s, i) => (count[i] ? [s[0] / count[i], s[1] / count[i], s[2] / count[i]] : [0, 0, 0]));
  const total = count.reduce((a, b) => a + b, 0);
  return { means, count, total };
}

/** 2-means over the 8 subcell means → { fg, bg, mask } (fg = cluster farther from
 *  the global background, so dots/ink read as foreground). `uniform` cells (tiny
 *  colour spread) collapse to a single colour + empty mask. */
function twoMeans(means: number[][], gbg: number[]): { fg: number[]; bg: number[]; mask: boolean[]; score: number; uniform: boolean } {
  const mn = [Math.min(...means.map((m) => m[0])), Math.min(...means.map((m) => m[1])), Math.min(...means.map((m) => m[2]))];
  const mx = [Math.max(...means.map((m) => m[0])), Math.max(...means.map((m) => m[1])), Math.max(...means.map((m) => m[2]))];
  const overall = means.reduce((a, m) => [a[0] + m[0], a[1] + m[1], a[2] + m[2]], [0, 0, 0]).map((v) => v / 8);
  if (dist3(mn, mx) < UNIFORM_TOL) {
    return { fg: overall, bg: overall, mask: new Array(8).fill(false), score: 0, uniform: true };
  }
  // init: darkest mean, and the mean farthest from it
  const lum = means.map((m) => 0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]);
  let a = means[lum.indexOf(Math.min(...lum))].slice();
  const dA = means.map((m) => dist3(m, a));
  let b = means[dA.indexOf(Math.max(...dA))].slice();
  let assignB = new Array(8).fill(false);
  for (let it = 0; it < 8; it++) {
    assignB = means.map((m) => dist3(m, b) < dist3(m, a));
    const acc = (sel: boolean) => {
      const g = means.filter((_, i) => assignB[i] === sel);
      if (!g.length) return null;
      return g.reduce((s, m) => [s[0] + m[0], s[1] + m[1], s[2] + m[2]], [0, 0, 0]).map((v) => v / g.length);
    };
    a = acc(false) ?? a;
    b = acc(true) ?? b;
  }
  // cluster split cost (how cleanly the cell is 2 colours) — the fit `score`
  const score = means.reduce((s, m) => s + Math.min(dist3(m, a), dist3(m, b)) ** 2, 0) / 8;
  const fgIsA = dist3(a, gbg) >= dist3(b, gbg);
  const fg = fgIsA ? a : b;
  const bg = fgIsA ? b : a;
  const mask = means.map((m) => (dist3(m, fg) <= dist3(m, bg)));
  return { fg, bg, mask, score, uniform: false };
}

const BLOCK_BIAS = 0.08; // a clean block char wins when within this fraction of braille

/** Colour each subcell from a mask under the colour model. */
function colorsFor(mask: boolean[], means: number[][], fg: RGB, bg: RGB, colorModel: ColorModel): RGB[] {
  return mask.map((lit, i) => (colorModel === "fg_bg" ? (lit ? fg : bg) : lit ? roundRGB(means[i]) : bg));
}

/** NORMALISED render-back error: mean ‖subcellMean − assignedColour‖² over subcells
 *  with pixels. Compares against the cell-downsampled source, so anti-aliasing
 *  *within* a subcell isn't charged as error (GLYPHFIT_PLAN §0/§4 — raw per-pixel
 *  MSE is diagnostic only). */
function normErr(means: number[][], count: number[], colors: RGB[]): number {
  let e = 0, n = 0;
  for (let i = 0; i < 8; i++) {
    if (!count[i]) continue;
    const m = means[i], c = colors[i];
    e += (m[0] - c[0]) ** 2 + (m[1] - c[1]) ** 2 + (m[2] - c[2]) ** 2;
    n++;
  }
  return n ? e / (n * 3) : 0;
}

interface GlyphCand { mask: boolean[]; char: string; kind: GlyphKind; }

function fitGlyphCell(
  data: Uint8ClampedArray, W: number, H: number, x0: number, y0: number,
  cellW: number, cellH: number, alphabet: Alphabet, colorModel: ColorModel,
  gbg: number[], cellBg: number[] | null, tau: number,
): GlyphCell {
  const st = cellStats(data, W, H, x0, y0, cellW, cellH);
  const base = cellBg ?? gbg;
  const coverage = st.means.map((m) => Math.min(1, dist3(m, base) / Math.max(1, tau)));

  // binary subcell mask + the two colours this cell is built from
  let rawMask: boolean[];
  let fg: RGB | undefined;
  let bg: RGB;
  let score: number;
  if (colorModel === "fg_bg") {
    const tm = twoMeans(st.means, gbg);
    rawMask = tm.mask;
    fg = roundRGB(tm.fg);
    bg = roundRGB(tm.bg);
    score = tm.score;
  } else {
    rawMask = st.means.map((m) => dist3(m, base) > tau);
    bg = roundRGB(base);
    fg = undefined;
    score = st.means.reduce((s, m) => s + Math.abs(dist3(m, base) - tau), 0) / 8;
  }

  // Compete the allowed glyphs and keep the lowest normalised error (a block char
  // gets a small bias so it wins ties vs an equivalent braille — cleaner recovery).
  // block first so it wins ties (the BLOCK_BIAS): braille must be meaningfully
  // better to displace a clean block char.
  const cands: GlyphCand[] = [];
  if (alphabet !== "braille") {
    const sm = snapToBlock(rawMask);
    cands.push({ mask: sm, char: maskToBlock(sm), kind: "block2x2" });
  }
  if (alphabet !== "blocks") cands.push({ mask: rawMask, char: maskToBraille(rawMask), kind: "braille2x4" });
  let best: { cand: GlyphCand; colors: RGB[]; adj: number } | null = null;
  for (const cand of cands) {
    const colors = colorsFor(cand.mask, st.means, fg as RGB, bg, colorModel);
    const adj = normErr(st.means, st.count, colors) * (cand.kind === "block2x2" ? 1 - BLOCK_BIAS : 1);
    if (!best || adj < best.adj) best = { cand, colors, adj };
  }
  const chosen = best!;
  const reconError = normErr(st.means, st.count, chosen.colors); // unbiased, reported
  return { glyph: chosen.cand.char, kind: chosen.cand.kind, mask: chosen.cand.mask, bg, colors: chosen.colors, fg, coverage, score, reconError };
}

/** Resolve the character grid: explicit (manual) or detected (auto, from the fine
 *  lattice — a char cell is 2×4 dots, so cell = 2·pitchX × 4·pitchY). */
export function resolveGrid(raster: Raster, opts: GlyphfitOptions): GlyphGrid {
  if ((opts.cellMode ?? "auto") === "manual") {
    return {
      cellW: Math.max(2, Math.round(opts.cellW ?? 16)),
      cellH: Math.max(4, Math.round(opts.cellH ?? 32)),
      phaseX: Math.max(0, Math.round(opts.phaseX ?? 0)),
      phaseY: Math.max(0, Math.round(opts.phaseY ?? 0)),
    };
  }
  const g = detectGrid(raster);
  const cellW = Math.max(2, Math.round(g.pitchX * SUB_W));
  const cellH = Math.max(4, Math.round(g.pitchY * SUB_H));
  const [ox, oy] = gridOrigin(g);
  return {
    cellW,
    cellH,
    phaseX: ((Math.round(ox) % cellW) + cellW) % cellW,
    phaseY: ((Math.round(oy) % cellH) + cellH) % cellH,
  };
}

/** Fit the whole image to a glyph document. */
export function fitGlyphDocument(raster: Raster, opts: GlyphfitOptions = {}): GlyphDocument {
  const { width: W, height: H, data } = raster;
  const alphabet = opts.alphabet ?? "blocks_braille";
  const colorModel = opts.colorModel ?? "fg_bg";
  const tau = opts.tau ?? 45;
  const gbg: number[] = opts.bgMode === "custom" ? parseColor(opts.bgColor ?? "#181818") : globalBg(raster);
  const cellBg = opts.bgMode === "custom" ? gbg : null; // per_subpixel uses gbg if not custom
  // Grid from the lattice detector (size AND phase). A reconstruction-error grid
  // SEARCH was tried and rejected: on a sparse frame its argmin drifts ±1px off the
  // true cell / misaligns the phase for noise-level gains (see GLYPHFIT_PLAN §5).
  // Error drives the residual + readout, not the grid choice.
  const grid = resolveGrid(raster, opts);
  const { cellW, cellH, phaseX, phaseY } = grid;
  const cols = Math.max(0, Math.floor((W - phaseX) / cellW));
  const rows = Math.max(0, Math.floor((H - phaseY) / cellH));

  const cells: GlyphCell[] = new Array(cols * rows);
  let errSum = 0;
  for (let ry = 0; ry < rows; ry++) {
    const y0 = phaseY + ry * cellH;
    for (let cx = 0; cx < cols; cx++) {
      const x0 = phaseX + cx * cellW;
      const cell = fitGlyphCell(data, W, H, x0, y0, cellW, cellH, alphabet, colorModel, gbg, cellBg, tau);
      cells[ry * cols + cx] = cell;
      errSum += cell.reconError;
    }
  }
  return {
    sourceWidth: W,
    sourceHeight: H,
    grid,
    cellSubW: 2,
    cellSubH: 4,
    cols,
    rows,
    cells,
    meanError: cols * rows ? errSum / (cols * rows) : 0,
  };
}

/** Expand the document to logical pixels: every cell → its 2×4 subcells, each
 *  painted its stored colour. Native size = cols·2 × rows·4 (one px per subcell).
 *  Returned indexed so the studio paints it; output-size upscaling happens later. */
export function renderGlyphDocument(doc: GlyphDocument): RenderResult {
  const W = doc.cols * SUB_W;
  const Hh = doc.rows * SUB_H;
  const data = new Uint8ClampedArray(W * Hh * 4);
  for (let ry = 0; ry < doc.rows; ry++) {
    for (let cx = 0; cx < doc.cols; cx++) {
      const cell = doc.cells[ry * doc.cols + cx];
      for (let r = 0; r < SUB_H; r++) {
        for (let c = 0; c < SUB_W; c++) {
          const px = cx * SUB_W + c;
          const py = ry * SUB_H + r;
          const o = (py * W + px) * 4;
          const col = cell.colors[r * 2 + c];
          data[o] = col[0]; data[o + 1] = col[1]; data[o + 2] = col[2]; data[o + 3] = 255;
        }
      }
    }
  }
  return rasterToIndexed({ width: W, height: Hh, data });
}

const HEAT_SCALE = 80; // RMS reconError mapped to full-red at this colour distance

/** Residual heatmap (same native size as the render): each cell's `reconError` →
 *  a colour (dark green = low, red = high), so the user can SEE where the glyph
 *  alphabet fails (GLYPHFIT_PLAN §0 — make the remaining error visible). */
export function renderGlyphResidual(doc: GlyphDocument): RenderResult {
  const W = doc.cols * SUB_W;
  const Hh = doc.rows * SUB_H;
  const data = new Uint8ClampedArray(W * Hh * 4);
  for (let ry = 0; ry < doc.rows; ry++) {
    for (let cx = 0; cx < doc.cols; cx++) {
      const cell = doc.cells[ry * doc.cols + cx];
      const t = Math.min(1, Math.sqrt(cell.reconError) / HEAT_SCALE);
      const q = Math.round(t * 15) / 15; // 16 levels → small palette
      const col: RGB = [Math.round(q * 255), Math.round((1 - q) * 45), 24];
      for (let r = 0; r < SUB_H; r++) {
        for (let c = 0; c < SUB_W; c++) {
          const o = ((ry * SUB_H + r) * W + (cx * SUB_W + c)) * 4;
          data[o] = col[0]; data[o + 1] = col[1]; data[o + 2] = col[2]; data[o + 3] = 255;
        }
      }
    }
  }
  return rasterToIndexed({ width: W, height: Hh, data });
}

// ── exports ─────────────────────────────────────────────────────────────────

/** Glyph characters only — SHAPE layer, lossy re colour (see GLYPHFIT_PLAN §2). */
export function glyphDocumentToText(doc: GlyphDocument): string {
  let out = "";
  for (let ry = 0; ry < doc.rows; ry++) {
    let row = "";
    for (let cx = 0; cx < doc.cols; cx++) row += doc.cells[ry * doc.cols + cx].glyph;
    out += row.replace(/\s+$/, "") + "\n";
  }
  return out;
}

/** Faithful glyph document (shape + colour) as JSON. */
export function glyphDocumentToJSON(doc: GlyphDocument): string {
  return JSON.stringify(doc);
}

/** Parse a glyph document back from JSON — round-trips to the same render. */
export function glyphDocumentFromJSON(json: string): GlyphDocument {
  return JSON.parse(json) as GlyphDocument;
}
