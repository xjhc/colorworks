/* ============================================================================
   Repixel — recover pixel-art from "glyph art": terminal/TUI screenshots that
   draw with a MIX of block / half-block characters (chunky foreground) and
   Unicode braille characters (2x4 dot fields, for fine dithered backgrounds).

   Both block edges and braille dots land on one fine sub-cell lattice (a
   half-block pixel == 2x2 braille dots), so "block vs braille" collapses to a
   single question — which fine cells are LIT. We detect that lattice and emit
   ONE output pixel per cell, painted with its recovered colour (foreground if
   lit, else background). The result is the true logical pixel-art bitmap:
   blocks become solid runs, braille dithering returns to a real-pixel dot field.

   Position-preserving by construction (unlike depixelate's coverage re-dither):
   a lit cell's colour is sampled where the dot/block actually sits.

   Detection reuses depixelate's machinery (`fundamental`, `bestPhase`,
   `gridOrigin`, `windowMode`, `dist`, `quantizeToPalette`, `rasterToIndexed`)
   but feeds it a LUMINANCE-ONLY edge profile: depixelate saturation-weights its
   edges, which would suppress the grey/white braille grid (saturation ~= 0) and
   lock onto the coarser colourful-block pitch instead.
   ========================================================================== */
import {
  buildTonePalette,
  kmeansPalette,
  parseColor,
  type PaletteMode,
  type PinnedColor,
  type Raster,
  type RGB,
  type RenderResult,
} from "./colorworks";
import {
  bestPhase,
  dist,
  fundamental,
  type Grid,
  gridOrigin,
  quantizeToPalette,
  rasterToIndexed,
  windowMode,
} from "./depixelate";

/** "original" keeps the recovered source colours; the others quantise to a palette. */
export type RepixelPalette = "original" | PaletteMode;

export type BgMode = "auto" | "custom";

/** Which pixel scale to lock onto — the image can carry more than one:
 *  - "fine":    luminance detector → the dense glyph/braille lattice (background).
 *  - "subject": saturation detector → the dominant colour sprite (foreground).
 *  - "manual":  use the `pitch` override.
 *  - "composite": BOTH at once — recover the background on the fine lattice and the
 *                 colour sprite on its own (coarser) subject grid, then composite the
 *                 sprite over the background. The fine grid is finer than the sprite
 *                 pixel, so the overlay is loss-free.
 *  There is no single "real" pixel size when a sprite and a dither field coexist;
 *  the caller picks which scale this render targets. */
export type PixelTarget = "fine" | "subject" | "manual" | "composite";

export interface RepixelOptions {
  target?: PixelTarget; // which scale to detect (default "fine"; "manual" uses pitch)
  pitch?: number; // manual pitch; >0 also forces manual when target is unset
  /** Shade each cell by its dot COVERAGE (size) — recovers size-modulated halftone
   *  gradients (e.g. a dotted sphere) instead of collapsing them to a flat blob.
   *  Solid cells (coverage≈1) keep full colour; sparse dots dim toward bg. Default true. */
  shade?: boolean;
  tau?: number; // colour distance from background that counts as "lit" (default 45)
  minLit?: number; // min lit pixels in a cell window to call it foreground (default 2)
  palette?: RepixelPalette; // colour treatment (default "original")
  colors?: number; // palette size when quantising (default 4)
  inkColor?: string; // duotone dark
  paperColor?: string; // duotone light
  bgMode?: BgMode; // "auto" detects the modal colour; "custom" uses bgColor (default "auto")
  bgColor?: string; // background when bgMode = "custom" (default "#181818")
  // composite target only:
  spriteSat?: number; // relative saturation (max-min)/max above which a pixel is "sprite" (default 0.3)
  eyeLuma?: number; // luma below which a body-interior pixel is a dark eye (default 45)
  pins?: PinnedColor[]; // steer the foreground palette (lock / boost / exclude)
}

// ── grid detection ──────────────────────────────────────────────────────────

/** Per-axis edge-energy profiles from the LUMINANCE gradient only (no saturation
 *  weighting). The dense braille grid is mostly grey, so weighting would zero it
 *  out; luminance keeps it the dominant periodic signal and pins the fine pitch. */
function lumaEdgeProfiles(r: Raster): { px: Float64Array; py: Float64Array } {
  const { width: W, height: H, data } = r;
  const lum = new Float32Array(W * H);
  for (let p = 0, i = 0; p < W * H; p++, i += 4) {
    lum[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  const px = new Float64Array(W - 1);
  const py = new Float64Array(H - 1);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W - 1; x++) px[x] += Math.abs(lum[row + x] - lum[row + x + 1]);
  }
  for (let y = 0; y < H - 1; y++) {
    const row = y * W;
    const row2 = row + W;
    for (let x = 0; x < W; x++) py[y] += Math.abs(lum[row + x] - lum[row2 + x]);
  }
  return { px, py };
}

/** Per-axis edge-energy profiles weighted by SATURATION, so a colourful sprite
 *  drawn on a grey/dither field dominates and its (often coarser) pixel pitch is
 *  what gets detected — the complement of the luminance profile. */
function satEdgeProfiles(r: Raster): { px: Float64Array; py: Float64Array } {
  const { width: W, height: H, data } = r;
  const lum = new Float32Array(W * H);
  const sat = new Float32Array(W * H);
  for (let p = 0, i = 0; p < W * H; p++, i += 4) {
    lum[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sat[p] = Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]);
  }
  const px = new Float64Array(W - 1);
  const py = new Float64Array(H - 1);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W - 1; x++) {
      px[x] += Math.abs(lum[row + x] - lum[row + x + 1]) * Math.max(sat[row + x], sat[row + x + 1]);
    }
  }
  for (let y = 0; y < H - 1; y++) {
    const row = y * W;
    const row2 = row + W;
    for (let x = 0; x < W; x++) {
      py[y] += Math.abs(lum[row + x] - lum[row2 + x]) * Math.max(sat[row + x], sat[row2 + x]);
    }
  }
  return { px, py };
}

/** Sub-pixel pitch on a single axis: maximise the comb score within +/-15% of coarse. */
function refineAxis(prof: Float64Array, coarse: number): number {
  const lo = coarse * 0.85;
  const hi = coarse * 1.15;
  let best = -Infinity;
  let bestP = coarse;
  for (let p = lo; p < hi; p += 0.05) {
    const s = bestPhase(prof, p).score;
    if (s > best) {
      best = s;
      bestP = p;
    }
  }
  return bestP;
}

/** Build a per-axis grid (not forced square) from given edge profiles. */
function gridFromProfiles(
  px: Float64Array,
  py: Float64Array,
  minPitch: number,
  maxPitch: number,
): Grid {
  const pitchX = refineAxis(px, fundamental(px, minPitch, maxPitch));
  const pitchY = refineAxis(py, fundamental(py, minPitch, maxPitch));
  return {
    pitchX,
    pitchY,
    phaseX: bestPhase(px, pitchX).phase,
    phaseY: bestPhase(py, pitchY).phase,
    confidence: 1,
  };
}

/** Detect the FINE glyph/braille lattice (luminance profile). Bounds [4,16] keep
 *  `fundamental`'s finest-peak bias from returning a 2x/4x character-cell harmonic. */
export function detectGrid(r: Raster, minPitch = 4, maxPitch = 16): Grid {
  const { px, py } = lumaEdgeProfiles(r);
  return gridFromProfiles(px, py, minPitch, maxPitch);
}

/** Detect the COLOUR SUBJECT pitch (saturation profile) — the foreground sprite,
 *  whose native pixel is usually coarser than the background lattice. Wider bounds
 *  [4,40] allow the larger sprite pixel. */
export function detectSubjectGrid(r: Raster, minPitch = 4, maxPitch = 40): Grid {
  const { px, py } = satEdgeProfiles(r);
  return gridFromProfiles(px, py, minPitch, maxPitch);
}

/** Both candidate pixel sizes for the UI readout (mean of the two axes each). */
export function detectCandidates(r: Raster): { fine: number; subject: number } {
  const f = detectGrid(r);
  const s = detectSubjectGrid(r);
  return { fine: (f.pitchX + f.pitchY) / 2, subject: (s.pitchX + s.pitchY) / 2 };
}

// ── render ──────────────────────────────────────────────────────────────────

/** The most common colour over the whole frame (the background, ~95% on the
 *  target screenshots). Reuses depixelate's modal sampler. */
function globalBg(r: Raster): RGB {
  return windowMode(r.data, r.width, 0, r.width - 1, 0, r.height - 1).colour;
}

/** Mean of the cell's foreground *cluster* — the non-bg pixels within `tau` of the
 *  modal `anchor`. This stays crisp at colour edges (a cell straddling two colours
 *  keeps the majority colour, not a muddy blend, and neighbour-bleed from the wide
 *  window is excluded) while averaging out the noise/jitter that a bare modal pick
 *  suffers on tiny (~2px) anti-aliased braille dots. Measured ~13% lower per-pixel
 *  reconstruction error than the modal, and far more stable colours on dot fields. */
function clusterMean(
  data: Uint8ClampedArray,
  W: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  bg: RGB,
  anchor: RGB,
  tau: number,
): RGB {
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let n = 0;
  for (let y = y0; y <= y1; y++) {
    const row = y * W;
    for (let x = x0; x <= x1; x++) {
      const i = (row + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      if (dist(r, g, b, bg) > tau && dist(r, g, b, anchor) <= tau) {
        sr += r;
        sg += g;
        sb += b;
        n++;
      }
    }
  }
  return n ? [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] : anchor;
}

/** Plain mean over EVERY pixel in the window (incl. background) — the area-average
 *  used by shade mode. For a size-modulated halftone this maps dot size → tone
 *  (small dot ≈ bg, big/solid ≈ full ink), the textbook way to downsample halftone. */
function windowMean(data: Uint8ClampedArray, W: number, x0: number, x1: number, y0: number, y1: number): RGB {
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let n = 0;
  for (let y = y0; y <= y1; y++) {
    const row = y * W;
    for (let x = x0; x <= x1; x++) {
      const i = (row + x) * 4;
      sr += data[i];
      sg += data[i + 1];
      sb += data[i + 2];
      n++;
    }
  }
  return n ? [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] : [0, 0, 0];
}

/** Adaptive palette built from the recovered FOREGROUND colours (+ a reserved
 *  background slot). The generic adaptive path runs k-means over the whole bitmap,
 *  which is ~91% background here and so wastes nearly every slot on near-blacks;
 *  clustering the lit colours instead spends all slots on the real ink. */
function foregroundPalette(litColors: RGB[], bg: RGB, colors: number, pins?: PinnedColor[]): RGB[] {
  if (litColors.length === 0) return [bg];
  const n = litColors.length;
  const data = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    data[o] = litColors[i][0];
    data[o + 1] = litColors[i][1];
    data[o + 2] = litColors[i][2];
    data[o + 3] = 255;
  }
  const fg = kmeansPalette({ width: n, height: 1, data }, Math.max(2, colors - 1), 42, 16, pins);
  return [bg, ...fg];
}

interface Recovery {
  width: number; // native grid columns
  height: number; // native grid rows
  data: Uint8ClampedArray; // RGBA, one cell per pixel
  coverage: Float32Array; // per-cell dot fill fraction 0..1 (0 = background)
  bg: RGB;
  litColors: RGB[];
}

/** Core recovery: detect the grid, sample each cell → colour + coverage. */
function recover(r: Raster, opts: RepixelOptions): Recovery {
  const tau = opts.tau ?? 45;
  const minLit = opts.minLit ?? 2;
  const bgMode = opts.bgMode ?? "auto";
  const shade = opts.shade ?? true;
  const hasPitch = !!(opts.pitch && opts.pitch > 0);
  // Default target: explicit, else "manual" when a pitch override is given
  // (back-compat — bare `pitch` still forces manual), else the fine lattice.
  const target: PixelTarget = opts.target ?? (hasPitch ? "manual" : "fine");
  if (target === "composite") return recoverComposite(r, opts);

  let grid: Grid;
  if (target === "manual" && hasPitch) {
    const { px, py } = lumaEdgeProfiles(r);
    grid = {
      pitchX: opts.pitch!,
      pitchY: opts.pitch!,
      phaseX: bestPhase(px, opts.pitch!).phase,
      phaseY: bestPhase(py, opts.pitch!).phase,
      confidence: 1,
    };
  } else if (target === "subject") {
    grid = detectSubjectGrid(r);
  } else {
    grid = detectGrid(r);
  }

  const { width: W, height: H, data } = r;
  const bg: RGB = bgMode === "custom" ? parseColor(opts.bgColor ?? "#181818") : globalBg(r);

  const { pitchX: px, pitchY: py } = grid;
  const [ox, oy] = gridOrigin(grid);
  const nCols = Math.max(1, Math.round((W - ox) / px));
  const nRows = Math.max(1, Math.round((H - oy) / py));
  const halfX = Math.max(1, Math.floor((px * 0.9) / 2));
  const halfY = Math.max(1, Math.floor((py * 0.9) / 2));
  const out = new Uint8ClampedArray(nCols * nRows * 4);
  const coverage = new Float32Array(nCols * nRows);
  const litColors: RGB[] = [];

  for (let rr = 0; rr < nRows; rr++) {
    const cy = Math.round(oy + (rr + 0.5) * py);
    const y0 = Math.max(0, cy - halfY);
    const y1 = Math.min(H - 1, cy + halfY);
    for (let cc = 0; cc < nCols; cc++) {
      const cx = Math.round(ox + (cc + 0.5) * px);
      const x0 = Math.max(0, cx - halfX);
      const x1 = Math.min(W - 1, cx + halfX);
      // Sample the WHOLE cell window (a braille dot sits off-centre). The modal
      // non-bg colour anchors the foreground cluster; its kept count is presence.
      const m = windowMode(data, W, x0, x1, y0, y1, (rp, gp, bp) => dist(rp, gp, bp, bg) > tau);
      const cell = rr * nCols + cc;
      let c: RGB;
      if (m.count >= minLit) {
        const area = (x1 - x0 + 1) * (y1 - y0 + 1);
        coverage[cell] = Math.min(1, m.count / area); // dot size as a fraction
        // shade: area-average the whole cell → a solid cell keeps its colour, a
        // small dot averages mostly-bg → the right dim tone, recovering size-
        // modulated halftone gradients. Off: the crisp pure-ink cluster colour.
        c = shade
          ? windowMean(data, W, x0, x1, y0, y1)
          : clusterMean(data, W, x0, x1, y0, y1, bg, m.colour, tau);
        litColors.push(c);
      } else {
        c = bg;
      }
      const oi = cell * 4;
      out[oi] = c[0];
      out[oi + 1] = c[1];
      out[oi + 2] = c[2];
      out[oi + 3] = 255;
    }
  }
  return { width: nCols, height: nRows, data: out, coverage, bg, litColors };
}

// ── composite (multi-grid): braille background + colour sprite ────────────────

/** Fill enclosed holes in a binary mask: flood the NON-mask region inward from the
 *  border; any non-mask pixel the flood can't reach is an interior hole. Mirrors
 *  scipy.ndimage.binary_fill_holes — turns the sprite outline into a solid body so
 *  dark interior pixels (eyes) can be told apart from the outside background. */
function fillHoles(mask: Uint8Array, W: number, H: number): Uint8Array {
  const reach = new Uint8Array(W * H);
  const stack: number[] = [];
  const push = (p: number) => {
    if (!mask[p] && !reach[p]) { reach[p] = 1; stack.push(p); }
  };
  for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
  while (stack.length) {
    const p = stack.pop() as number;
    const x = p % W, y = (p / W) | 0;
    if (x > 0) push(p - 1);
    if (x < W - 1) push(p + 1);
    if (y > 0) push(p - W);
    if (y < H - 1) push(p + W);
  }
  const body = new Uint8Array(W * H);
  for (let p = 0; p < W * H; p++) body[p] = mask[p] || !reach[p] ? 1 : 0;
  return body;
}

/** 8-neighbour binary erosion by `iters` — shrinks a mask inward so the eye
 *  detector stays off the silhouette boundary. */
function erode(mask: Uint8Array, W: number, H: number, iters: number): Uint8Array {
  let cur = mask;
  for (let it = 0; it < iters; it++) {
    const next = new Uint8Array(W * H);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const p = y * W + x;
        if (cur[p] && cur[p - 1] && cur[p + 1] && cur[p - W] && cur[p + W] &&
            cur[p - W - 1] && cur[p - W + 1] && cur[p + W - 1] && cur[p + W + 1]) next[p] = 1;
      }
    }
    cur = next;
  }
  return cur;
}

interface SpriteMasks { body: Uint8Array; eye: Uint8Array; lum: Float32Array; }

/** Per-region masks for the colour sprite: a filled body silhouette and its
 *  near-black interior eye pixels. NB eyes are found by LOW LUMA, not colour —
 *  on near-black pixels saturation is unreliable (tiny channel noise reads as
 *  fully saturated), which would wrongly classify an eye as coloured sprite. */
function spriteMasks(r: Raster, satThr: number, eyeLuma: number): SpriteMasks {
  const { width: W, height: H, data } = r;
  const lum = new Float32Array(W * H);
  const fg = new Uint8Array(W * H);
  for (let p = 0, i = 0; p < W * H; p++, i += 4) {
    const R = data[i], G = data[i + 1], B = data[i + 2];
    const mx = Math.max(R, G, B);
    lum[p] = 0.299 * R + 0.587 * G + 0.114 * B;
    // RELATIVE saturation (max-min)/max, not absolute: shadowed sprite stays
    // foreground so the silhouette fully encloses the eyes (absolute saturation
    // leaves gaps in dark sprite, letting the hole-fill leak into the eyes).
    if (mx > 0 && (mx - Math.min(R, G, B)) / mx > satThr) fg[p] = 1;
  }
  const body = fillHoles(fg, W, H);
  const inner = erode(body, W, H, 2);
  const eye = new Uint8Array(W * H);
  for (let p = 0; p < W * H; p++) if (inner[p] && lum[p] < eyeLuma) eye[p] = 1;
  return { body, eye, lum };
}

interface SpriteArt { cols: number; rows: number; ox: number; oy: number; px: number; py: number; cells: (RGB | null)[]; }

/** Snap the sprite to its OWN (subject) grid: one clean colour per native pixel,
 *  black where an eye sits, null outside the body. Body colour is the mean of the
 *  non-dark body pixels in the cell (excludes filled eye-holes). */
function regridSprite(r: Raster, grid: Grid, m: SpriteMasks, bodyThr: number, eyeThr: number, eyeLuma: number): SpriteArt {
  const { width: W, height: H, data } = r;
  const { pitchX: px, pitchY: py } = grid;
  const [ox, oy] = gridOrigin(grid);
  const cols = Math.max(1, Math.round((W - ox) / px));
  const rows = Math.max(1, Math.round((H - oy) / py));
  const cells: (RGB | null)[] = new Array(cols * rows).fill(null);
  for (let j = 0; j < rows; j++) {
    const y0 = Math.max(0, Math.round(oy + j * py));
    const y1 = Math.min(H - 1, Math.round(oy + (j + 1) * py) - 1);
    for (let i = 0; i < cols; i++) {
      const x0 = Math.max(0, Math.round(ox + i * px));
      const x1 = Math.min(W - 1, Math.round(ox + (i + 1) * px) - 1);
      let bodyN = 0, eyeN = 0, area = 0, sr = 0, sg = 0, sb = 0, n = 0;
      for (let y = y0; y <= y1; y++) {
        const row = y * W;
        for (let x = x0; x <= x1; x++) {
          const p = row + x; area++;
          if (m.body[p]) bodyN++;
          if (m.eye[p]) eyeN++;
          if (m.body[p] && m.lum[p] >= eyeLuma) { const i4 = p * 4; sr += data[i4]; sg += data[i4 + 1]; sb += data[i4 + 2]; n++; }
        }
      }
      if (area === 0 || bodyN / area < bodyThr) continue;                 // outside body
      if (eyeN / area > eyeThr) { cells[j * cols + i] = [0, 0, 0]; continue; } // eye → black
      cells[j * cols + i] = n ? [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] : null;
    }
  }
  return { cols, rows, ox, oy, px, py, cells };
}

/** Bright 3-tone grey ramp for the braille background (mirrors the prototype's
 *  SHADE3). Lit cells are remapped onto it by their cluster luma so the dither
 *  field reads as crisp poster shapes instead of dim, varied original colours.
 *  Cuts are on the 0..255 luma scale: <105 → dim, <175 → mid, else bright. */
const BG_SHADES: RGB[] = [
  [112, 112, 112],
  [180, 180, 180],
  [245, 245, 245],
];

/** Grey from the 3-tone ramp for a recovered cell luma (cuts at 105 / 175). */
const bgShade = (luma: number): RGB => BG_SHADES[luma >= 175 ? 2 : luma >= 105 ? 1 : 0];

/** Composite recovery — rendered at SOURCE resolution to reproduce the braille-art
 *  look of the standalone prototype. The background braille field is drawn as
 *  individual DOTS (a gapped square per lit lattice cell, tinted onto the bright
 *  3-tone ramp); the colour sprite is snapped to its own grid and drawn as solid
 *  pixels on top. NB the prior one-output-pixel-per-cell version collapsed each dot
 *  into a solid block (blocky mush at any size) — drawing real dots with gaps (≈ the
 *  prototype's dot=6 on an 8px pitch) is what actually reads as braille. Because the
 *  gaps are intrinsic to the bitmap, this output must NOT be nearest-neighbour
 *  downscaled (studio skips conform for composite); the browser/export keep it crisp. */
function recoverComposite(r: Raster, opts: RepixelOptions): Recovery {
  const satThr = opts.spriteSat ?? 0.3;
  const eyeLuma = opts.eyeLuma ?? 45;
  const { width: W, height: H } = r;
  const bg: RGB =
    (opts.bgMode ?? "auto") === "custom" ? parseColor(opts.bgColor ?? "#181818") : globalBg(r);

  // 1) Fine braille lattice (two-tone) → which cells are lit, and their tone.
  const fine = recover(r, { ...opts, target: "fine", shade: false });
  const fineGrid = detectGrid(r);
  const [ox, oy] = gridOrigin(fineGrid);
  const { pitchX: px, pitchY: py } = fineGrid;

  // 2) Colour sprite snapped to its own (subject) grid → one clean colour per pixel.
  const m = spriteMasks(r, satThr, eyeLuma);
  const art = regridSprite(r, detectSubjectGrid(r), m, 0.45, 0.25, eyeLuma);

  // 3) Paint a source-resolution canvas: background colour, then gapped braille
  //    dots, then solid sprite pixels on top (compositing order).
  const data = new Uint8ClampedArray(W * H * 4);
  for (let o = 0; o < data.length; o += 4) {
    data[o] = bg[0]; data[o + 1] = bg[1]; data[o + 2] = bg[2]; data[o + 3] = 255;
  }
  const litColors: RGB[] = [];
  const fill = (x0: number, y0: number, w: number, h: number, c: RGB): void => {
    const xa = Math.max(0, x0), xb = Math.min(W, x0 + w);
    const ya = Math.max(0, y0), yb = Math.min(H, y0 + h);
    for (let y = ya; y < yb; y++) {
      let o = (y * W + xa) * 4;
      for (let x = xa; x < xb; x++, o += 4) {
        data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 255;
      }
    }
  };

  // Background dots: each fills ~72% of the lattice pitch, leaving the inter-dot gap.
  const dotW = Math.max(1, Math.round(px * 0.72));
  const dotH = Math.max(1, Math.round(py * 0.72));
  for (let rr = 0; rr < fine.height; rr++) {
    for (let cc = 0; cc < fine.width; cc++) {
      const cell = rr * fine.width + cc;
      if (fine.coverage[cell] <= 0) continue; // unlit → background
      const fo = cell * 4;
      const lum = 0.299 * fine.data[fo] + 0.587 * fine.data[fo + 1] + 0.114 * fine.data[fo + 2];
      const g = bgShade(lum);
      const cx = Math.round(ox + (cc + 0.5) * px);
      const cy = Math.round(oy + (rr + 0.5) * py);
      fill(cx - (dotW >> 1), cy - (dotH >> 1), dotW, dotH, g);
      litColors.push(g);
    }
  }

  // Colour sprite: solid pixels on top (one block per snapped native pixel).
  const blockW = Math.round(art.px) + 1;
  const blockH = Math.round(art.py) + 1;
  for (let j = 0; j < art.rows; j++) {
    for (let i = 0; i < art.cols; i++) {
      const col = art.cells[j * art.cols + i];
      if (!col) continue;
      fill(Math.round(art.ox + i * art.px), Math.round(art.oy + j * art.py), blockW, blockH, col);
      litColors.push(col);
    }
  }

  return { width: W, height: H, data, coverage: new Float32Array(0), bg, litColors };
}

/** Recover the logical bitmap: one output pixel per detected sub-cell. */
export function renderRepixel(r: Raster, opts: RepixelOptions = {}): RenderResult {
  const palMode = opts.palette ?? "original";
  const rec = recover(r, opts);
  const outRaster: Raster = { width: rec.width, height: rec.height, data: rec.data };

  if (palMode === "original") return rasterToIndexed(outRaster);

  // Palette modes: build the palette, remap the bitmap to it, then dedupe via
  // rasterToIndexed. Adaptive clusters the FOREGROUND only (the bitmap is mostly
  // background); grayscale/duotone keep their fixed tonal ramps.
  const palette =
    palMode === "adaptive"
      ? foregroundPalette(rec.litColors, rec.bg, opts.colors ?? 4, opts.pins)
      : buildTonePalette(outRaster, opts.colors ?? 4, palMode, opts.inkColor, opts.paperColor, 42, opts.pins);
  const idx = quantizeToPalette(outRaster, palette);
  const remapped = new Uint8ClampedArray(rec.data.length);
  for (let p = 0; p < idx.length; p++) {
    const c = palette[idx[p]];
    const o = p * 4;
    remapped[o] = c[0];
    remapped[o + 1] = c[1];
    remapped[o + 2] = c[2];
    remapped[o + 3] = 255;
  }
  return rasterToIndexed({ width: rec.width, height: rec.height, data: remapped });
}

/** Recover just the per-cell coverage map (dot size, 0..1) — feeds the dithered
 *  glyph-text export so it can reproduce halftone tone as braille dot density. */
export function recoverCoverage(r: Raster, opts: RepixelOptions = {}): {
  coverage: Float32Array;
  width: number;
  height: number;
} {
  const rec = recover(r, opts);
  return { coverage: rec.coverage, width: rec.width, height: rec.height };
}

// ── glyph-text export ─────────────────────────────────────────────────────────

/** Unicode braille dot bit for a sub-cell at (col 0..1, row 0..3). */
const brailleBit = (cx: number, ry: number): number => (ry < 3 ? cx * 3 + ry : 6 + cx);

/** Re-encode a recovered bitmap back into braille + block glyph text — the inverse
 *  of the terminal art it came from. Each char packs one 2x4 block of cells, so
 *  grouping the fine cells by 2x4 reconstructs ~the source's character grid: a
 *  fully-lit block becomes "█", an empty block a space, anything in between the
 *  matching braille pattern (U+2800 + dot mask). Background = the modal palette
 *  index (the dominant ~background of the recovery). Run this on the NATIVE render
 *  (1px per cell), before any output-size upscaling. */
export function toGlyphText(res: RenderResult): string {
  const { width: w, height: h, indices } = res;
  if (w === 0 || h === 0) return "";
  const counts = new Map<number, number>();
  for (const i of indices) counts.set(i, (counts.get(i) ?? 0) + 1);
  let bg = 0;
  let best = -1;
  for (const [k, v] of counts) if (v > best) { best = v; bg = k; }
  const lit = (x: number, y: number): boolean => x < w && y < h && indices[y * w + x] !== bg;
  const cols = Math.ceil(w / 2);
  const rows = Math.ceil(h / 4);
  let outText = "";
  for (let by = 0; by < rows; by++) {
    let row = "";
    for (let bx = 0; bx < cols; bx++) {
      let mask = 0;
      for (let cx = 0; cx < 2; cx++) {
        for (let ry = 0; ry < 4; ry++) {
          if (lit(2 * bx + cx, 4 * by + ry)) mask |= 1 << brailleBit(cx, ry);
        }
      }
      row += mask === 0xff ? "█" : mask === 0 ? " " : String.fromCharCode(0x2800 + mask);
    }
    outText += row.replace(/\s+$/, "") + "\n";
  }
  return outText;
}
