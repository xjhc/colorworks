# Glyph-fit — ABANDONED (postmortem)

**Status: removed from the Colorworks Pages SPA.** The older "Glyph art" (repixel)
mode stays. This file is kept only as the record of *why* glyph-fit didn't work;
everything below "Historical design" is the design of the renderer that was built
and then removed.

## Why it didn't work

Goal: recover the source's *glyph structure* — braille dot-fields (planet, stars,
comet) and a block sprite (frog) — as a faithful glyph document. On the hero image
(`Screenshot_20260604_144633.png`), everything but the frog is braille; glyph-fit
never recovers that. Three structural reasons:

1. **It never classifies regions.** glyph-fit applies ONE strategy everywhere —
   per-cell 2-means -> binary 2x4 mask -> nearest glyph. It has no notion of
   "braille dither field" vs "block sprite" vs "text", so it can't honour that the
   planet should be braille and the frog blocks. It just quantises every cell to
   one fg/bg + a thresholded mask and emits whatever glyph that mask matches.

2. **One global grid can't fit the image.** The braille background sits on a 16x32
   terminal cell (8px dots); the frog sprite is an incommensurate ~12.4px grid
   (measured). A single grid is right for one and wrong for the other.

3. **The fit can't represent braille.** Real braille is a sub-cell dot pattern,
   anti-aliased. Collapsing a 16x32 cell to one fg/bg + an 8-bit on/off mask
   discards the dot pattern and tone, so braille fields come back as coarse blobs,
   not braille.

Net: it conflates "detect the glyph structure" with "quantise each cell to two
colours", and does neither — detecting neither braille-vs-block regions nor the
true braille dot patterns. The plausible-looking reconstruction hid this: it is a
2-colour-per-cell downsample relabelled as glyphs, not a recovery of the source
glyphs. A passable raster is not a faithful glyph document, which was the point.

## What it would have taken

Per-region **segmentation** first (sprite vs dither-field vs text), then a separate
grid + strategy per region: block/fg-bg for the sprite at its own pitch; **true
braille-pattern matching at the dot lattice** for the dither field (read the 2x4
dots, don't 2-means the whole cell); leave text alone. That is the composite /
multi-grid problem — substantially harder than a single-grid fitter, and out of
scope for the SPA.

---

# Historical design (the removed renderer)

## 0. Goal & success criteria (the north star)

**Goal:** represent the image as the *best possible* block/braille glyph
document, then **prove that document renders back close to the input.**

```
input image
  -> detected/selected character grid (cellW/cellH + phaseX/phaseY)
  -> one ALLOWED glyph per character cell
  -> colour data attached to that glyph
  -> rendered-back image   ≈   input
```

North star: **find the lowest-error, glyph-constrained explanation of the image,
and make the remaining error visible.** Not "make pretty ASCII" — an *explanation*
with a residual we report, not hide.

Treating residual as a first-class output forces two rules the rest of this doc
must honour:

- **Fit objective ≠ reported metric.** The per-cell glyph is *chosen* by
  coverage-based shape loss + lit-area penalty (§4 — avoids the colour-copy
  degeneracy). The number we *report, rank grids by, and gate success on* is the
  actual **render-back reconstruction error** (RGB MSE of the expanded glyph
  render vs the source). Store both: per-cell `score` (fit loss) and per-cell
  `reconError` + document `meanError` (the honest metric).
- **A bad fit must be loud, not hidden** (criterion 6). If an image isn't well
  explained by block/braille, `meanError` stays high and the UI says so. "This
  isn't really glyph art" is a *successful*, detectable outcome.

### Success criteria

1. **Synthetic exactness** — on fixtures where we know the answer, recover the
   correct `cellW/cellH`, `phaseX/phaseY`, glyph chars, masks, and colours:
   near-zero-error round trips.
2. **Font/cell-size correctness** — a live readout
   (`grid: 16×32 px · offset 2,5 · cells 106×29 · mean err 42`); changing cell
   size/offset visibly snaps in/out of alignment (wrong → smeared/chopped glyphs;
   right → cell boundaries line up with the source).
3. **Constrained glyph truth** — every cell is one allowed glyph (`block2x2`,
   `braille2x4`, later `sextant2x3`). No arbitrary bitmap pixels except as the
   colour expansion of a chosen mask.
4. **Render-back fidelity** — expanded `GlyphDocument` visually matches the
   source *after normalization*; reconstruction error low; *better grid/glyph
   choices rank lower-error* (the app must at least order candidates correctly).
   **Normalize before comparing**: cell-aware downsample (or slight blur) +
   alpha-insensitive thresholding, so we don't secretly re-optimize toward the
   anti-aliased screenshot pixels. **Raw screenshot MSE is diagnostic only, not
   the optimization target.** RGB MSE first, SSIM/perceptual later.
5. **Honest exports** — PNG faithful; JSON faithful (shape + colour) and
   round-trips to the *same* PNG render; text shape-only and clearly labelled
   lossy (never pretends to carry per-subpixel colour).
6. **Useful failure signal** — poorly-explained images surface as high residual,
   not silently-wrong output.

### Definition of done

- **v1:** manual cell size + offset works; synthetic tests pass exactly; a real
  screenshot yields a plausible glyph document; PNG render-back is close; JSON
  round-trips; text export labelled lossy.
- **v2:** auto mode finds the same cell size/offset that manual tuning finds.

### Hero acceptance fixture — the space-frog screenshot

`/home/xjhc/Screenshots/Screenshot_20260604_144633.png` (1691×935). The canonical
test image. **Every *artwork* region appears intended to be block/braille; the
top-right "now playing" text widget is explicitly out of scope unless we extend
the alphabet to a real font.**

Measured grids (luma + fg-mask autocorrelation, `/tmp/measure_grid.py`):

| Region | Lattice | Implied char cell |
| --- | --- | --- |
| Planet / stars / comet (braille) | 8 px dot pitch, both axes (autocorr 0.97) | **16 × 32 px** (2×4 dots @ 8px), phase ≈ (3,3) mod 8 |
| Frog sprite (blocks) | ~12.4 px pixel (runs ≈ 24/37/49) | does **not** divide 8 or 16 |
| Text widget | ~7–8 px (likely a separate UI font) | out of scope |

> **Key finding — this image is NOT one grid.** The braille background sits cleanly
> on a 16×32 terminal cell, but the frog is ~12.4 px and **incommensurate** with it
> (the same two-scale problem `repixel` documented — see `project_repixel`). A
> *single* `GlyphGrid` cannot recover both, so **"reproduce 100% with one glyph
> grid" is not achievable here.** Full recovery needs a composite document
> (background grid + the frog's own ~12.4 px grid; text via an extended alphabet) —
> the deferred multi-grid work.

This makes the fixture a precise **failure-signal** test (criterion 6). Expected
residual under the **single background grid (16×32)**:

```
low residual:   planet, ring, stars, comet, braille field   (genuinely this grid)
edge halo:      frog OUTLINE / legs / eyes — 12.4 px edges don't sit on the 8 px
                sub-cell → quantization residual along edges. Frog INTERIOR is flat
                salmon → still low. This halo IS the "wrong grid" signal.
high residual:  text widget — alphabet can't represent letters.
```

(An earlier sketch said "low on the frog, high only on text." The measurement
corrects it: under one grid the frog shows an edge halo because it lives on a
different lattice — honest, not a bug.)

**v1 success on this fixture** = background recovered crisp on the 16×32 grid; the
frog-edge halo and the text widget both light up in the residual map (two distinct,
explainable failures). **Full recovery** = add the frog's ~12.4 px grid (a
subject / second-grid pass, cf. repixel's `target:"subject"`) so only the text
remains hot.

**Prototype result** (`glyphfit_proto.py` — single 16×32 grid, `fg_bg`,
blocks+braille). Mean cell-MSE **171**; glyph document = **2806 space · 58 block ·
181 braille** over 105×29 cells. The residual map matches the prediction *exactly*:
frog OUTLINE + text widget hot, planet body / stars / empty sky low (empty-sky
cell-MSE 31 vs frog 871 vs text 720). Frog, planet, ring, comet and stars all
render back recognizably (`glyphfit_proto_recon.png`). Confirmed limitations:
(a) frog edges chunky — 8 px subcell vs 12.4 px sprite pixel → needs the frog's own
grid; (b) planet reads dimmer — size-modulated halftone collapses to binary on/off
(braille can't carry partial dot coverage; cf. repixel's `shade`); (c) text fuzzy
(out of alphabet). **Algorithm validated → port to TS per §8.**

---

## 1. Why this is different from `repixel`

Current `repixel` pipeline (see `colorworks/pages/src/repixel.ts`):

```
image
  -> detectGrid()        (luminance edge comb → fine lattice pitch)   repixel.ts:162
  -> recover()           (one output pixel per lattice cell)          repixel.ts:279
  -> renderRepixel()     (RenderResult: indexed bitmap)               repixel.ts:355
  -> toGlyphText()       (post-hoc: group bitmap 2x4 → braille/block) repixel.ts:405
```

`toGlyphText()` is **post-hoc packing**: it thresholds an already-recovered
bitmap into braille chunks. It is not "represent the source as constrained
glyphs." The glyph alphabet never participates in the fit.

Target `glyphfit` pipeline (the v1 shape, revised):

```
detect/choose GlyphGrid { cellW, cellH, phaseX, phaseY }
  -> for each char cell: sample into CANONICAL 2x4 subcells (+ per-subcell coverage)
  -> fit allowed glyph masks against COVERAGE (shape), with a lit-area penalty
  -> attach colours to the chosen mask (fg_bg for recovery, or per_subpixel)
  -> GlyphDocument (glyph char + colour(s) per cell, one canonical grid)
  -> render-back to bitmap / glyph text / JSON
```

The output is **genuinely constrained** to the block + braille alphabet, and the
*shape* is driven by reconstruction error against per-subcell coverage — which
also answers "what's the pixel/font size?" honestly (the grid that best explains
the image). Keep `repixel.ts` as-is. `glyphfit` is a **separate renderer**.

---

## 2. Correctness notes (carry these into code)

- **Braille is 2×4, not 2×3.** Unicode braille `U+2800..U+28FF`, 8 dots:

  ```
  dot index layout      bit positions (U+2800 + mask)
    c0 c1                 0  3
    c2 c3                 1  4
    c4 c5                 2  5
    c6 c7                 6  7
  ```

  Reuse the existing helper (currently **private** — must be `export`ed):

  ```ts
  // repixel.ts:396  →  export it
  const brailleBit = (cx, ry) => (ry < 3 ? cx * 3 + ry : 6 + cx);
  ```

  A 2×4 boolean mask in canonical row-major order maps to a codepoint via
  `sum(1 << brailleBit(c, r))` over lit cells. **Do not** assume row-major == bit
  order.

- **2×3 is sextant/block-symbol territory** (`U+1FB00..`), *not* braille. Start
  with block + braille only; `sextant2x3` is a later, separately-named option.

- **Per-subpixel colour cannot round-trip to plain text.** Plain terminal text
  carries glyph *shape* + at most one fg/bg pair per cell. The faithful artifact
  is the `GlyphDocument` (JSON) or a rendered PNG. Glyph-text export is a
  deliberately lossy/stylized view. The UI must make that explicit (separate
  buttons); don't pretend text is faithful.

---

## 3. Core data model — one canonical subgrid

> **Review fix (High #1):** if each cell rendered to *its own* `subW×subH`, a row
> mixing block `2×2` and braille `2×4` cells would expand ragged (different
> heights) — impossible to lay out. So the **document** owns a single canonical
> subgrid (`cellSubW=2, cellSubH=4`), and block masks are **promoted** into it by
> vertical expansion (each block row → two canonical rows). Every cell renders to
> 2×4 logical pixels regardless of glyph family, so `blocks_braille` is legal.

New file `colorworks/pages/src/glyphfit.ts`:

```ts
import type { RGB } from "./colorworks";

export type GlyphKind = "block2x2" | "braille2x4"; // sextant2x3 later

/** Where the character cells sit on the source — cell size AND phase/origin. */
export interface GlyphGrid {
  cellW: number;        // source px per char cell, x
  cellH: number;        // source px per char cell, y
  phaseX: number;       // x offset of the first cell boundary (origin)
  phaseY: number;       // y offset of the first cell boundary (origin)
}

export interface GlyphCell {
  glyph: string;        // the chosen character
  kind: GlyphKind;
  mask: boolean[];      // length cellSubW*cellSubH (= 8), CANONICAL grid, row-major
  bg: RGB;              // background colour for this cell
  colors: RGB[];        // length 8 — colour per canonical subpixel (lit cells meaningful)
  fg?: RGB;             // optional single fg (fg_bg colour model)
  coverage: number[];   // length 8 — measured foreground coverage 0..1 per subcell (for scoring/debug)
  score: number;        // shape-fit loss that CHOSE this glyph (coverage-based; lower = better)
  reconError: number;   // render-back RGB MSE of this cell vs source — the HONEST per-cell metric
}

export interface GlyphDocument {
  sourceWidth: number;
  sourceHeight: number;
  grid: GlyphGrid;
  cellSubW: 2;          // canonical subgrid — fixed
  cellSubH: 4;          // canonical subgrid — fixed
  cols: number;
  rows: number;
  cells: GlyphCell[];   // length cols*rows, row-major
  meanError: number;    // mean reconError over all cells — drives the readout + failure signal (§0)
}
```

Two colour models (colour is attached **after** the mask is chosen — see §4):

- **`fg_bg`** (default for **terminal recovery**): one foreground + one background
  colour per cell. This *is* the model that produced terminal art — a braille cell
  shares one fg across all 8 dots; a half-block is fg-over-bg — so it recovers the
  original buffer and **denoises** anti-aliasing rather than fitting it.
  Round-trips to ANSI/HTML later.
- **`per_subpixel`** (opt-in, **expressive approximation**): each lit subpixel keeps
  its own sampled colour; `bg` fills the rest. More expressive than any real
  terminal cell — right for explaining an *arbitrary* image (e.g. a photo) as
  glyphs, **wrong** for recovering an original terminal buffer (it spends 8 colours
  where the terminal had 2). Not representable as plain text.

> **`per_subpixel` is not "wrong" in general — only for recovery.** And note the
> **metric/goal divergence**: against the raw anti-aliased screenshot,
> `per_subpixel` scores *lower* pixel-MSE than `fg_bg` (it fits the AA fringe), yet
> `fg_bg` is the truthful recovery. This is exactly why criterion 4 compares against
> a **normalized** target, not raw pixels (§0).

---

## 4. New files & the fit algorithm

```
colorworks/pages/src/glyph_alphabet.ts   # allowed masks (canonical 2x4) + char mapping
colorworks/pages/src/glyphfit.ts         # grid, fitter, render-back, exports
colorworks/pages/src/glyphfit.test.ts    # vitest, synthetic cells first
```

### `glyph_alphabet.ts`

```ts
export interface GlyphMask {
  id: string;
  char: string;
  kind: GlyphKind;
  mask: boolean[];   // CANONICAL 2x4, length 8, row-major (block masks pre-promoted)
}

export function brailleAlphabet(): GlyphMask[];   // 256, U+2800..U+28FF, mask via brailleBit
export function blockAlphabet(): GlyphMask[];      // 16 quadrant combos, each PROMOTED to 2x4
export function alphabetFor(kind: "blocks" | "braille" | "blocks_braille"): GlyphMask[];
```

- **braille**: all 256 from the bit mapping (mask → codepoint via `brailleBit`).
- **block 2×2 → promoted to 2×4**: the 16 quadrant combos cover every 2×2 mask;
  map each to its char (`space`,`█`,`▀`,`▄`,`▌`,`▐`,`▖▗▘▝`,`▙▟▛▜`,`▚▞`), then
  **vertically expand**: 2×2 `[a b / c d]` → 2×4 `[a b / a b / c d / c d]`. Block
  and braille masks now live on the same canonical grid.
- Reuse `brailleBit` from `repixel.ts` (export it).

### `glyphfit.ts` — grid

```ts
export function detectGlyphGrid(raster: Raster, opts): GlyphGrid   // milestone B
export function manualGlyphGrid(cellW, cellH, phaseX, phaseY): GlyphGrid  // milestone A
```

> **Review fix (High #3):** correct font size in a screenshot is
> `cellW + cellH + phaseX + phaseY`, not just W/H — `repixel` itself needs
> `gridOrigin` before sampling (`repixel.ts:308`). Manual mode exposes X/Y offset
> controls; auto mode finds phase too (reuse `bestPhase`/`gridOrigin` from
> `depixelate`, already used by repixel).

### `glyphfit.ts` — fit (shape first, colour after)

> **Review fix (High #2):** if lit subpixels just copy their source sample and the
> score compares reconstruction to those same samples, an all-lit glyph copies
> everything and scores ~0 → `█`/full braille always wins. **Separate the two
> stages**: score masks against per-subcell *coverage* (a shape signal), with a
> lit-area penalty so "full" doesn't trivially win; attach colours only after.

```ts
function fitGlyphCell(raster, grid, col, row, masks, colorModel, bg): GlyphCell {
  // 1. SAMPLE: split the cell window into canonical 2x4 subcells.
  //    For each subcell compute mean RGB (for colour) AND coverage 0..1
  //    (fraction of pixels with dist(px, bg) > tau — the "is this lit" signal).
  const { meanRGB, coverage } = sampleSubcells(raster, grid, col, row); // arrays len 8

  // 2. FIT SHAPE: pick the mask whose boolean pattern best matches coverage.
  //    shapeLoss = Σ (mask[i] ? (1-coverage[i]) : coverage[i])^2  + λ * litCount(mask)
  //    The λ·litCount penalty stops a near-empty cell choosing █ to "cover" noise,
  //    and stops a near-full cell over-paying for a missing-corner braille glyph.
  let best; for (const m of masks) { const s = shapeLoss(m, coverage); if (!best || s < best.s) best = {m, s}; }

  // 3. ATTACH COLOUR to the chosen mask only:
  //    per_subpixel: colors[i] = mask[i] ? meanRGB[i] : bg
  //    fg_bg:        fg = mean(meanRGB over lit subcells); bg as given; colors = mask?fg:bg
  return buildCell(best.m, meanRGB, coverage, colorModel, bg);
}
```

- `sampleSubcells`: mirror the windowed sampling in `recover()` / `windowMean()`
  (`repixel.ts:232`). Coverage uses the same bg-distance test (`dist`, `tau`).
- `bg`: reuse repixel's `globalBg()` (`repixel.ts:186`) **or** a custom colour
  (same `bgMode` knob). `globalBg` is **private** — export it, move it to a shared
  sampling util, or keep a local copy in glyphfit (review medium #6).
- `fitGlyphDocument(raster, opts) -> GlyphDocument`: build grid, loop cells.

### `glyphfit.ts` — render-back (uniform 2×4)

```ts
function renderGlyphDocument(doc: GlyphDocument): RenderResult
```

Every cell expands to the canonical **2×4** logical pixels (no per-cell ragged
size), painted from `colors[]`. Native output = `cols*2 × rows*4`. Build an
indexed `RenderResult` (dedupe palette via `rasterToIndexed` from `depixelate`,
like `renderRepixel` at `repixel.ts:360`). The studio's existing `conformIndexed`
pass scales it to the output-size control — **no renderer-local scale** (see §6).

### Exports

```ts
function glyphDocumentToText(doc): string   // shape only (chars), lossy re colour
function glyphDocumentToJSON(doc): string   // faithful: full GlyphDocument
// PNG falls out of the canvas → existing exportPng path
```

---

## 5. Grid detection — two milestones

**Milestone A (build first): manual grid.** Knobs: cell W, cell H, **offset X,
offset Y** (phase). Fit → render-back → measure error. Prove the fitter is
correct before any auto-detection. De-risks everything.

**Milestone B: auto grid search.** Search cell size **and phase**, pick the grid
minimising total shape loss (+ a small complexity penalty so it doesn't prefer
tiny cells):

```
seed from detectCandidates() (repixel.ts:176) for cell size, bestPhase() for phase
refine: argmin over (cellW, cellH, phaseX, phaseY) of meanError (render-back recon)
        + λ*(cols*rows)   // rank candidates by the HONEST recon error, not shape loss
```

Ranking by render-back `meanError` is what makes criterion 4 true (better grids
rank lower) and reuses the same number the readout shows. Better than
edge-periodicity alone (`fundamental`/`bestPhase`) because the goal
isn't "find edges" — it's "best explain the image with this alphabet" — but those
detectors are the right *seed*.

---

## 6. SPA integration

### `schema.ts`

- Add renderer id (`schema.ts:419`):
  `export type RendererId = "tone_dither" | "depixelate" | "repixel" | "glyphfit";`
- Add `GLYPHFIT_PARAMS: ParamDef[]` (mirror `REPIXEL_PARAMS`, `schema.ts:295`):
  - `cell_mode`: `"auto" | "manual"`
  - `cell_w`, `cell_h`, **`offset_x`, `offset_y`** (int, `visibleWhen` cell_mode == manual)
  - `alphabet`: `"blocks" | "braille" | "blocks_braille"`
  - `color_model`: `"per_subpixel" | "fg_bg"`
  - `tau` (foreground/coverage threshold), `bg_mode` / `bg_color` (reuse repixel pattern)
  - `max_colors` (optional global palette cap)
  - **No `scale` knob** — output-size owns scaling (review medium #4: grid
    renderers already pass through `conformIndexed` at `studio.ts:449` →
    `output_size.ts:64`).
- Add a `StyleDef` to `STYLES` (`schema.ts:434`) with `renderer:"glyphfit"`,
  `params: GLYPHFIT_PARAMS`.
- Add `GLYPHFIT_PARAMS` to the `PARAM_BY_KEY` spread (`schema.ts:467`).

### `studio.ts` (review medium #5 — be explicit)

- Add state field next to `glyphText` (`studio.ts:70`):
  `glyphDoc: null as GlyphDocument | null,`
- In `renderFocus`, **reset** both each render (near `studio.ts:402`):
  `state.glyphText = ""; state.glyphDoc = null;`
- New dispatch branch alongside repixel (`studio.ts:420`):
  ```ts
  } else if (style.renderer === "glyphfit") {
    const doc = fitGlyphDocument(raster, glyphOpts(vals));
    res = renderGlyphDocument(doc);              // native cols*2 × rows*4
    state.glyphDoc = doc;
    state.glyphText = glyphDocumentToText(doc);  // reuse existing copy path
  }
  ```
- Include `glyphfit` in the `gridRenderer` test (`studio.ts:405`) so it sees the
  native-size source and gets `conformIndexed` output-size handling.
- **Error readout in the caption** (criteria 2 + 6): model on repixel's live
  `repixelInfo` string (`studio.ts:440`). Show
  `grid W×H px · offset x,y · cells C×R · mean err N`, driven from `doc.grid` and
  `doc.meanError`. This is how cell-size/offset alignment becomes *visible*.
- **Failure signal**: when `doc.meanError` exceeds a threshold, surface a
  "low glyph-art confidence" note rather than hiding a bad fit. Calibrate the
  threshold between the synthetic-exact baseline (≈0) and a real-photo baseline.
- The copy button is hidden unless renderer is exactly `repixel`
  (`studio.ts:468`) — widen to `repixel || glyphfit`.
- Add a **Download glyph JSON** handler `downloadGlyphJson()` (model on
  `copyGlyphText`, `studio.ts:697`) that blob-downloads
  `glyphDocumentToJSON(state.glyphDoc)`; wire its click in `init` (`studio.ts:731`)
  and show/hide it for the glyphfit renderer.

### `index.html`

Export row at `index.html:129` (`exportPng` :136, `copyGlyphs` :137). Add honest,
separate exports:
- **Export PNG** (faithful render) — keep.
- **Copy glyph text** (existing — clarify it's shape-only / lossy).
- **Download glyph JSON** (new button — the faithful `GlyphDocument`).

---

## 7. Tests (`glyphfit.test.ts`)

Follow `repixel.test.ts` style (synthetic rasters via `makeRaster`). Cover:

1. **Alphabet**: 256 braille masks; mask↔codepoint round-trips via `brailleBit`;
   16 block-quadrant masks map to the right chars; block masks are promoted to
   2×4 (each row duplicated); full → `█`, empty → space.
2. **Shape fit is non-degenerate** *(the key review test)*: a half-lit cell
   (top half foreground, bottom bg) must choose `▀` / the matching braille, **not**
   `█`; a near-empty cell chooses space, not `█`. Proves coverage+penalty scoring,
   not "full always wins".
3. **Per-cell fit**: a synthetic cell that *is* a known glyph recovers that exact
   glyph + correct per-subpixel colours; shape loss ≈ 0.
4. **Mixed alphabet on one grid**: a row with one block-shaped and one
   braille-shaped cell renders to a clean `cols*2 × rows*4` bitmap (no ragged
   heights) — guards the canonical-subgrid fix.
5. **Grid phase**: a glyph image drawn at a non-zero pixel offset is fit
   correctly only when `phaseX/phaseY` are found/supplied — guards High #3.
6. **Colour models**: `per_subpixel` keeps distinct subpixel colours; `fg_bg`
   collapses to two.
7. **Render-back fidelity**: `renderGlyphDocument(fitGlyphDocument(img))`
   reconstructs a synthetic glyph image within tolerance.
8. **Exports**: `glyphDocumentToText` is shape-only; `glyphDocumentToJSON`
   round-trips to an equal `GlyphDocument`.
9. **Failure signal** (criterion 6): a non-glyph image (photo / random noise)
   yields `meanError` well above a clean synthetic fixture — proves a bad fit is
   detectable, not hidden.
10. (Milestone B) **auto grid** picks the planted cell size *and* phase, and a
    better grid scores lower `meanError` than a misaligned one.

Run: `cd colorworks/pages && npm test` (vitest).

---

## 8. Build order (checklist)

**STATUS: built & shipped** (incl. a design-review round 2). 14 glyphfit tests + 55
total green; `npm run build` clean; hero reproduced end-to-end through the TS path
(auto finds 16×32 itself; `gf_recon.png`). Error is now **normalised** (cell-mean,
not raw AA pixels — §0 criterion 4) so meanError reads ~10, and a **residual heatmap
view** (`renderGlyphResidual` + a "Show residual" toggle) makes the failure visible:
on the hero, frog OUTLINE + text widget glow red, everything else dark (`gf_resid.png`).

- [x] **A0** Exported `brailleBit` and `globalBg` from `repixel.ts`. Repixel tests pass.
- [x] **A1** `glyph_alphabet.ts`: braille (256) + block (16, promoted to 2×4) +
      `chooseGlyph`/`isBlockShaped`/`snapToBlock`. Alphabet tests.
- [x] **A2** `glyphfit.ts` types: `GlyphGrid`, `GlyphCell`, `GlyphDocument`
      (canonical 2×4). `resolveGrid` (manual + auto).
- [x] **A3** `cellStats` (subcell means), `twoMeans` (fg_bg), `fitGlyphCell`
      (shape→colour), `fitGlyphDocument`.
      NOTE: shipped fit differs from the §4 sketch — for the `fg_bg` recovery
      default the fit is **2-means → binary subcell mask**, then a **block-vs-braille
      competition** picks the lowest-`normErr` glyph (a clean block char gets a small
      bias so AA doesn't downgrade a real block to braille). braille encodes any
      pattern exactly, so no mask search is needed. `score` = cluster split cost;
      `reconError` = **normalised** render-back error (cell-mean vs assigned colour,
      so AA *within* a subcell isn't charged — §0 criterion 4). Raw per-pixel MSE is
      dropped as the optimisation target.
- [x] **A4** `renderGlyphDocument` (uniform 2×4 → indexed RenderResult).
- [x] **A5** Tests green: non-degenerate shape fit, render-back, colour models,
      failure signal, exports (12 tests).
- [x] **A6** Exports: `glyphDocumentToText`, `glyphDocumentToJSON` + tests.
- [x] **A7** SPA wiring: `schema.ts` `GLYPHFIT_PARAMS`/style/RendererId,
      `studio.ts` `state.glyphDoc`+dispatch+gridRenderer+readout+`downloadGlyphJson`
      + **`state.showResidual` + "Show residual" toggle** (export stays the faithful
      recon via an offscreen, never the residual), `index.html` buttons.
- [x] **A8** (review r2) `glyphDocumentFromJSON` + a render-equivalence test (JSON
      round-trips to the SAME render, not just serialisation).
- [x] **B1** `resolveGrid` auto mode: cell size AND phase from the lattice
      (`detectGrid` + `gridOrigin`). On the hero auto finds 16×32, phase 3,1 itself.
      FINDING: a reconstruction-error grid SEARCH was implemented and **rejected** —
      on a sparse frame the normalised error is multimodal/flat (min at 16×31, not
      the true 16×32; phase search drifts off the dot lattice) so it traded the
      physically-correct grid for noise-level gains. The lattice detector is the
      right tool for cell size+phase; error is for the residual/readout. A robust
      *global* grid search (for images `detectGrid` can't seed) stays future work.
- [ ] **B2** (deferred) `sextant2x3`; `fg_bg` ANSI/HTML export; **composite /
      second grid** for the incommensurate frog (the remaining edge halo).

Mental model to preserve:
- **Glyph text** = shape layer (lossy colour)
- **Glyph document (JSON)** = shape + colour layer (faithful)
- **Rendered PNG** = expanded visual result

---

## 9. Key file/line references (as of this writing)

| What | Location |
| --- | --- |
| Repixel options | `colorworks/pages/src/repixel.ts:55` |
| Fine-lattice detect | `colorworks/pages/src/repixel.ts:162` |
| Subject detect | `colorworks/pages/src/repixel.ts:170` |
| `detectCandidates` (seed cell size) | `colorworks/pages/src/repixel.ts:176` |
| `globalBg`/`windowMode` (private — export) | `colorworks/pages/src/repixel.ts:186` |
| `windowMean` (subcell sampling pattern) | `colorworks/pages/src/repixel.ts:232` |
| `recover` (grid loop) | `colorworks/pages/src/repixel.ts:279` |
| `gridOrigin` use (phase before sampling) | `colorworks/pages/src/repixel.ts:308` |
| `renderRepixel` (→ RenderResult, palette dedupe) | `colorworks/pages/src/repixel.ts:355` |
| `brailleBit` (private — export, reuse) | `colorworks/pages/src/repixel.ts:396` |
| `toGlyphText` (the thing we're superseding) | `colorworks/pages/src/repixel.ts:405` |
| Core types `RGB`/`Raster`/`RenderResult` | `colorworks/pages/src/colorworks.ts:25,36,65` |
| `rasterToIndexed`, `quantizeToPalette`, `bestPhase`, `gridOrigin` | `colorworks/pages/src/depixelate.ts` |
| `REPIXEL_PARAMS` (param pattern) | `colorworks/pages/src/schema.ts:295` |
| `RendererId` / `StyleDef` / `STYLES` | `colorworks/pages/src/schema.ts:419,421,434` |
| `PARAM_BY_KEY` spread | `colorworks/pages/src/schema.ts:467` |
| Studio render dispatch | `colorworks/pages/src/studio.ts:392` |
| Repixel branch (clone for glyphfit) | `colorworks/pages/src/studio.ts:420` |
| `gridRenderer` test + `conformIndexed` | `colorworks/pages/src/studio.ts:405,449` |
| `conformIndexed`/`boxFit` (output-size owns scale) | `colorworks/pages/src/output_size.ts:64` |
| `state.glyphText` / `copyGlyphText` (model JSON handler on it) | `colorworks/pages/src/studio.ts:70,697` |
| `#copyGlyphs` hidden condition (widen to glyphfit) | `colorworks/pages/src/studio.ts:468` |
| Export UI row | `colorworks/pages/index.html:129` |
