# Colorworks — handoff (4-colour dither studio rebuild)

This session rebuilt Colorworks around its actual purpose: turn a photo into
**true N-colour dithered art** (the yellow/navy "4-colour portrait" look) with an
artistic "flow" texture — and replaced the two-mode research UI with a single
seamless studio. Driven and verified end-to-end in a real browser.

## The core quality fix: "4 colours" now means 4 *distinct* colours

The old `tone_dither` adaptive path used median-cut (population-weighted) + a 1-D
luminance ramp. On a yellow-background portrait it spent two of four slots on
near-identical yellows → effectively ~3 colours, and collapsed chroma to a tonal
ramp. Now:

- **`kmeans_palette()`** (in `dither.py`) extracts N *perceptually distinct*,
  image-faithful colours (k-means++ on a downsample, deterministic by seed).
- **`dither_to_palette()`** dithers **in colour space** — each pixel picks
  between its two nearest palette colours via the threshold mask. **`fs_to_palette()`**
  does Floyd–Steinberg onto an arbitrary palette. So output uses exactly N
  distinct colours with dithered transitions, for every method.
- `render_tone_dither` now routes all palettes through colour-space dithering
  (gray/duotone ramps are unchanged in look; adaptive is dramatically better).
- Verified in-browser: 4→4 and 6→6 *distinct* served colours.

## New "Flow (waves)" method = the artistic waves

`flow_threshold_map()` is a directional sine carrier whose phase is **domain-warped
by the image's own blurred luminance**, so dither bands bend and flow around the
subject's features (face, hair) instead of sitting as a flat overlay. Knobs:
Wave Density, Flow Strength, Flow Angle, Flow Detail. This is the differentiated
look (vs. the removed maze/wave overlays).

## Algorithm surgery

- **Removed from the product surface:** halftone (`pang`) — muddy even at high
  density; and the `maze`/`wave` overlay methods — gimmicky. (Primitives kept in
  `dither.py`/`compositor.py` for parity tests; just not offered in the UI.)
- **Fixed & kept: Stippling (`cvt`).** Defaults were calibrated for tiny test
  images (300 dots) → blank noise on real frames. Now `n_stipples` defaults 2500
  / max 16000, and dot radius scales up when the working res was downsampled, so
  large exports stay covered. Reads as a real stipple portrait now.
- **Fixed `palette_quantize` dither no-op:** PIL ignores `dither=` with
  `method=MEDIANCUT`; now re-maps onto a fixed palette with explicit FS.
- Curated Quick set (`quick_mode.CANDIDATES`): Flow · Ordered (Bayer) · Blue
  Noise · Floyd–Steinberg · Flat poster · Stippling. Adaptive is the default
  palette.
- Versions bumped to `1.1.0` (tone_dither, palette_quantize, cvt_stippling) —
  semantically correct *and* busts the on-disk artifact cache (cache key includes
  `algo_version`; changing behaviour without bumping version serves stale renders).

## New unified UI (full redesign, `web/static/{index,app,styles}`)

One seamless surface, no modes, no research scaffolding (Comparison Gallery,
Recipes, analyzer pipelines, raw-artifact tabs, Print plates all gone):

1. **Left rail — setup:** source (drag/drop or click), output size, colour count
   + palette, style filter, Generate.
2. **Centre — stage:** a variant grid that streams in; click one and it promotes
   to a **big live preview** on a print "plate" (registration marks), with a
   filmstrip to switch variants.
3. **Right rail — adjust:** the schema-driven knobs for the selected variant
   (conditional visibility per method) + Export PNG with dims/render/checksum.

Aesthetic: risograph print-studio — warm newsprint, ink, vermilion spot colour,
grain + halftone atmosphere, registration motifs; Fraunces / Archivo / DM Mono.

### Bugs found & fixed by driving the browser
- `[hidden]` was overridden by `.empty{display:flex}` → empty + grid both showed.
  Added `[hidden]{display:none!important}`.
- Iterative (cvt) focus render crashed: `ink_color`/`paper_color` were `STR`
  without `ui_hint="color"`, so the knob builder made them sliders → invalid hex.
  Added the hint + a defensive text/colour fallback for string params.
- Iterative completed-event carries no dims/time → caption read them off the
  loaded image; render time measured client-side.

## Verified
- `python -m pytest` → **110 passed** (updated tone_dither/palette tests; added a
  flow-mask test).
- Full browser pass: upload → generate (6 variants) → Flow big + live knob edit
  (4→6 colours, exact distinct counts) → FS/Bayer/Stipple knob switching →
  Export filename/dims. Server runs at `127.0.0.1:8765` (`./serve`).

## Known follow-ups
- **Stippling focus re-render is slow (~15s)** at 360px (Lloyd is O(W·H·N), pure
  numpy). Fine while streaming in the grid; sluggish for live knob tweaks. The
  synchronous methods (flow/bayer/blue-noise/FS/flat) are instant. Could cap
  working res / iterations or move Lloyd to a faster kernel.
- Server output is buffered to the log; `print` cache HIT/MISS lines only flush
  on exit. Add `flush=True` if you need live logs.
