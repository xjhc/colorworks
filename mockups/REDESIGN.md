# Colorworks UI redesign — design doc

**Status:** proposal · **Author:** (draft) · **Date:** 2026-05-28

## Goal

Make the obvious workflow — *"upload an image, get a low-res quantized render
out"* — a 3-control task. The current UI exposes algorithm phases and ink
layers as primary concepts; users have to learn the renderer before they can
use it. We want a product surface where the renderer is an implementation
detail.

## Two-tier UI

**Quick mode** (new default landing experience)
- Drop image
- Output size: max width / max height / fit mode
- Colors: count (2-8) + palette source (adaptive / grayscale / ink+paper / custom)
- Optional style filter chips: All / Dither / Halftone / Stipple / Pattern
- "Generate variants" → 4×2 candidate grid using sensible per-algorithm defaults
- Pick one → export, or "Refine in Studio"

**Studio mode** (today's UI, cleaned up)
- Rename pipelines to plain names (drop the `Phase X:` prefix)
- Add the same Output Size + Colors controls at the top of the parameters panel
- Reframe "Layers" as "Print plates (advanced)" with a one-line explainer
- Recipe/Preset machinery stays as-is — just demoted visually

Mockup: `mockups/quick-mode.html`

## Backend gaps

The UI changes are cheap. The backend has three real gaps that need to be filled
before Quick mode is honest.

### 1. Resize preprocessing stage

**Today:** renderers receive the source image at upload resolution.
**Needed:** a preprocessing stage that takes `(max_width, max_height, fit_mode)`
and resamples *before* the renderer runs.

Place it in `colorworks/algorithms/image_ops.py` (already exists) as a
`resize_for_output(image, max_w, max_h, fit)` helper. Wire it into the request
pipeline at the entry of both `/api/preview_runs` and `/api/render_runs`, and
extend `Recipe` (in `colorworks/recipe.py`) with an `output` field:

```python
@dataclass
class OutputSpec:
    max_width: int | None = None
    max_height: int | None = None
    fit: Literal["fit", "cover", "stretch"] = "fit"
    resample: Literal["lanczos", "nearest"] = "lanczos"
```

Cost: ~1 day. Pure addition. Backwards compatible (missing = original size).

### 2. Multi-color quantization pipeline

**Today:** every shipped algorithm is 1-bit (ink vs paper) or layered-ink. The
"Colors: 4" control has no obvious mapping.

**Two options, in order of cost:**

**(A) Add a `palette_quantize` pipeline** — new algorithm in
`colorworks/algorithms/palette_quantize.py` that:
- Resizes to target box.
- Uses `PIL.Image.quantize(colors=N, method=MEDIANCUT, dither=...)` for adaptive.
- For grayscale: convert to L, then `quantize(colors=N)`.
- For ink+paper: hand off to the existing FS/Bayer/DBS path with N=2.
- Returns a paletted image artifact.

This is the cleanest match for the user's mental model ("flat 4-color
pixel-art"). ~1-2 days.

**(B) Auto-layering of existing ink algorithms** — given `N=4` and an adaptive
palette, build 4 ink layers automatically, one per palette band, each rendered
through whichever 1-bit algorithm the user selected. This produces stylized
multi-color halftones, not flat pixel art.

This is *harder* and produces a different aesthetic. Worth doing later as a
"stylized" path, but **(A) is what Quick mode should ship with**.

**Recommendation:** ship (A) as `palette_quantize`. Add (B) later as a "Mixed
ink layers" style option once (A) proves the loop works.

### 3. Candidate generation endpoint

**Today:** the client submits one preview run at a time via `/api/preview_runs`.
**Needed:** one call → N renders in parallel, one per algorithm, returning
thumbnail URIs and run IDs for each.

```
POST /api/candidates
Body:
  {
    "asset_id": "abc123",
    "output": { "max_width": 200, "max_height": null, "fit": "fit" },
    "colors": { "count": 4, "palette": "adaptive" },
    "style_filter": ["dither", "halftone", "stipple"]   // optional
  }
Response:
  {
    "candidate_set_id": "cs_xyz",
    "candidates": [
      {
        "id": "cand_01",
        "algorithm": "floyd_steinberg",
        "label": "Floyd–Steinberg",
        "description": "Classic error diffusion",
        "params": { ... auto-tuned defaults ... },
        "preview_run_id": "pr_aaa",
        "thumb_url": "/api/preview_runs/pr_aaa/thumb"
      },
      ...
    ]
  }
```

Internally this fans out to the existing `RunScheduler` (in
`colorworks/scheduler.py`) — one preview run per candidate. Each candidate
streams progress via the existing SSE endpoint. The client renders skeletons
and replaces them with thumbs as each run finishes.

A small **algorithm-selector registry** decides which algorithms are
compatible with a given `(colors, palette, style_filter)` triple and what
default params they should run at. Live in
`colorworks/algorithms/quick_mode.py` as a pure-data table.

Cost: ~2-3 days.

## File-by-file migration

**New:**
- `mockups/quick-mode.html` *(done)*
- `colorworks/algorithms/palette_quantize.py`
- `colorworks/algorithms/quick_mode.py` — selector registry + default params
- `colorworks/web/static/quick.html` *or* a `mode=quick` query path off
  `index.html` (recommend the latter to share assets)

**Modified:**
- `colorworks/recipe.py` — add `OutputSpec`
- `colorworks/algorithms/image_ops.py` — add `resize_for_output`
- `colorworks/web/server.py` — add `POST /api/candidates`, thread `output`
  through existing preview/render endpoints
- `colorworks/web/static/index.html` — drop `Phase X:` prefixes; add Output
  panel; reframe Layers as "Print plates (advanced)"
- `colorworks/web/static/app.js` — handle Quick/Studio mode toggle, candidate
  grid rendering

**Untouched:**
- Comparison Gallery (it's a developer/validation tool)
- Preset/Recipe persistence
- The algorithm implementations themselves (FS, Bayer, CVT, DBS, SAED, Pang)

## Open questions

1. **Default palette for "Adaptive"** — Pillow's MEDIANCUT vs. our own
   k-means? Pillow is faster and good enough for a v1. Revisit if users find
   the palette choices ugly.
2. **Candidate count** — 8 feels right; we have 6-7 algorithms today. Should
   the grid grow if we add more, or should we cap and pick the top-N by some
   heuristic?
3. **Thumb size vs. fullsize render** — generate candidates at the *output*
   size (200px) and upscale with `image-rendering: pixelated`, or at a fixed
   thumb size? The first is more honest about what you'll get.
4. **Studio mode parity** — does Studio still expose the candidate grid as a
   secondary view, or is it Quick-only? (Suggestion: Quick-only; Studio is
   single-algorithm focus.)

## Sequencing

1. **Cheapest wins** (today): rename `Phase X:` labels; add Output panel to
   the existing UI; thread `OutputSpec` through `Recipe` and the renderers.
   *Ships value without committing to the redesign.*
2. **Palette quantizer**: new algorithm + UI exposure in Studio.
3. **Candidate endpoint + Quick mode UI**: the full new front door.
4. **Studio cleanup**: rename "Layers" → "Print plates (advanced)", reorder
   panels.
