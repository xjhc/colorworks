# Colorworks

A local web tool for turning raster images into low-resolution, palette-reduced
prints and halftone studies. Drop an image in, pick a target size and color
count, and pick from a gallery of rendered variants — or open the Studio for
single-algorithm fine-tuning.

## Run

```sh
./serve
```

That starts the server on `http://127.0.0.1:8765` and writes runtime data
(uploaded assets, presets, recipes, caches, outputs) to `./colorworks_data/`.

Override the port or host inline:

```sh
PORT=9000 ./serve
HOST=0.0.0.0 PORT=8765 ./serve
```

Or invoke the module directly if you don't want the wrapper:

```sh
python -m colorworks.web.server --host 127.0.0.1 --port 8765 --data-dir colorworks_data
```

HEIC/HEIF photos (e.g. straight from an iPhone) are transcoded to PNG on upload
via [`pillow-heif`](https://pypi.org/project/pillow-heif/) (`pip install
pillow-heif`). Without that package installed the server still runs, but HEIC
uploads are rejected as an unsupported image type.

## GitHub Pages target

The next product surface is a browser-only static app built in this repo and
deployed directly to GitHub Pages, separate from `knitlab2`. The implementation
plan lives in [COLORWORKS_PLAN.md](COLORWORKS_PLAN.md).

The current Python server remains the local research/workbench surface. The
Pages app owns the dependency-free TypeScript dither module and can be consumed
by other projects later if needed.

The app lives in [`colorworks/pages/`](colorworks/pages/) (vanilla TypeScript +
canvas, built with Vite):

```sh
cd colorworks/pages
npm install
npm run dev        # local studio at http://localhost:5173
npm test           # vitest suite for the algorithm core
npm run build      # production build into pages/dist
```

`src/colorworks.ts` is the browser-safe port of `render_tone_dither` (no DOM,
no fetch) returning `{ indices, palette }`; `src/studio.ts` is the canvas UI.
Blue-noise tiles are precomputed by `scripts/gen_blue_noise.py`. Pushing to
`main` builds and deploys to GitHub Pages via
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
(set `PAGES_BASE` for the served subpath; defaults to `/colorworks/`).

## How to use it

Colorworks has two modes, toggled from the top bar. New users land in Quick.

### Quick mode

The intended loop is:

1. **Upload an image** — click the source tile on the left.
2. **Set the output size** — `Max width`, optional `Max height`, and `Fit` /
   `Cover` / `Stretch`. Blank means use the source dimensions.
3. **Set the color count** — 2 to 16. Default 4. Pick `Adaptive`, `Grayscale`,
   `Duotone`, or supply an exact custom ink palette.
4. **Optionally filter by style** — All, Dither, Pattern, Halftone, Stipple.
5. **Generate variants** — every compatible algorithm runs in parallel and
   returns a thumbnail.
6. **Pick a rendering intent** — faithful palette, organic grain, graphic
   pattern, or print screen — then fine-tune the method if needed.
7. **Read the proof strip** — perceptual and low-frequency error, actual used
   colours, plate angles, and encoded PNG/WebP/JPEG sizes are measured from the
   current render before export.

Most of the gallery is powered by the **Dither (Multi-tone)** renderer, which
honours the colour count — set *Colors = 4* and the Ordered, Blue-Noise,
Floyd–Steinberg, and Wave cards all produce genuine 4-colour dithered
output (not just ink/paper). This is what produces the "N-colour photo broken
into dither texture" look. `Flat (Pixel-art)` gives a flat N-colour quantize
with no dither; the halftone/stipple cards (Pang, CVT) remain 1-bit and are
labelled "2-color".

### Studio mode

A single-algorithm workspace with every parameter exposed:

- **Pipeline** select — choose one algorithm.
- **Parameters** panel — dynamically rendered based on the algorithm schema.
- **Output size** panel — same controls as Quick.
- **Print plates (advanced)** — stack ink layers with patterns (hatch, wave,
  maze, blue-noise, etc.) over a paper color. Each layer can have its own
  pattern, density source, threshold, and blend mode.
- **Presets** — save and reload named parameter snapshots.
- **Recipes** — full project state (asset + parameters + composition) as a
  reloadable JSON file.
- **Export PNG / SVG** — SVG export needs at least one hatch or crosshatch
  layer.

## Algorithms

| ID | Style | Notes |
| -- | ----- | ----- |
| `tone_dither` | Dither | **N-colour dithering.** 2-16 colours × palette (grayscale / adaptive / duotone / exact custom) × method (ordered, candidate-mix, blue-noise, and error diffusion). The workhorse for the multi-colour dithered look. |
| `print_halftone` (Pages) | Halftone | Single spot or CMYK process plates with AM, FM, and hybrid screens, dot shape/gain controls, conventional plate angles, and overprint preview. |
| `palette_quantize` | Pixel art | 2-8 colors, median-cut, optional FS dither |
| `ordered_bayer` | Dither | Crisp grid threshold matrix |
| `floyd_steinberg` | Dither | Classic error diffusion |
| `saed` | Dither | Structure-Aware Error Diffusion, follows image gradients |
| `dbs` | Dither | Direct Binary Search, best quality, slowest |
| `cvt_stippling` | Stipple | Lloyd-relaxed dot field, iterative |
| `pang_halftoning` | Halftone | Newsprint-style adaptive dots |
| `tonal_analyzer` | Composited | Tonal wave with ink layers, supports SVG hatch export |
| `structure_analyzer` | Composited | Structure tensor / ETF orientation, hatch + crosshatch SVG |

In the Pages studio, a new image opens on **All styles**: every style rendered
side by side for the current image, palette, and output size (re-rendered as the
knobs change), so choosing a look is visual. Click a card to open it.

**Output size** sets the dither grain; **export scale** (1×/2×/4×/8×,
nearest-neighbour) sets the size of the exported file, so a 360 px dither can be
shared at 1440 px without changing its look. The on-screen preview never
resamples dots unevenly: whole-number zooms are crisp, fractional ones use sharp
bilinear, and reductions are averaged rather than aliased.

Adaptive palettes have a **palette spread** knob (default 0.35): k-means inks are
cluster averages, which sit inside the image's real range, so dithered output
loses its lights, darks, and saturation. Spread pushes each ink outward toward
colours the image actually contains (measured at viewing distance, every dither
method improves at 0.25–0.5). Flat poster ignores it.

Editing a palette swatch on a palette-aware method promotes the visible colours
to an exact custom palette and recalculates the dither — it does not merely
repaint the old index map. The palette block then offers **revert** to return to
the previous (e.g. adaptive) palette. Dithering is presented as an
aesthetic or physical-palette constraint, not as a compression guarantee; the
proof strip reports the actual browser-encoded sizes for each format.

## Data directory

```
colorworks_data/
  assets/          uploaded images (content-addressed)
  outputs/         exported PNGs (content-addressed)
  artifacts/       cached intermediate artifacts (tone maps, edge masks, etc.)
  presets/         saved presets
  recipes/         saved recipes
```

Safe to delete the cache subdirectories (`artifacts/`) at any time — they'll
be rebuilt on the next render.

## Test

```sh
python -m pytest
```
