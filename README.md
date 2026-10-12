# Colorworks

Turn a photo into a few-colour print: dithered, halftoned, tiled into blocks,
or charted for knitting. The main surface is a browser-only studio deployed to
GitHub Pages at <https://xjhc.github.io/colorworks/>. Everything renders on your
machine; nothing is uploaded. A Python server in the same repo is the local
research workbench.

## The studio (GitHub Pages)

### How to use it

1. **Open a photo.** Choose one, drop it anywhere on the window, or paste it
   (Ctrl/⌘ V). PNG, JPEG, WebP, GIF, AVIF, SVG, TIFF, and HEIC all decode
   locally.
2. **Compare every style.** A new image opens on **All styles**: each style
   rendered side by side for the current image, palette, and output size,
   grouped by intent (faithful palette, organic grain, graphic pattern, print
   screen, more workflows). The cards re-render as the knobs change. Click one
   to open it.
3. **Set the grain.** **Output size** (max W/H, fit/cover/stretch) is the
   resolution the dither is computed at, and so the size of its dots. It's
   shared by every image style. Low sizes (~150–250 px) give the chunky look.
4. **Turn the knobs.** Colours, palette (adaptive, grayscale, duotone, or exact
   custom inks), palette spread, sharpen, contrast, and the method's own
   parameters. **Colour intent** pins steer adaptive palettes before
   generation: *lock* forces a colour in, *boost* favours its hue, *exclude*
   keeps it out.
5. **Recolour.** Hover a swatch to isolate that ink; click it to change it.
6. **Export.** **Download PNG** above the preview, or PNG/WebP in the proof
   strip. **Export scale** (1×/2×/4×/8×) enlarges the file with
   nearest-neighbour pixels, so a 360 px dither can be shared at 1440 px
   without changing its look.

The session (style, knobs, pins, inks, export scale, and the image itself) is
kept in the browser, so a refresh picks up where you left off.

### What the controls do

**Palette spread** (adaptive palettes, default 0.35). K-means inks are cluster
averages, so they sit inside the image's real range and dithered output loses
its lights, darks, and saturation. Spread pushes each ink outward toward
colours the image actually contains. Measured at viewing distance, every
dither method improves at 0.25–0.5. Flat poster ignores it.

**Sharpen** (tone dithers, default 0.5) is an unsharp mask on the resized
image before dithering, since downscaling softens edges and dither texture
blurs them further. It is clamped to each pixel's neighbourhood range, so soft
photo edges steepen but hard graphic edges don't grow halos. The palette is
still chosen from the unsharpened image.

**Swatch edits.** On palette-aware methods (the tone dithers), editing a
swatch promotes the visible colours to an exact custom palette and
recalculates the dither, rather than repainting the old index map. The
palette block then offers **revert** to return to the previous (e.g.
adaptive) palette. Other renderers remap the result's colours, and **reset**
undoes that.

**Preview fidelity.** The on-screen preview never resamples dots unevenly:
whole-number zooms are crisp, fractional ones use sharp bilinear, and
reductions are averaged rather than aliased.

**Proof strip.** Measured from the current render:

- **error at a distance**: OKLab ΔE×100 between source and render after both
  are blurred in linear light (σ 1.5 px), i.e. what the dots add up to. A
  pixel-by-pixel comparison would rank a flat poster above every dither.
- **tone drift**: average difference over 8 px blocks (broad areas coming out
  lighter, darker, or shifted).
- used colours, render time, and the exact export dimensions.

**Export formats** are both lossless; JPEG and lossy WebP smear dither dots.
PNG is written as indexed colour (1–8 bits per pixel) straight from the index
map, typically several times smaller than a canvas PNG. Knit charts with
symbols are the exception: their symbols are anti-aliased text. WebP is
offered where the browser can encode it losslessly (Chrome can). The PNG's
SHA-256 is in its card's tooltip.

### Develop

The app lives in [`colorworks/pages/`](colorworks/pages/) (vanilla TypeScript +
canvas, built with Vite):

```sh
cd colorworks/pages
npm install
npm run dev        # local studio at http://localhost:5173
npm test           # vitest suite for the algorithm core
npm run build      # typecheck + production build into pages/dist
```

- `src/colorworks.ts`: the dependency-free dither core (no DOM, no fetch),
  returning `{ indices, palette }`. It is the browser port of
  `render_tone_dither`, plus palette spread and sharpening.
- `src/halftone.ts`, `blockmosaic.ts`, `knit.ts`, `depixelate.ts`: the other
  renderers. `repixel.ts` is wired up but unlisted.
- `src/schema.ts`: styles and knobs. `src/studio.ts`: the canvas UI.
- `src/proof.ts`: proof metrics. `src/png.ts`: the indexed-PNG encoder.
- Blue-noise tiles are precomputed by `scripts/gen_blue_noise.py`.

Pushing to `main` tests, builds, and deploys to GitHub Pages via
[`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
(set `PAGES_BASE` for the served subpath; defaults to `/colorworks/`). The
original plan is in [COLORWORKS_PLAN.md](COLORWORKS_PLAN.md).

## Python workbench (local research server)

The Python package holds the original algorithms, including several the
browser app doesn't have (DBS, SAED, CVT stippling, Pang halftoning, the
tonal/structure analyzers with SVG hatch export). It also holds the server
and API that research scripts use for renders, presets, recipes, SVG export,
and comparisons. Its own page is an earlier, smaller version of the studio.

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

### Algorithms

| ID | Style | Notes |
| -- | ----- | ----- |
| `tone_dither` | Dither | **N-colour dithering.** 2-16 colours × palette (grayscale / adaptive / duotone / exact custom) × method (ordered, candidate-mix, blue-noise, and error diffusion). The workhorse for the multi-colour dithered look; the browser port adds palette spread and sharpening. |
| `palette_quantize` | Pixel art | 2-8 colors, median-cut, optional FS dither |
| `ordered_bayer` | Dither | Crisp grid threshold matrix |
| `floyd_steinberg` | Dither | Classic error diffusion |
| `saed` | Dither | Structure-Aware Error Diffusion, follows image gradients |
| `dbs` | Dither | Direct Binary Search, best quality, slowest |
| `cvt_stippling` | Stipple | Lloyd-relaxed dot field, iterative |
| `pang_halftoning` | Halftone | Newsprint-style adaptive dots |
| `tonal_analyzer` | Composited | Tonal wave with ink layers, supports SVG hatch export |
| `structure_analyzer` | Composited | Structure tensor / ETF orientation, hatch + crosshatch SVG |

### Data directory

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

### Test

```sh
python -m pytest
```
