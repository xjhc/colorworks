# Colorworks GitHub Pages Plan

Build the next Colorworks surface in this repository, as a browser-only app
targeting GitHub Pages. Do not stage the app inside `knitlab2`; that repo can
consume the finished module later if needed.

## Direction

Colorworks should have two surfaces:

- The existing Python server: local research/workbench mode, useful for storage,
  backend algorithms, caches, comparison runs, and exploratory tooling.
- A new static Pages app: the product-facing client, with upload, live canvas
  preview, style controls, and PNG export. It runs entirely in the browser.

The Pages app owns the TypeScript port of the shipped dither pipeline. The
module is authored, tested, and deployed here first. Any future knitlab
integration should import, vendor, or copy that module intentionally after it is
stable; it should not be the primary build target.

## Proposed Layout

```text
colorworks/
  pages/                         # NEW - static GitHub Pages app
    index.html
    package.json                 # dev/build/preview/test
    package-lock.json            # required if workflow uses npm ci
    vite.config.ts               # base controlled by PAGES_BASE
    src/
      colorworks.ts              # canonical browser-safe TS algorithms
      colorworks.test.ts         # vitest suite
      studio.ts                  # vanilla TS + canvas UI
      styles.css
      blue-noise-tiles.ts        # deferred precomputed tiles
  .github/workflows/
    deploy-pages.yml             # NEW - build pages/ and deploy artifact
```

Use `pages/` rather than `colorworks/web/static/` so the static app can evolve
without being tied to the Python package data path. The Python server can keep
serving its current static UI while the Pages app is built and validated.

## Pages Build

The GitHub Actions workflow should publish a single Pages artifact from
`pages/dist`.

Skeleton:

```yaml
name: Deploy Colorworks to GitHub Pages

on:
  push:
    branches: [main, master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

env:
  # Project Pages default for a repo named colorworks.
  # Change to "/" when using a custom domain.
  PAGES_BASE: /colorworks/

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - name: Install
        working-directory: pages
        run: npm ci
      - name: Test
        working-directory: pages
        run: npm test
      - name: Build
        working-directory: pages
        run: npm run build -- --base="$PAGES_BASE"
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: pages/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

If the scaffold uses pnpm instead of npm, change the workflow and commit the
matching lockfile. Do not leave `npm ci` in the workflow without
`pages/package-lock.json`.

## Client App Scope

Port only the shipped client-facing path first:

- `tone_dither` with `flow`, `bayer`, `floyd_steinberg`, and `maze`.
- `flat` as deterministic k-means plus nearest assignment.
- `blue_noise` after precomputing and embedding 16/32/64/128 threshold tiles.
- Palette modes: adaptive, grayscale, duotone.
- Tone controls: colors 2-8, contrast, midpoint, ink, paper.
- Flow controls: frequency, warp, angle, detail.
- Bayer control: matrix size 2/4/8/16.
- Browser export: PNG from the preview canvas.

Out of scope for the first Pages build:

- Python server storage, recipes, caches, SSE, or backend jobs.
- Iterative halftoners such as CVT/Pang/DBS/SAED.
- Any knitlab2 modal integration.
- Dithering onto a knitting/yarn palette.

## Algorithm Module Contract

`pages/src/colorworks.ts` should be pure TypeScript:

- No DOM dependency in the core functions.
- No fetch, no backend, no Node-only APIs.
- Inputs are RGB cell/raster grids plus width/height/options.
- Outputs are `{ indices: number[], palette: RGB[] }`.
- Palette output is deduped and indices are reindexed.
- Include helpers like `rgbToHex` and `rgbToCss` for UI adapters.

Important fidelity notes from the Python implementation:

- Adaptive palette uses deterministic k-means++ and sorts by luma.
- Palette sampling should downsample/subsample like Python's max-96 palette pass
  so 360px previews do not make k-means unexpectedly heavy.
- Flow uses raw grayscale for the threshold mask and tone-remapped RGB for
  palette assignment.
- Floyd-Steinberg diffuses color-space error onto the chosen palette.
- Flat is not PIL median-cut parity unless a median-cut port is explicitly added.

## Tests

The Pages app should own the browser-module test suite:

```sh
cd pages
npm test
npm run build
```

Minimum tests:

- Palette dedupe collapses duplicate swatches and reindexes cells.
- `ditherToPalette(..., mask=null)` equals nearest-color assignment.
- Bayer threshold maps tile known 2x2 and 4x4 matrices.
- Floyd-Steinberg output stays in-palette and tracks mean tone within tolerance.
- Flow mask uses raw gray while palette assignment uses toned RGB.
- `renderToneDither` returns `rows * cols` indices and a unique palette.

Keep the existing Python suite for the server:

```sh
python -m pytest
```

## Local Dev

After scaffolding:

```sh
cd pages
npm install
npm run dev
```

Expected local workflow:

1. Upload an image.
2. The browser draws it into a canvas and scales to the selected max dimension.
3. Controls rerender the preview using `renderToneDither`.
4. Export writes the current canvas as PNG.

## Future Knitlab Integration

If the knitlab image modal later wants these algorithms, treat that as a separate
integration:

- Keep this repo's Pages app and `pages/src/colorworks.ts` as the source of truth.
- Consume a tagged copy, a vendored file, or a small published package.
- Add a knitlab-specific adapter that converts `{ indices, palette }` into its
  existing `rgb(r, g, b)` modal state.
- Preserve knitlab pixel-perfect import and existing flat import unless that
  integration explicitly decides otherwise.

That keeps Colorworks independently deployable and prevents the algorithm module
from being shaped around knitlab2's chart handler.
