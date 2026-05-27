# Colorworks

Colorworks is a local raster-to-print-study workbench for ordered dithering, tonal pattern composition, direct error-diffusion rendering, and structure-guided hatch export.

## Run

```sh
python -m colorworks.web.server
```

Then open the printed local URL. Runtime assets, recipes, presets, cached artifacts, output PNGs, and SVG exports are written under `./colorworks_data/` by default.

## Test

```sh
python -m pytest
```

## Current Scope

- Phase 0 ordered Bayer renderer.
- Phase 1 tonal analyzer with composited ink layers, presets, and pattern catalog.
- Phase 1B Floyd-Steinberg direct renderer.
- Phase 2 structure tensor / ETF orientation analyzer with hatch and crosshatch SVG export.
- Recipe and preset JSON saved to disk and reloadable.
- Content-addressed caches for intermediate artifacts and final outputs.
