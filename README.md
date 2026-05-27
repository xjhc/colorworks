# Colorworks

Phase 0 is a focused local web tool for one renderer: ordered dithering with a Bayer matrix.

## Run

```sh
python -m colorworks.web.server
```

Then open the printed local URL. Runtime assets, recipes, and output PNGs are written under `./colorworks_data/` by default.

## Test

```sh
python -m pytest
```

## Scope

- Single raster input.
- Single black-and-paper output.
- Bayer matrix size, threshold, and contrast controls.
- Synchronous local render requests.
- Recipe JSON saved to disk and reloadable.
- Export PNG uses the same content-addressed output shown in the UI.
