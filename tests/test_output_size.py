"""Export-size alignment: a render at a requested output size produces an
artifact (and therefore an exported PNG) of exactly that size.

The web server applies ``resize_for_output`` to the source before handing the
asset to a renderer, so the rendered ``final_raster`` — which is what the SPA
loads into its export canvas — must match the resized dimensions. These tests
pin that contract for the renderers the studio actually exposes.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from colorworks.algorithms import MediaAsset, RenderContext
from colorworks.algorithms.image_ops import ResizeSpec, resize_for_output
from colorworks.algorithms.palette_quantize import PaletteQuantizeRenderer
from colorworks.algorithms.tone_dither import ToneDitherRenderer
from colorworks.domain import ArtifactStore, RasterGrid

# Re-export so the renderers register themselves on import (registry side effect).
_ = (PaletteQuantizeRenderer, ToneDitherRenderer)


def _source(w: int = 1000, h: int = 750) -> Image.Image:
    """A deterministic colourful source so palette extraction has something to do."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _asset_at_output(src: Image.Image, spec: ResizeSpec) -> tuple[MediaAsset, tuple[int, int]]:
    """Resize like the server does, then wrap as an asset; return (asset, size)."""
    img = resize_for_output(src, spec)
    asset = MediaAsset(id="t", checksum="t", image=img, substrate=RasterGrid(img.width, img.height))
    return asset, img.size


async def _final_size(algo, asset: MediaAsset, params: dict) -> tuple[int, int]:
    ctx = RenderContext(input=asset, params=params, composition=None, seed=0, store=ArtifactStore())
    events = [e async for e in algo.render(ctx)]
    assert events[-1].kind == "completed"
    art = ctx.store.get_by_name("final_raster")
    assert isinstance(art.value, Image.Image)
    return art.value.size


SPECS = [
    (ResizeSpec(max_width=200, fit="fit"), (200, 150)),
    (ResizeSpec(max_width=200, max_height=200, fit="fit"), (200, 150)),
    (ResizeSpec(max_width=200, max_height=200, fit="cover"), (200, 200)),
    (ResizeSpec(max_width=200, max_height=200, fit="stretch"), (200, 200)),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("spec,expected", SPECS)
async def test_tone_dither_matches_requested_output_size(spec, expected):
    asset, size = _asset_at_output(_source(), spec)
    assert size == expected  # resize itself honours the request
    out = await _final_size(ToneDitherRenderer(), asset, {"method": "bayer", "colors": 4})
    assert out == expected  # ...and the render preserves it end-to-end


@pytest.mark.asyncio
@pytest.mark.parametrize("spec,expected", SPECS)
async def test_palette_quantize_matches_requested_output_size(spec, expected):
    asset, _size = _asset_at_output(_source(), spec)
    out = await _final_size(PaletteQuantizeRenderer(), asset, {"colors": 4, "dither": False})
    assert out == expected
