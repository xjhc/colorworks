from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from colorworks.algorithms import MediaAsset, RenderContext
from colorworks.algorithms.depixelate import (
    DepixelateRenderer,
    detect_grid,
    depixelate,
    reduce_to_tiles,
)
from colorworks.domain import ArtifactStore, RasterGrid


def make_native() -> np.ndarray:
    """A 6x6 'native' sprite of distinct solid cells (strong cell-boundary edges)."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    # force strong contrast between neighbours so the grid is unambiguous
    base[::2, ::2] = [20, 20, 20]
    base[1::2, 1::2] = [230, 230, 230]
    return base


def upscale(native: np.ndarray, factor: int) -> Image.Image:
    img = Image.fromarray(native, "RGB")
    return img.resize((native.shape[1] * factor, native.shape[0] * factor), Image.Resampling.NEAREST)


def make_asset(img: Image.Image) -> MediaAsset:
    return MediaAsset(
        id="t",
        checksum="t",
        image=img,
        substrate=RasterGrid(img.width, img.height),
    )


def test_detect_grid_finds_upscale_factor():
    native = make_native()
    up = upscale(native, 10)
    grid = detect_grid(up)
    assert round(grid.pitch_x) == 10
    assert round(grid.pitch_y) == 10


def test_reduce_to_tiles_recovers_native_solid_cells():
    native = make_native()
    up = upscale(native, 10)
    # explicit pitch keeps the test independent of detection robustness
    out, _grid = depixelate(up, reducer="tiles", block=2, pitch=10)
    assert out.size == (native.shape[1] * 2, native.shape[0] * 2)  # block=2 -> 2x grid
    # each solid cell becomes a uniform 2x2 tile; top-left subpixels recover native
    rec = np.asarray(out)[::2, ::2]
    assert np.array_equal(rec, native)


def test_block_size_scales_output():
    native = make_native()
    up = upscale(native, 10)
    for block in (2, 3, 4):
        out, _ = depixelate(up, reducer="tiles", block=block, pitch=10)
        assert out.size == (native.shape[1] * block, native.shape[0] * block)


def test_checkerboard_tile_on_two_colour_cell():
    """A cell split 50/50 between two colours yields the o/x checkerboard at block=2."""
    native = np.zeros((1, 2, 3), np.uint8)
    native[0, 0] = [0, 0, 0]
    native[0, 1] = [255, 255, 255]
    # upscale and place the two native pixels side by side inside ONE detected cell
    # by treating the whole 2x1 native as a single cell of pitch = full width.
    up = upscale(native, 10)  # 20x10, two solid halves
    from colorworks.algorithms.depixelate import Grid
    grid = Grid(pitch_x=20.0, pitch_y=10.0, phase_x=0.0, phase_y=0.0, confidence=1.0)
    tile = np.asarray(reduce_to_tiles(up, grid, block=2, tau=45))
    assert tile.shape == (2, 2, 3)
    # diagonal pair equal, anti-diagonal equal, and the two differ -> checkerboard
    assert np.array_equal(tile[0, 0], tile[1, 1])
    assert np.array_equal(tile[0, 1], tile[1, 0])
    assert not np.array_equal(tile[0, 0], tile[0, 1])


@pytest.mark.asyncio
async def test_renderer_end_to_end():
    native = make_native()
    up = upscale(native, 10)
    ctx = RenderContext(
        input=make_asset(up),
        params={"block": 2, "tau": 45, "pitch": 10},
        composition=None,
        seed=0,
        store=ArtifactStore(),
    )
    algo = DepixelateRenderer()
    events = [p async for p in algo.render(ctx)]
    assert events[-1].kind == "completed"

    art = ctx.store.get_by_name("final_raster")
    assert art.type == "raster_image"
    assert isinstance(art.value, Image.Image)
    assert art.value.size == (12, 12)  # 6x6 cells, block=2
