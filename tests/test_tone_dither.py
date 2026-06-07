from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image

from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.algorithms import dither
from colorworks.algorithms.tone_dither import ToneDitherRenderer
from colorworks.domain import ArtifactStore, RasterGrid


def gradient(w: int = 48, h: int = 48) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    val = ((xx + yy) * 255.0 / (w + h - 2.0)).astype(np.uint8)
    return Image.fromarray(np.stack([val] * 3, axis=-1), mode="RGB")


# ── palettes ────────────────────────────────────────────────────────────────
def test_grayscale_palette_ordered_and_spanning():
    pal = dither.build_tone_palette(gradient(), 4, "grayscale")
    assert len(pal) == 4
    assert pal[0] == (0, 0, 0)
    assert pal[-1] == (255, 255, 255)
    lums = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pal]
    assert lums == sorted(lums)


def test_duotone_palette_ramps_between_endpoints():
    pal = dither.build_tone_palette(gradient(), 3, "duotone", ink_color="#000000", paper_color="#ffffff")
    assert pal[0] == (0, 0, 0)
    assert pal[-1] == (255, 255, 255)
    assert pal[1] == (128, 128, 128) or pal[1] == (127, 127, 127)


def test_adaptive_palette_count_and_ordering():
    pal = dither.build_tone_palette(gradient(), 5, "adaptive")
    assert len(pal) == 5
    lums = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pal]
    assert lums == sorted(lums)


def test_adaptive_palette_on_solid_image_does_not_crash():
    solid = Image.new("RGB", (16, 16), (90, 90, 90))
    pal = dither.build_tone_palette(solid, 4, "adaptive")
    assert len(pal) == 4


# ── threshold masks ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("fn,kwargs", [
    (dither.bayer_threshold_map, {"matrix_size": 8}),
    (dither.blue_noise_threshold_map, {"size": 32, "seed": 3}),
    (dither.maze_threshold_map, {"scale": 10.0, "line_width": 2.0, "seed": 3}),
    (dither.wave_threshold_map, {"frequency": 8.0, "angle_deg": 30.0}),
])
def test_threshold_maps_shape_and_range(fn, kwargs):
    m = fn(37, 29, **kwargs)
    assert m.shape == (29, 37)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_blue_noise_deterministic_with_seed():
    a = dither.blue_noise_threshold_map(40, 40, 32, seed=7)
    b = dither.blue_noise_threshold_map(40, 40, 32, seed=7)
    assert np.array_equal(a, b)


# ── quantisation ──────────────────────────────────────────────────────────────
def test_ordered_dither_levels_in_range():
    g = np.linspace(0, 1, 50 * 50).reshape(50, 50).astype(np.float32)
    mask = dither.bayer_threshold_map(50, 50, 8)
    lv = dither.ordered_dither_levels(g, 4, mask)
    assert lv.min() >= 0 and lv.max() <= 3
    # black stays level 0, white stays top level
    assert dither.ordered_dither_levels(np.zeros((8, 8), np.float32), 4, mask[:8, :8]).max() == 0
    assert dither.ordered_dither_levels(np.ones((8, 8), np.float32), 4, mask[:8, :8]).min() == 3


def test_error_diffuse_levels_in_range_and_tracks_tone():
    g = np.linspace(0, 1, 32 * 32).reshape(32, 32).astype(np.float32)
    lv = dither.error_diffuse_levels(g, 4)
    assert lv.min() == 0 and lv.max() == 3
    # mean level should increase with brightness
    assert lv[:, :8].mean() < lv[:, -8:].mean()


# ── full render ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["bayer", "blue_noise", "floyd_steinberg", "flow", "maze", "wave"])
def test_render_tone_dither_color_count(method):
    img = dither.render_tone_dither(gradient(), colors=4, palette_mode="grayscale", method=method, seed=1)
    assert img.size == (48, 48)
    assert len(img.getcolors(maxcolors=99999)) <= 4


def test_flow_threshold_map_follows_image():
    """Flow mask is image-dependent (domain-warped) and stays in [0, 1]."""
    g = gradient()
    import numpy as np
    arr = np.asarray(g.convert("L"), dtype=np.float32) / 255.0
    mask = dither.flow_threshold_map(arr, frequency=5.0, warp=7.0, angle_deg=45.0, detail=2.5)
    assert mask.shape == arr.shape
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
    # warp=0 (straight carrier) must differ from warp>0 (image-warped) → structure-aware
    flat = dither.flow_threshold_map(arr, frequency=5.0, warp=0.0, angle_deg=45.0, detail=2.5)
    assert not np.allclose(mask, flat)


def test_render_tone_dither_bad_method_falls_back():
    img = dither.render_tone_dither(gradient(), colors=4, method="nonsense")  # type: ignore[arg-type]
    assert isinstance(img, Image.Image)
    assert len(img.getcolors(maxcolors=99999)) <= 4


@pytest.mark.asyncio
async def test_tone_dither_renderer_publishes_final_raster():
    img = gradient(40, 40)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=RasterGrid(40, 40))
    ctx = RenderContext(
        input=asset,
        params={"colors": 4, "palette": "grayscale", "method": "bayer", "matrix_size": 8},
        composition=None,
        seed=42,
        store=ArtifactStore(),
    )
    algo = ToneDitherRenderer()
    evs = [p async for p in algo.render(ctx)]
    assert evs[-1].kind == "completed"
    art = ctx.store.get_by_name("final_raster")
    assert isinstance(art.value, Image.Image)
    assert art.value.size == (40, 40)


def test_tone_dither_registered():
    algo = registry.get("tone_dither")
    assert algo.definition.id == "tone_dither"


# ── compositor parity (shared mask source of truth) ──────────────────────────
def test_compositor_delegates_to_shared_masks():
    from colorworks.compositor import Compositor
    from colorworks.domain import PatternSpec, PatternCoordinateSpec
    c = Compositor(ArtifactStore())

    def ps(kind, params, seed=5):
        return PatternSpec(kind=kind, params=params, coordinates=PatternCoordinateSpec(seed=seed))

    assert np.allclose(
        c._generate_maze(ps("maze", {"scale": 12.0, "line_width": 2.0}), 30, 24, 5),
        dither.maze_threshold_map(30, 24, 12.0, 2.0, 5),
    )
    assert np.allclose(
        c._generate_blue_noise(ps("blue_noise", {"size": 32}), 30, 24, 5),
        dither.blue_noise_threshold_map(30, 24, 32, 5),
    )
    assert np.allclose(
        c._generate_ordered_dither(ps("ordered_dither", {"matrix_size": 8}), 30, 24, 5),
        dither.bayer_threshold_map(30, 24, 8),
    )
