from __future__ import annotations

"""Recover the native resolution of an upscaled pixel-art / dithered image.

The input is assumed to have been authored on a small pixel grid (a sprite plus
dither marks placed on the same grid) and then blown up by some roughly-integer
factor. This module finds that grid (pitch + phase, per axis) and re-renders each cell as
a small ordered-dither tile (or a single pixel) -- "one dot -> one pixel". It is
the inverse of a nearest-neighbour upscale.

The grid detection + reduction are pure numpy/PIL functions; a thin
``DepixelateRenderer`` (registered at import) exposes them as an app algorithm.
Run the standalone CLI with ``python -m colorworks.algorithms.depixelate <img>``.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext
from colorworks.algorithms.image_ops import parse_color
from colorworks.domain import (
    AlgorithmDefinition,
    AlgorithmFamily,
    AlgorithmRole,
    InputSpec,
    OutputSpec,
    ParameterDef,
    ParameterType,
    OptionDef,
    ArtifactKindDef,
    ArtifactViewerSpec,
    ExecutionProfile,
    AlgorithmCapabilities,
    RenderResult,
)


@dataclass(frozen=True)
class Grid:
    pitch_x: float      # cell width in source pixels
    pitch_y: float      # cell height in source pixels
    phase_x: float      # source x of the first cell's left boundary
    phase_y: float      # source y of the first cell's top boundary
    confidence: float   # 0..1, peak-to-mean contrast of the boundary comb


def _edge_profiles(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis boundary-energy profiles from colour gradients.

    Returns (profile_x, profile_y): profile_x[i] is the edge energy at column i
    (peaks at vertical cell boundaries), profile_y[j] likewise for rows.

    When the image carries a colourful subject on a grey/dark field, edges are
    weighted by saturation so the *subject's* grid wins over any competing grid in
    the grey background (compression blocks, dither fields, UI chrome) -- those sit
    on a different pitch and would otherwise dominate a plain luminance projection.
    Falls back to unweighted colour gradient for near-grayscale images.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    sat = arr.max(2) - arr.min(2)                      # per-pixel saturation
    dx = np.abs(np.diff(arr, axis=1)).sum(2)           # (H, W-1)  edges along x
    dy = np.abs(np.diff(arr, axis=0)).sum(2)           # (H-1, W)  edges along y

    if int((sat > 60).sum()) > 2000:
        wx = np.maximum(sat[:, :-1], sat[:, 1:])
        wy = np.maximum(sat[:-1, :], sat[1:, :])
        return (dx * wx).sum(0).astype(np.float64), (dy * wy).sum(1).astype(np.float64)
    return dx.sum(0).astype(np.float64), dy.sum(1).astype(np.float64)


def _fundamental(signal: np.ndarray, min_p: int, max_p: int) -> int:
    """Coarse fundamental period via autocorrelation.

    Picks the *smallest* lag whose autocorrelation peak is at least half the
    strongest peak -- not the global argmax. Harmonics (2x, 3x...) often correlate
    marginally better than the true pitch, so argmax lands on them; the fundamental
    is the shortest period that already explains most of the periodicity.
    """
    s = signal - signal.mean()
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-12)

    hi = min(max_p, len(ac) - 2)
    peaks = [(lag, ac[lag]) for lag in range(min_p, hi + 1)
             if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]]
    if not peaks:
        return min_p
    strongest = max(v for _, v in peaks)
    for lag, val in peaks:          # peaks are in ascending-lag order
        if val >= 0.5 * strongest:
            return lag
    return peaks[0][0]


def _comb_score(signal: np.ndarray, pitch: float, phase: float) -> float:
    """Mean boundary energy sampled by a comb of period `pitch` at `phase`.

    Normalised per-tooth so shorter pitches (more teeth) aren't unfairly favoured
    -- the bias that makes a naive sum collapse onto harmonics.
    """
    idx = np.round(np.arange(phase, len(signal), pitch)).astype(int)
    idx = idx[idx < len(signal)]
    if idx.size == 0:
        return 0.0
    return float(signal[idx].sum() / idx.size)


def _best_phase(signal: np.ndarray, pitch: float) -> tuple[float, float]:
    """Sub-pixel phase maximising the comb score for a given pitch."""
    phases = np.arange(0.0, pitch, 0.5)
    scores = [_comb_score(signal, pitch, ph) for ph in phases]
    i = int(np.argmax(scores))
    return float(phases[i]), scores[i]


def detect_grid(
    image: Image.Image,
    min_pitch: int = 6,
    max_pitch: int = 64,
    square: bool = True,
) -> Grid:
    """Detect cell pitch and phase from an image.

    Coarse pitch comes from autocorrelation of the (saturation-weighted) edge
    profiles; it is then refined to sub-pixel precision with a comb search in a
    narrow window -- important because a 0.25px pitch error drifts a whole cell
    across a wide image, and a narrow window keeps the refinement from jumping to a
    harmonic. `square=True` forces one common pitch for both axes -- the right prior
    for pixel-art / dither grids, and what keeps the aspect ratio from collapsing
    when one axis locks onto a harmonic.
    """
    px_prof, py_prof = _edge_profiles(image)

    px0 = _fundamental(px_prof, min_pitch, max_pitch)
    py0 = _fundamental(py_prof, min_pitch, max_pitch)

    if square:
        coarse = min(px0, py0)
        px = py = _refine([px_prof, py_prof], coarse)
    else:
        px = _refine([px_prof], px0)
        py = _refine([py_prof], py0)

    phx, score_x = _best_phase(px_prof, px)
    phy, score_y = _best_phase(py_prof, py)

    contrast = 0.5 * (score_x / (px_prof.mean() + 1e-12) + score_y / (py_prof.mean() + 1e-12))
    confidence = float(np.clip((contrast - 1.0) / 4.0, 0.0, 1.0))

    return Grid(float(px), float(py), float(phx), float(phy), confidence)


def _refine(profiles: list[np.ndarray], coarse: int) -> float:
    """Sub-pixel pitch maximising combined comb score within +/-15% of `coarse`.

    The narrow window is deliberate: across the full range the per-tooth comb score
    drifts toward large pitches (fewer teeth -> inflated max), so we only trust it
    to sharpen an already-correct coarse estimate.
    """
    lo, hi = coarse * 0.85, coarse * 1.15
    best = (float(coarse), -np.inf)
    for p in np.arange(lo, hi, 0.05):
        s = sum(_best_phase(prof, p)[1] for prof in profiles)
        if s > best[1]:
            best = (float(p), s)
    return best[0]


def _grid_origin(grid: Grid) -> tuple[float, float]:
    """Phase normalised to the nearest-zero grid origin, per axis.

    `_best_phase` locks onto the boundary-energy comb, which can land near the far
    edge of a cell (e.g. phase ~ pitch-1). Treating that as the first cell's left
    edge would drop the leading cell, so fold any phase past the half-pitch back by
    one pitch to keep the origin near 0 and cover the whole image.
    """
    ox = grid.phase_x - grid.pitch_x if grid.phase_x > grid.pitch_x / 2 else grid.phase_x
    oy = grid.phase_y - grid.pitch_y if grid.phase_y > grid.pitch_y / 2 else grid.phase_y
    return ox, oy


def _cell_mode(flat: np.ndarray) -> np.ndarray:
    """Most common RGB(A) tuple among (N, C) pixels."""
    uniq, counts = np.unique(flat, axis=0, return_counts=True)
    return uniq[np.argmax(counts)]


def _global_bg(arr: np.ndarray) -> np.ndarray:
    """Dominant (background) colour, estimated from a subsample for speed."""
    samp = arr[::3, ::3].reshape(-1, arr.shape[-1])
    return _cell_mode(samp).astype(np.int32)


def _two_colors(flat: np.ndarray, tau: int, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """The cell's two representative colours (c0, c1).

    c0 is the majority colour; c1 is the dominant colour among pixels far enough
    from c0 to count as a second colour. A solid cell returns (c0, c0).
    """
    c0 = _cell_mode(flat).astype(np.int32)
    far = flat[np.abs(flat.astype(np.int32) - c0).max(1) > tau]
    c1 = _cell_mode(far) if far.shape[0] >= min_count else c0.astype(flat.dtype)
    return c0.astype(flat.dtype), c1


def _bayer(n: int) -> np.ndarray:
    """Recursive Bayer (dispersed-dot) matrix, ranks 0..n*n-1. n must be 2**k."""
    if n == 1:
        return np.zeros((1, 1), dtype=np.int64)
    h = _bayer(n // 2)
    return np.block([[4 * h, 4 * h + 2], [4 * h + 3, 4 * h + 1]])


@lru_cache(maxsize=32)
def _dither_order(n: int) -> np.ndarray:
    """An n x n ordering, ranks 0..n*n-1, dispersed so low ranks spread out.

    Exact Bayer for power-of-two n; for other sizes (e.g. 3) sample a 16x16 Bayer
    on an n x n lattice and rank, which keeps the dispersed-dot character.
    """
    if n & (n - 1) == 0:
        return _bayer(n)
    base = _bayer(16)
    idx = np.linspace(0, 15, n).round().astype(int)
    sub = base[np.ix_(idx, idx)].ravel()
    return np.argsort(np.argsort(sub)).reshape(n, n)


def _cell_two_colors_indexed(
    idx_cell: np.ndarray, palette: np.ndarray, min_count: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two dominant palette colours of a cell from its palette-index window."""
    counts = np.bincount(idx_cell, minlength=palette.shape[0])
    o = np.argsort(counts)[::-1]
    i0, i1 = int(o[0]), int(o[1]) if palette.shape[0] > 1 else int(o[0])
    n0, n1, total = int(counts[i0]), int(counts[i1]), int(idx_cell.size)
    c0 = palette[i0]
    c1 = palette[i1] if (i1 != i0 and n1 >= min_count) else c0
    return c0, c1, (total - n0) / total


def reduce_to_tiles(
    image: Image.Image,
    grid: Grid,
    block: int = 2,
    tau: int = 45,
    min_frac: float = 0.04,
    keep_marks: bool = False,
    palette: np.ndarray | None = None,
    indices: np.ndarray | None = None,
) -> Image.Image:
    """Render each grid cell as a `block` x `block` two-colour ordered-dither tile.

    Output is `block`x the native cell grid. Per cell we take two representative
    colours c0 (majority) / c1 (the mark) and fill the tile by an ordered dither:
    the number of c1 subpixels is proportional to c1's coverage of the cell, placed
    on the lowest ranks of a dispersed (Bayer) pattern. So the tile self-selects:

      - solid cell            -> uniform tile (all c0)
      - midtone two-colour    -> checkerboard-like spread (at block=2, f~0.5, the
                                 classic ``c0 c1 / c1 c0``)

    With honest proportional fill a near-solid cell rounds to 0 and stays solid.
    `keep_marks=True` re-floors it to one c1 subpixel so a sparse mark (a star)
    survives instead of vanishing. When `palette`/`indices` are given the two cell
    colours are taken over the quantised palette; otherwise raw colours via `tau`.

    `block` is the configurable tile size; `tau` is the colour distance that counts
    as a second colour; `min_frac` is the cell fraction it must cover.
    """
    arr = np.asarray(image)
    H, W = arr.shape[:2]
    C = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, None]

    order = _dither_order(block)
    n_sub = block * block

    px, py = grid.pitch_x, grid.pitch_y
    ox, oy = _grid_origin(grid)
    n_cols = int(round((W - ox) / px))
    n_rows = int(round((H - oy) / py))

    out = np.zeros((n_rows * block, n_cols * block, C), dtype=arr.dtype)
    half_x = max(1, int(px * 0.9 / 2))
    half_y = max(1, int(py * 0.9 / 2))

    for r in range(n_rows):
        cy = int(round(oy + (r + 0.5) * py))
        y0, y1 = max(0, cy - half_y), min(H, cy + half_y + 1)
        ro = r * block
        for c in range(n_cols):
            cx = int(round(ox + (c + 0.5) * px))
            x0, x1 = max(0, cx - half_x), min(W, cx + half_x + 1)
            cell = arr[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            flat = cell.reshape(-1, C)
            n_px = flat.shape[0]
            min_count = max(2, int(min_frac * n_px))

            if palette is not None and indices is not None:
                c0, c1, frac = _cell_two_colors_indexed(
                    indices[y0:y1, x0:x1].ravel(), palette, min_count
                )
            else:
                c0, c1 = _two_colors(flat, tau, min_count)
                frac = float(np.mean(np.abs(flat.astype(np.int32) - c0).max(1) > tau))

            if np.array_equal(c0, c1):
                on = 0                                  # single colour -> solid tile
            else:
                base = round(frac * n_sub)
                if keep_marks:
                    base = max(1, base)                 # preserve a sparse mark
                on = int(np.clip(base, 0, n_sub - 1))

            tile = np.where((order < on)[:, :, None], c1, c0).astype(arr.dtype)
            out[ro:ro + block, c * block:c * block + block] = tile

    mode = "RGBA" if C == 4 else ("RGB" if C == 3 else "L")
    if C == 1:
        out = out[:, :, 0]
    return Image.fromarray(out, mode=mode)


def _build_palette(image: Image.Image, colors: int, mode: str, ink: str, paper: str) -> np.ndarray:
    """Build an (N, 3) uint8 palette matching the studio's palette modes."""
    colors = max(2, int(colors))
    if mode == "grayscale":
        ramp = np.linspace(0, 255, colors)
        return np.stack([ramp, ramp, ramp], axis=1).round().astype(np.uint8)
    if mode == "duotone":
        a = np.array(parse_color(ink), dtype=np.float64)
        b = np.array(parse_color(paper), dtype=np.float64)
        t = np.linspace(0, 1, colors)[:, None]
        return (a * (1 - t) + b * t).round().astype(np.uint8)
    pal_img = image.convert("RGB").quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    pal = np.asarray(pal_img.getpalette()[: colors * 3], dtype=np.uint8).reshape(-1, 3)
    return pal


def _quantize_to_palette(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Per-pixel nearest-palette index for an (H, W, 3) array."""
    a = arr.reshape(-1, 1, 3).astype(np.int32)
    p = palette.reshape(1, -1, 3).astype(np.int32)
    return ((a - p) ** 2).sum(2).argmin(1).reshape(arr.shape[:2]).astype(np.int32)


# Reducers that need to preserve sub-cell marks scan a wider window than the
# centre-crop `mode` uses -- a lone star can sit anywhere in the cell, not just
# its centre, so a tight crop would miss it.
_MARK_REDUCERS = ("foreground", "marks")


def reduce_to_native(
    image: Image.Image,
    grid: Grid,
    reducer: str = "marks",
    inner: float = 0.5,
    tau: int = 45,
) -> Image.Image:
    """Collapse each grid cell to one output pixel.

    reducer:
      - "mode"       most common colour -- crisp solids, but discards sparse marks
                     (a star is a minority in its cell, so it gets voted away).
      - "marks"      (default) surface a sub-cell mark on a background-dominant cell,
                     else fall back to mode. Keeps stars and the dither stipple while
                     keeping solid regions and dark features (eyes) intact.
      - "foreground" take the single pixel farthest from the global background.
                     Maximally preserves marks; fills dithered regions to solid.
      - "median" / "center"  per-cell median / centre sample.

    inner: fraction of the cell (centred) sampled by `mode`/`median`, so AA / JPEG
    cell edges don't pollute the colour. Mark reducers always scan ~the full cell.
    tau: per-channel distance from background above which a pixel counts as a mark.
    """
    arr = np.asarray(image)
    H, W = arr.shape[:2]
    C = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, None]

    bg = _global_bg(arr) if reducer in _MARK_REDUCERS else None

    px, py = grid.pitch_x, grid.pitch_y
    ox, oy = _grid_origin(grid)
    n_cols = int(round((W - ox) / px))
    n_rows = int(round((H - oy) / py))

    out = np.zeros((n_rows, n_cols, C), dtype=arr.dtype)
    eff_inner = 0.9 if reducer in _MARK_REDUCERS else inner
    half_x = max(1, int(px * eff_inner / 2))
    half_y = max(1, int(py * eff_inner / 2))

    for r in range(n_rows):
        cy = int(round(oy + (r + 0.5) * py))
        y0, y1 = max(0, cy - half_y), min(H, cy + half_y + 1)
        for c in range(n_cols):
            cx = int(round(ox + (c + 0.5) * px))
            x0, x1 = max(0, cx - half_x), min(W, cx + half_x + 1)
            block = arr[y0:y1, x0:x1]
            if block.size == 0:
                continue
            flat = block.reshape(-1, C)

            if reducer == "center":
                out[r, c] = arr[min(cy, H - 1), min(cx, W - 1)]
            elif reducer == "median":
                out[r, c] = np.median(flat, axis=0)
            elif reducer == "foreground":
                dist = np.abs(flat.astype(np.int32) - bg).max(1)
                out[r, c] = flat[np.argmax(dist)]
            elif reducer == "marks":
                dist = np.abs(flat.astype(np.int32) - bg).max(1)
                far = flat[dist > tau]
                if far.shape[0] == 0:
                    out[r, c] = bg.astype(arr.dtype)            # empty cell -> background
                elif far.shape[0] < 0.5 * flat.shape[0]:
                    out[r, c] = _cell_mode(far)                 # bg-dominant cell -> surface the mark
                else:
                    out[r, c] = _cell_mode(flat)                # solid feature -> plain mode
            else:  # mode
                out[r, c] = _cell_mode(flat)

    mode = "RGBA" if C == 4 else ("RGB" if C == 3 else "L")
    if C == 1:
        out = out[:, :, 0]
    return Image.fromarray(out, mode=mode)


def depixelate(
    image: Image.Image,
    reducer: str = "tiles",
    pitch: float | None = None,
    block: int = 2,
    tau: int = 45,
    palette: str = "original",
    colors: int = 4,
    ink_color: str = "#161616",
    paper_color: str = "#f4ebd9",
    keep_marks: bool = False,
) -> tuple[Image.Image, Grid]:
    """Convenience: detect grid (or use an explicit `pitch`) + reduce.

    reducer="tiles" (default) renders each cell as a `block` x `block` two-colour
    ordered-dither tile (output is `block`x the native grid); any other reducer
    collapses each cell to a single pixel via `reduce_to_native`.

    palette="original" keeps source colours (two-dominant via `tau`); "adaptive" /
    "grayscale" / "duotone" quantise to a `colors`-entry palette first. `keep_marks`
    re-floors a sparse cell to one minority subpixel so it isn't dropped.

    Pass `pitch` to override auto-detection. Returns (native_image, grid).
    """
    if pitch is not None:
        px_prof, py_prof = _edge_profiles(image)
        phx, _ = _best_phase(px_prof, pitch)
        phy, _ = _best_phase(py_prof, pitch)
        grid = Grid(float(pitch), float(pitch), phx, phy, 1.0)
    else:
        grid = detect_grid(image)

    if reducer not in ("tiles", "checker"):
        return reduce_to_native(image, grid, reducer=reducer, tau=tau), grid

    if palette == "original":
        native = reduce_to_tiles(image, grid, block=block, tau=tau, keep_marks=keep_marks)
    else:
        rgb = image.convert("RGB")
        pal = _build_palette(rgb, colors, palette, ink_color, paper_color)
        idx = _quantize_to_palette(np.asarray(rgb), pal)
        native = reduce_to_tiles(rgb, grid, block=block, keep_marks=keep_marks, palette=pal, indices=idx)
    return native, grid


# ── App algorithm wrapper ─────────────────────────────────────────────────────

DEFINITION = AlgorithmDefinition(
    id="depixelate",
    version="1.0.0",
    family=AlgorithmFamily.DITHERING,
    role=AlgorithmRole.RENDERER,
    name="Depixelate",
    description="Recover the native pixel grid of an upscaled image; re-render each cell as a 2-colour ordered-dither tile",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="final_raster",
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "block",
            "Tile size",
            ParameterType.INT,
            default=2,
            min=2,
            max=6,
            step=1,
            group="tiles",
            description="Each recovered cell becomes a block×block ordered-dither tile. 2 = the o/x checkerboard; larger gives more tonal levels.",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "palette",
            "Palette",
            ParameterType.STR,
            default="adaptive",
            options=[
                OptionDef(value="original", label="Original colors"),
                OptionDef(value="adaptive", label="Adaptive (from image)"),
                OptionDef(value="grayscale", label="Grayscale"),
                OptionDef(value="duotone", label="Duotone (ink → paper)"),
            ],
            group="palette",
            description="Original keeps source colours; the others quantise to a limited palette first.",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "colors",
            "Colors",
            ParameterType.INT,
            default=4,
            min=2,
            max=8,
            step=1,
            group="palette",
            description="Palette size when quantising (ignored for Original colors).",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "ink_color",
            "Ink Color (duotone)",
            ParameterType.STR,
            default="#161616",
            group="palette",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "paper_color",
            "Paper Color (duotone)",
            ParameterType.STR,
            default="#f4ebd9",
            group="palette",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "keep_marks",
            "Keep sparse marks",
            ParameterType.BOOL,
            default=False,
            group="tiles",
            description="Force a minority subpixel for any second colour, so faint dots/stars survive. Off keeps near-solid cells solid (no forced dark/light pair).",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "tau",
            "Mark threshold",
            ParameterType.INT,
            default=45,
            min=0,
            max=255,
            step=1,
            group="tiles",
            description="Colour distance from a cell's main colour that registers as a second colour. Lower keeps fainter dots.",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "pitch",
            "Grid pitch",
            ParameterType.INT,
            default=0,
            min=0,
            max=64,
            step=1,
            group="grid",
            description="Source pixels per cell. 0 auto-detects the upscale grid; set a value to override on multi-grid images. Leave the output size at original so detection sees the full-res grid.",
            invalidates=["final_raster"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="final_raster",
            type="raster_image",
            label="Depixelated Image",
            viewer=ArtifactViewerSpec(default_view="image"),
        ),
    ],
    calibration_assets=[],
    execution_profile=ExecutionProfile(
        typical_runtime="sub_second",
        is_iterative=False,
        is_streamable=False,
        is_cancellable=False,
        parallelism="serial",
        memory_class="small",
    ),
    capabilities=AlgorithmCapabilities(
        supports_raster_output=True,
        supports_vector_output=False,
        supports_multi_class=False,
        supports_interactive_preview=True,
        supports_progressive_refinement=False,
        deterministic=True,
        requires_gpu=False,
    ),
)


class DepixelateRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return

        block = int(ctx.params.get("block", 2))
        tau = int(ctx.params.get("tau", 45))
        pitch = int(ctx.params.get("pitch", 0))
        palette = str(ctx.params.get("palette", "adaptive"))
        colors = int(ctx.params.get("colors", 4))
        ink_color = str(ctx.params.get("ink_color", "#161616"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))
        keep_marks = bool(ctx.params.get("keep_marks", False))

        img = ctx.input.image.convert("RGB")
        native, _grid = depixelate(
            img,
            reducer="tiles",
            block=block,
            tau=tau,
            pitch=pitch if pitch > 0 else None,
            palette=palette,
            colors=colors,
            ink_color=ink_color,
            paper_color=paper_color,
            keep_marks=keep_marks,
        )
        ctx.store.publish("final_raster", native)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        art = ctx.store.get_by_name("final_raster")
        return RenderResult(
            algorithm_primary_artifact_id=art.id,
            default_composition=None,
        )


registry.register(DepixelateRenderer())


if __name__ == "__main__":
    import sys

    src = sys.argv[1]
    reducer = sys.argv[2] if len(sys.argv) > 2 else "tiles"
    block = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    pitch = float(sys.argv[4]) if len(sys.argv) > 4 else None
    img = Image.open(src).convert("RGB")
    native, grid = depixelate(img, reducer=reducer, pitch=pitch, block=block)
    print(f"source     : {img.size[0]}x{img.size[1]}")
    print(f"pitch      : {grid.pitch_x:.1f} x {grid.pitch_y:.1f} px/cell")
    print(f"phase      : ({grid.phase_x:.1f}, {grid.phase_y:.1f})")
    print(f"confidence : {grid.confidence:.2f}")
    print(f"native     : {native.size[0]}x{native.size[1]}")

    native.save("/tmp/depix_native.png")
    # nearest-neighbour re-upscale for easy eyeballing
    preview = native.resize((native.size[0] * 12, native.size[1] * 12), Image.Resampling.NEAREST)
    preview.save("/tmp/depix_preview.png")
    print("wrote /tmp/depix_native.png and /tmp/depix_preview.png")
