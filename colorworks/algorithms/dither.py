"""Multi-tone dithering primitives.

The shipped 1-bit renderers (Floyd-Steinberg, Ordered/Bayer, SAED, DBS) only
ever emit two colours: ink and paper. This module provides the building blocks
for *N-tone* dithering — quantising an image to an ordered palette of `colors`
levels while dithering across the bands so smooth gradients break into texture
instead of flat posterised steps.

Everything here is pure (no I/O, no global state) so it is cheap to unit test
and reuse. The threshold-mask generators are the single source of truth shared
with :mod:`colorworks.compositor`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from colorworks.algorithms.image_ops import parse_color, to_gray, remap_tone, gaussian_blur
from colorworks.renderers.bayer import bayer_matrix

PaletteMode = Literal["grayscale", "adaptive", "duotone"]
DitherMethod = Literal["bayer", "blue_noise", "floyd_steinberg", "flow", "maze", "wave"]

RGB = tuple[int, int, int]


# ── Palettes ──────────────────────────────────────────────────────────────────
def _lerp_rgb(a: RGB, b: RGB, t: float) -> RGB:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def _luma(c: RGB) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def kmeans_palette(image: Image.Image, colors: int, seed: int = 0, iters: int = 16) -> list[RGB]:
    """Extract `colors` perceptually distinct swatches via k-means (k-means++ init).

    Unlike median-cut (which splits by *population* and happily spends two slots on
    near-identical dominant colours), k-means balances the cluster centres — so
    "4 colours" yields four genuinely distinct, image-faithful colours. Runs on a
    small downsample for speed; deterministic given `seed`.
    """
    colors = max(2, int(colors))
    rgb = image.convert("RGB")
    # downsample for speed — palette doesn't need full resolution
    w, h = rgb.size
    if max(w, h) > 96:
        s = 96.0 / max(w, h)
        rgb = rgb.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    X = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    if len(X) == 0:
        return build_tone_palette(image, colors, "grayscale")

    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centres = [X[rng.integers(len(X))]]
    for _ in range(colors - 1):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centres], axis=0)
        total = d2.sum()
        probs = d2 / total if total > 1e-9 else np.full(len(X), 1.0 / len(X))
        centres.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centres, dtype=np.float32)

    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
        for j in range(colors):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
            else:  # re-seed a dead cluster on the worst-fit pixel
                far = np.argmax(np.min(((X[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1))
                C[j] = X[far]

    swatches = [tuple(int(round(v)) for v in c) for c in C]
    swatches.sort(key=_luma)
    return swatches[:colors]


def build_tone_palette(
    image: Image.Image,
    colors: int,
    mode: PaletteMode = "grayscale",
    ink_color: str = "#161616",
    paper_color: str = "#f4ebd9",
    seed: int = 0,
) -> list[RGB]:
    """Return `colors` RGB tuples ordered dark→light (index 0 is darkest).

    - grayscale: evenly spaced neutral ramp from black to white.
    - duotone:   evenly spaced ramp from ink_color to paper_color.
    - adaptive:  distinct representative colours extracted from the image via
                 k-means, sorted by luminance.
    """
    colors = max(2, int(colors))

    if mode == "grayscale":
        return [_lerp_rgb((0, 0, 0), (255, 255, 255), i / (colors - 1)) for i in range(colors)]

    if mode == "duotone":
        ink = parse_color(ink_color)
        paper = parse_color(paper_color)
        return [_lerp_rgb(ink, paper, i / (colors - 1)) for i in range(colors)]

    return kmeans_palette(image, colors, seed=seed)


def dither_to_palette(rgb01: np.ndarray, palette: list[RGB], mask: np.ndarray | None) -> Image.Image:
    """Dither an image to `palette` **in colour space** (true N-colour rendering).

    For each pixel we find its two nearest palette colours and choose between them
    using the per-pixel threshold `mask` and how far the pixel lies toward the
    second colour. With `mask=None` (or for flat fills) it snaps to the nearest
    colour. This is what makes "4 colours" mean four *distinct, chromatic* colours
    with dithered transitions — not a 1-D luminance ramp.
    """
    pal = np.asarray(palette, dtype=np.float32) / 255.0
    h, w, _ = rgb01.shape
    # squared distance to each palette colour → (h, w, N)
    d2 = ((rgb01[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(axis=3)
    order = np.argsort(d2, axis=2)
    c0 = order[:, :, 0]
    c1 = order[:, :, 1]
    p0 = pal[c0]
    p1 = pal[c1]
    if mask is None:
        out = p0
    else:
        direction = p1 - p0
        denom = (direction * direction).sum(axis=2) + 1e-6
        t = np.clip(((rgb01 - p0) * direction).sum(axis=2) / denom, 0.0, 1.0)
        out = np.where((t > mask)[..., None], p1, p0)
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _hilbert_xy(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Hilbert curve: return (x, y) visiting order for a 2^order grid."""
    n = 1 << order
    total = n * n
    t = np.arange(total, dtype=np.int64)
    x = np.zeros(total, dtype=np.int64)
    y = np.zeros(total, dtype=np.int64)
    s = 1
    while s < n:
        rx = 1 & (t >> 1)
        ry = 1 & (t ^ rx)
        # rot(s): where ry == 0, optionally reflect (rx == 1) then swap x/y
        m0 = ry == 0
        m01 = m0 & (rx == 1)
        x[m01] = s - 1 - x[m01]
        y[m01] = s - 1 - y[m01]
        tmp = x[m0].copy()
        x[m0] = y[m0]
        y[m0] = tmp
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def hilbert_dither_to_palette(rgb01: np.ndarray, palette: list[RGB], history: int = 16) -> Image.Image:
    """Riemersma dithering — error diffusion along a Hilbert space-filling curve.

    Because the traversal snakes through the image as a maze rather than scanning
    left-to-right, the quantisation error spreads isotropically and the texture
    becomes a connected labyrinth (diagonals + orthogonals) instead of Floyd–
    Steinberg's horizontal grain. This is the "maze" look.
    """
    h, w, _ = rgb01.shape
    if h == 0 or w == 0:
        return Image.fromarray((rgb01 * 255).astype(np.uint8), mode="RGB")
    order = max(1, int(np.ceil(np.log2(max(w, h)))))
    hx, hy = _hilbert_xy(order)
    inb = (hx < w) & (hy < h)
    hx = hx[inb]
    hy = hy[inb]

    pal = np.asarray(palette, dtype=np.float64) / 255.0
    ratio = (1.0 / 16.0) ** (1.0 / history)
    weights = ratio ** np.arange(history)
    weights /= weights.sum()
    weights = weights[:, None]

    src = rgb01.astype(np.float64)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    errq = np.zeros((history, 3), dtype=np.float64)
    xs = hx.tolist()
    ys = hy.tolist()
    for x, y in zip(xs, ys):
        val = src[y, x] + (weights * errq).sum(axis=0)
        j = int(((pal - val) ** 2).sum(axis=1).argmin())
        chosen = pal[j]
        errq[1:] = errq[:-1]
        errq[0] = val - chosen
        out[y, x] = (np.clip(chosen, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def fs_to_palette(image: Image.Image, palette: list[RGB]) -> Image.Image:
    """Floyd–Steinberg error diffusion onto an arbitrary `palette`, in colour space."""
    pal_img = Image.new("P", (1, 1))
    flat: list[int] = []
    for c in palette:
        flat.extend(int(v) for v in c)
    # fill all 256 slots with palette colours (repeat) so FS can't pick a stray entry
    if flat:
        while len(flat) < 768:
            flat.extend(flat[: min(len(flat), 768 - len(flat))])
    flat = (flat + [0] * 768)[:768]
    pal_img.putpalette(flat)
    return image.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")


def palette_to_array(palette: list[RGB]) -> np.ndarray:
    return np.asarray(palette, dtype=np.uint8)


# ── Threshold masks (shared with the compositor) ───────────────────────────────
def bayer_threshold_map(width: int, height: int, matrix_size: int = 8) -> np.ndarray:
    """Tiled Bayer ordered-dither thresholds in [0, 1), shape (height, width)."""
    if matrix_size not in (2, 4, 8, 16):
        matrix_size = 8
    matrix = bayer_matrix(matrix_size)
    ry = (height + matrix_size - 1) // matrix_size
    rx = (width + matrix_size - 1) // matrix_size
    return np.tile(matrix, (ry, rx))[:height, :width]


def blue_noise_threshold_map(width: int, height: int, size: int = 64, seed: int = 0) -> np.ndarray:
    """Void-and-cluster-style blue-noise thresholds in [0, 1), tiled to size."""
    if size not in (16, 32, 64, 128):
        size = 64
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    f = np.fft.fftshift(np.fft.fft2(noise))
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[-cy : size - cy, -cx : size - cx]
    r2 = x * x + y * y
    sigma = 1.5
    f *= 1.0 - np.exp(-r2 / (2.0 * sigma**2))
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
    ranks = np.argsort(np.argsort(filtered.ravel()))
    matrix = (ranks.reshape(size, size) + 0.5) / float(size * size)
    ry = (height + size - 1) // size
    rx = (width + size - 1) // size
    return np.tile(matrix, (ry, rx))[:height, :width]


def maze_threshold_map(
    width: int, height: int, scale: float = 16.0, line_width: float = 2.0, seed: int = 0
) -> np.ndarray:
    """Truchet-arc 'maze' thresholds in [0, 1] — the labyrinth texture."""
    if scale <= 1.0:
        scale = 16.0
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    CX = (X / scale).astype(np.int32)
    CY = (Y / scale).astype(np.int32)
    u = X % scale
    v = Y % scale
    hashes = (CX * 15991 + CY * 27763 + seed * 99187) % 2
    d1_0 = np.sqrt(u**2 + v**2)
    d2_0 = np.sqrt((u - scale) ** 2 + (v - scale) ** 2)
    dist_0 = np.minimum(np.abs(d1_0 - scale / 2.0), np.abs(d2_0 - scale / 2.0))
    d1_1 = np.sqrt((u - scale) ** 2 + v**2)
    d2_1 = np.sqrt(u**2 + (v - scale) ** 2)
    dist_1 = np.minimum(np.abs(d1_1 - scale / 2.0), np.abs(d2_1 - scale / 2.0))
    dist = np.where(hashes == 0, dist_0, dist_1)
    return np.clip((dist - line_width / 2.0) / (scale / 2.0) + 0.5, 0.0, 1.0)


def wave_threshold_map(
    width: int, height: int, frequency: float = 8.0, angle_deg: float = 45.0, phase: float = 0.0
) -> np.ndarray:
    """Sinusoidal 'wave' thresholds in [0, 1] for flowing directional texture."""
    theta = np.radians(angle_deg)
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    dist = X * np.cos(theta) + Y * np.sin(theta)
    val = np.sin(2.0 * np.pi * (dist * (frequency / 100.0) + phase))
    return (val + 1.0) / 2.0


def flow_threshold_map(
    gray: np.ndarray,
    frequency: float = 6.0,
    warp: float = 5.0,
    angle_deg: float = 45.0,
    detail: float = 2.5,
) -> np.ndarray:
    """Structure-aware 'flow' thresholds — the artistic-waves texture.

    A directional sine carrier whose phase is *domain-warped by the image's own
    (blurred) luminance*. Where the subject is bright vs. dark, the wave crests
    are displaced, so the dither bands bend and flow around features (face, hair,
    folds) instead of sitting as a flat overlay. Returns 0..1, shape `gray.shape`.

    - frequency: carrier wave density (cycles per ~100px) — the base stripe pitch.
    - warp:      how strongly image luminance bends the waves (0 = straight waves,
                 high = bands wrap tightly around contours).
    - angle_deg: base flow direction of the carrier.
    - detail:    blur radius of the warp field — small = fine local flow, large =
                 broad sweeping flow.
    """
    h, w = gray.shape
    g = gaussian_blur(gray.astype(np.float32), max(0.5, float(detail)))
    y, x = np.mgrid[0:h, 0:w]
    theta = np.radians(angle_deg)
    carrier = (x * np.cos(theta) + y * np.sin(theta)).astype(np.float32) * (float(frequency) / 100.0)
    phase = carrier + float(warp) * g
    return (np.sin(2.0 * np.pi * phase) + 1.0) / 2.0


def truchet_diagonal_field(width: int, height: int, scale: float = 5.0, seed: int = 0) -> np.ndarray:
    """Diagonal-Truchet 'maze' threshold field in [0, 1].

    Each `scale`×`scale` cell carries a random diagonal (``/`` or ``\\``). Adjacent
    diagonals meet at cell corners and link into a connected diagonal labyrinth.
    The value is 0 along the diagonal lines and rises to 1 at the cell interior,
    so an ordered dither against it reveals thin connected lines in the midtones
    that thicken and fill as the tone darkens — the maze look (diagonals, not the
    horizontal grain of Floyd-Steinberg).
    """
    if scale < 2.0:
        scale = 2.0
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    cx = (X // scale).astype(np.int64)
    cy = (Y // scale).astype(np.int64)
    u = X % scale
    v = Y % scale
    hsh = (cx * 73856093) ^ (cy * 19349663) ^ (np.int64(seed) * 83492791 + 1)
    slash = (hsh & 1) == 0
    d_back = np.abs(u - v) / np.sqrt(2.0)          # "\"
    d_fwd = np.abs(u + v - scale) / np.sqrt(2.0)   # "/"
    dist = np.where(slash, d_back, d_fwd)
    return np.clip(dist / (scale / 2.0), 0.0, 1.0)


def threshold_map(method: DitherMethod, width: int, height: int, params: dict, seed: int = 0) -> np.ndarray:
    if method == "bayer":
        return bayer_threshold_map(width, height, int(params.get("matrix_size", 8)))
    if method == "blue_noise":
        return blue_noise_threshold_map(width, height, int(params.get("noise_size", 64)), seed)
    if method == "maze":
        return maze_threshold_map(
            width, height, float(params.get("mask_scale", 12.0)), float(params.get("line_width", 2.0)), seed
        )
    if method == "wave":
        return wave_threshold_map(
            width, height, float(params.get("frequency", 8.0)), float(params.get("angle_deg", 45.0))
        )
    raise ValueError(f"no threshold map for method {method!r}")


# ── Quantisation across levels ─────────────────────────────────────────────────
def ordered_dither_levels(lightness: np.ndarray, n_levels: int, mask: np.ndarray) -> np.ndarray:
    """Quantise `lightness` (0..1) to n_levels using a per-pixel threshold mask.

    Returns an int array of level indices in [0, n_levels-1].
    """
    scaled = lightness * (n_levels - 1)
    low = np.floor(scaled).astype(np.int32)
    frac = scaled - low
    level = low + (frac > mask).astype(np.int32)
    return np.clip(level, 0, n_levels - 1)


def error_diffuse_levels(lightness: np.ndarray, n_levels: int) -> np.ndarray:
    """Floyd-Steinberg error diffusion quantising to n_levels. Returns indices."""
    h, w = lightness.shape
    arr = lightness.astype(np.float64).copy()
    out = np.zeros((h, w), dtype=np.int32)
    max_idx = n_levels - 1
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            idx = int(round(old * max_idx))
            idx = 0 if idx < 0 else (max_idx if idx > max_idx else idx)
            out[y, x] = idx
            err = old - idx / max_idx
            if x + 1 < w:
                arr[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * (3.0 / 16.0)
                arr[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * (1.0 / 16.0)
    return out


def levels_to_image(levels: np.ndarray, palette: list[RGB]) -> Image.Image:
    pal = palette_to_array(palette)
    return Image.fromarray(pal[np.clip(levels, 0, len(palette) - 1)], mode="RGB")


# ── Top-level render ───────────────────────────────────────────────────────────
def render_tone_dither(
    image: Image.Image,
    colors: int = 4,
    palette_mode: PaletteMode = "grayscale",
    method: DitherMethod = "bayer",
    *,
    contrast: float = 1.0,
    midpoint: float = 0.5,
    ink_color: str = "#161616",
    paper_color: str = "#f4ebd9",
    params: dict | None = None,
    seed: int = 0,
) -> Image.Image:
    """Render an image as an N-colour dither — the heart of the target aesthetic.

    Dithering happens **in colour space** against an N-colour palette, so the
    output uses exactly N distinct, chromatically-faithful colours (adaptive =
    k-means) with dithered transitions — not a flattened luminance ramp.
    """
    params = params or {}
    colors = max(2, min(8, int(colors)))
    if method not in ("bayer", "blue_noise", "floyd_steinberg", "flow", "maze", "wave"):
        method = "bayer"
    if palette_mode not in ("grayscale", "adaptive", "duotone"):
        palette_mode = "grayscale"

    palette = build_tone_palette(image, colors, palette_mode, ink_color, paper_color, seed=seed)

    # Hue-preserving tone control: rescale each pixel's luminance by the remap
    # curve, keeping chroma — so contrast/midpoint shift which palette colour a
    # region lands on without desaturating it.
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = to_gray(image)
    if abs(contrast - 1.0) > 1e-6 or abs(midpoint - 0.5) > 1e-6:
        remapped = remap_tone(gray, contrast, midpoint)
        scale = (remapped / np.maximum(gray, 1e-4))[:, :, None]
        rgb = np.clip(rgb * scale, 0.0, 1.0)
    h, w = gray.shape

    if method == "floyd_steinberg":
        toned = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
        return fs_to_palette(toned, palette)

    if method == "maze":
        mask = truchet_diagonal_field(w, h, float(params.get("mask_scale", 5.0)), seed)
    elif method == "flow":
        mask = flow_threshold_map(
            gray,
            frequency=float(params.get("frequency", 6.0)),
            warp=float(params.get("warp", 5.0)),
            angle_deg=float(params.get("angle_deg", 45.0)),
            detail=float(params.get("detail", 2.5)),
        )
    else:
        mask = threshold_map(method, w, h, params, seed)

    return dither_to_palette(rgb, palette, mask)
