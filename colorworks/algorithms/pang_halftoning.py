"""
Pang-style structure-aware halftoning — Phase 3.5.

Self-contained renderer (HALFTONING family, RENDERER role).  Internally
computes a tone map and structure-tensor orientation field from the input
image, reusing Phase 2 helpers (to_gray, gaussian_blur, convolve2d_nearest).

orientation_source parameter:
  "internal" or "orientation_field" (default) — use the internally computed
  field.  Any other non-empty string is reserved for future cross-run artifact
  borrowing and raises ValueError with a user-safe message.

Energy:  E = w_tone * E_tone + w_orient * E_orient
  E_tone   = ||dot_density_map - tone_map||_F^2
  E_orient = sum over nearby pairs of sin^2(angle(disp_ij, orient_at_midpoint))
             (lower = dots more aligned with orientation field)

Annealing: Metropolis with geometric temperature schedule.
Warm-start: serialise / restore point positions; rebuild density map.
Deterministic: all randomness via ctx.rng (seeded from ctx.seed).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colorworks.algorithms import IterativeAlgorithm, RenderContext, registry
from colorworks.domain import (
    AlgorithmCapabilities,
    AlgorithmDefinition,
    AlgorithmFamily,
    AlgorithmRole,
    ArtifactKindDef,
    ArtifactViewerSpec,
    ExecutionProfile,
    InputSpec,
    IterationPreview,
    OutputSpec,
    ParameterDef,
    ParameterType,
    PointSet,
    RenderResult,
    WarmStartState,
)

_MAX_DIM = 256
# Accepted values for orientation_source — all map to the internal field.
# Other strings are rejected (reserved for future cross-run borrowing).
_INTERNAL_SOURCES = frozenset({"internal", "orientation_field", ""})

DEFINITION = AlgorithmDefinition(
    id="pang_halftoning",
    version="1.0.0",
    family=AlgorithmFamily.HALFTONING,
    role=AlgorithmRole.RENDERER,
    name="Pang Structure-Aware Halftoning",
    description=(
        "Structure-aware halftoning via Metropolis annealing. "
        "Dot density matches the tone map; dot arrangement follows the local "
        "orientation field derived from the structure tensor."
    ),
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="halftone_points",
        optional_artifacts=["final_raster"],
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "n_dots", "Number of dots", ParameterType.INT,
            default=500, min=10, max=2000,
            group="halftoning",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "dot_radius", "Dot radius (px)", ParameterType.FLOAT,
            default=2.0, min=0.5, max=10.0, step=0.5,
            group="halftoning",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "max_iterations", "Max iterations", ParameterType.INT,
            default=50, min=1, max=200,
            group="annealing",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "temperature_start", "Temperature start", ParameterType.FLOAT,
            default=1.0, min=0.01, max=10.0, step=0.01,
            group="annealing",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "temperature_end", "Temperature end", ParameterType.FLOAT,
            default=0.01, min=0.001, max=1.0, step=0.001,
            group="annealing",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "w_tone", "Tone weight", ParameterType.FLOAT,
            default=1.0, min=0.0, max=10.0, step=0.1,
            group="energy",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "w_orient", "Orientation weight", ParameterType.FLOAT,
            default=0.5, min=0.0, max=10.0, step=0.1,
            group="energy",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "ssim_window", "Neighbourhood radius (px)", ParameterType.INT,
            default=7, min=3, max=21, step=2,
            group="energy",
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "orientation_source", "Orientation source", ParameterType.STR,
            default="internal",
            group="structure",
            description=(
                "Use 'internal' (default) to compute the orientation field directly "
                "from the input image.  Other values are reserved for a future phase "
                "that borrows orientation artifacts from prior analyzer runs; they "
                "will raise ValueError until that phase is implemented."
            ),
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "convergence_threshold", "Convergence threshold", ParameterType.FLOAT,
            default=0.0, min=0.0, max=10.0, step=0.001,
            group="annealing",
            description=(
                "Stop early when |ΔE| drops below this value.  Default 0 runs all "
                "max_iterations."
            ),
            invalidates=["halftone_points", "final_raster"],
        ),
        ParameterDef(
            "ink_color", "Ink colour", ParameterType.STR,
            default="#1a1a1a",
            group="appearance",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "paper_color", "Paper colour", ParameterType.STR,
            default="#f4ebd9",
            group="appearance",
            invalidates=["final_raster"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="halftone_points",
            type="point_set",
            label="Halftone Points",
            suitable_as=["density_source"],
            viewer=ArtifactViewerSpec(default_view="points"),
        ),
        ArtifactKindDef(
            name="final_raster",
            type="raster_image",
            label="Final Raster",
            suitable_as=["final"],
            viewer=ArtifactViewerSpec(default_view="raster"),
        ),
    ],
    execution_profile=ExecutionProfile(
        typical_runtime="seconds",
        is_iterative=True,
        is_streamable=True,
        is_cancellable=True,
        parallelism="serial",
        memory_class="small",
    ),
    capabilities=AlgorithmCapabilities(
        supports_raster_output=True,
        supports_vector_output=False,
        supports_multi_class=False,
        supports_interactive_preview=True,
        supports_progressive_refinement=True,
        deterministic=True,
        requires_gpu=False,
    ),
)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_hex(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _compute_fields(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Compute tone map and orientation field from image.

    Returns:
      tone_map:  [H, W] float32 in [0,1], 1 = dark pixel (place dots here)
      orient:    [H, W, 2] float32 unit vectors, edge tangent direction
    """
    # Reuse Phase 2 helpers — no duplication.
    from colorworks.algorithms.image_ops import (
        convolve2d_nearest,
        gaussian_blur,
        to_gray,
    )

    gray = to_gray(image)           # [H, W] float32
    tone = (1.0 - gray).astype(np.float32)   # dark → 1.0

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    Gx = convolve2d_nearest(gray, Kx)
    Gy = convolve2d_nearest(gray, Ky)

    sigma = 2.0
    Jxx = gaussian_blur(Gx * Gx, sigma)
    Jxy = gaussian_blur(Gx * Gy, sigma)
    Jyy = gaussian_blur(Gy * Gy, sigma)

    # Edge tangent (perpendicular to gradient)
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0
    orient = np.stack(
        [np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)],
        axis=-1,
    )
    return tone, orient


def _resize_2d(arr: np.ndarray, max_dim: int) -> tuple[np.ndarray, float, float]:
    """Resize a [H, W] float32 array to fit within max_dim."""
    H, W = arr.shape
    if H <= max_dim and W <= max_dim:
        return arr, 1.0, 1.0
    scale = max_dim / max(H, W)
    nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img_r = img.resize((nW, nH), Image.Resampling.BILINEAR)
    return np.asarray(img_r, dtype=np.float32) / 255.0, nW / W, nH / H


def _resize_orient(orient: np.ndarray, nH: int, nW: int) -> np.ndarray:
    """Resize a [H, W, 2] orientation field to (nH, nW), re-normalising."""
    H, W = orient.shape[:2]
    if H == nH and W == nW:
        return orient
    channels = []
    for c in range(2):
        ch8 = np.clip((orient[:, :, c] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(ch8, mode="L")
        img_r = img.resize((nW, nH), Image.Resampling.BILINEAR)
        channels.append(np.asarray(img_r, dtype=np.float32) / 127.5 - 1.0)
    out = np.stack(channels, axis=-1)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(norms > 1e-6, out / np.maximum(norms, 1e-6), out)


def _rasterize(
    points: np.ndarray,
    radius: float,
    ink_rgb: tuple[int, int, int],
    paper_rgb: tuple[int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    img = Image.new("RGB", (width, height), paper_rgb)
    draw = ImageDraw.Draw(img)
    for x, y in points:
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=ink_rgb)
    return img


# ── Algorithm class ───────────────────────────────────────────────────────────

class PangHalftoning(IterativeAlgorithm):
    """Pang-style structure-aware halftoning via Metropolis annealing."""

    definition = DEFINITION

    # ── Per-run state (set in initialize / import_warm_state) ─────────────────
    _points: np.ndarray       # (N, 2) float32  [x, y] in working-res coords
    _dot_density: np.ndarray  # (H, W) float32  running Gaussian splat density
    _tone_work: np.ndarray    # (H, W) float32  tone map at working res
    _orient_work: np.ndarray  # (H, W, 2) float32  orientation field at working res
    _density_full: np.ndarray # full-res tone map (for finalize scaling reference)
    _sx: float                # working_width / full_width
    _sy: float                # working_height / full_height
    _energy: float
    # cached parameters (set in _load_params)
    _gauss_sigma: float
    _gauss_radius: int
    _orient_radius: float
    _w_tone: float
    _w_orient: float
    _T_start: float
    _T_end: float
    _max_iters: int

    # ── IterativeAlgorithm hooks ──────────────────────────────────────────────

    def initialize(self, ctx: RenderContext) -> None:
        self._load_params(ctx)

        tone_full, orient_full = _compute_fields(ctx.input.image)
        tone_work, sx, sy = _resize_2d(tone_full, _MAX_DIM)
        nH, nW = tone_work.shape
        orient_work = _resize_orient(orient_full, nH, nW)

        self._tone_work = tone_work
        self._orient_work = orient_work
        self._density_full = tone_full
        self._sx = sx
        self._sy = sy

        H, W = tone_work.shape
        n = int(ctx.params.get("n_dots", 500))
        flat = tone_work.flatten().astype(np.float64)
        total = flat.sum()
        probs = flat / total if total > 1e-10 else np.ones(H * W) / (H * W)
        probs /= probs.sum()

        indices = ctx.rng.choice(H * W, size=n, replace=(n > H * W), p=probs)
        jitter = ctx.rng.standard_normal((n, 2)).astype(np.float32) * 0.4
        xs = np.clip((indices % W).astype(np.float32) + jitter[:, 0], 0.0, W - 1.0)
        ys = np.clip((indices // W).astype(np.float32) + jitter[:, 1], 0.0, H - 1.0)

        self._points = np.stack([xs, ys], axis=1)
        self._dot_density = self._build_density_map(H, W)
        self._energy = self._compute_total_energy()

    def step(self, ctx: RenderContext, iteration: int) -> float:
        H, W = self._tone_work.shape
        n = len(self._points)
        T = self._temperature(iteration)
        old_energy = self._energy

        # Proposal sigma: proportional to sqrt(T), at least 0.5 px
        prop_sigma = max(0.5, 3.0 * (T / max(self._T_start, 1e-10)) ** 0.5)

        # Pre-draw all randoms for the sweep (determinism: ctx.rng is seeded)
        order = ctx.rng.permutation(n)
        disps = ctx.rng.standard_normal((n, 2)).astype(np.float32) * prop_sigma
        mc_randoms = ctx.rng.random(n)

        for step_i, idx in enumerate(order):
            old_pos = self._points[idx].copy()
            new_pos = np.array(
                [
                    float(np.clip(old_pos[0] + disps[step_i, 0], 0.0, W - 1.0)),
                    float(np.clip(old_pos[1] + disps[step_i, 1], 0.0, H - 1.0)),
                ],
                dtype=np.float32,
            )

            delta_tone, patch_info = self._delta_tone(old_pos, new_pos)
            delta_orient = self._delta_orient(idx, old_pos, new_pos)
            delta_E = self._w_tone * delta_tone + self._w_orient * delta_orient

            accept = delta_E <= 0.0
            if not accept and T > 1e-12:
                accept = mc_randoms[step_i] < np.exp(-delta_E / T)

            if accept:
                self._points[idx] = new_pos
                ys_sl, xs_sl, g_old, g_new = patch_info
                self._dot_density[ys_sl, xs_sl] -= g_old
                self._dot_density[ys_sl, xs_sl] += g_new
                np.clip(self._dot_density, 0.0, None, out=self._dot_density)

        self._energy = self._compute_total_energy()
        return abs(old_energy - self._energy)

    def current_energy(self) -> float:
        return self._energy

    def should_stream_preview(self, it: int) -> bool:
        return it % 3 == 0

    def build_iteration_preview(
        self, ctx: RenderContext, iteration: int
    ) -> IterationPreview:
        return IterationPreview(
            mode="direct_raster",
            direct_raster=self._render_current(ctx),
        )

    def max_iterations(self, ctx: RenderContext) -> int:
        return self._max_iters

    def convergence_threshold(self, ctx: RenderContext) -> float:
        # Use explicit param if provided; default 0 = run all max_iterations.
        return float(ctx.params.get("convergence_threshold", 0.0))

    def finalize(
        self,
        ctx: RenderContext,
        *,
        partial: bool,
        warm_state: WarmStartState | None,
    ) -> RenderResult:
        W_full = ctx.input.substrate.width
        H_full = ctx.input.substrate.height

        full_pts = self._points.copy()
        full_pts[:, 0] /= self._sx
        full_pts[:, 1] /= self._sy
        full_pts = np.clip(full_pts, [0, 0], [W_full - 1, H_full - 1])

        ps = PointSet(substrate=ctx.input.substrate, coords=full_pts)
        pt_id = ctx.store.publish("halftone_points", ps)

        final_img = self._render_current(ctx, full_res=True)
        final_id = ctx.store.publish("final_raster", final_img)

        return RenderResult(
            algorithm_primary_artifact_id=pt_id,
            final_artifact_id=final_id,
            partial=partial,
            warm_state=warm_state,
        )

    # ── Warm-start ────────────────────────────────────────────────────────────

    def can_warm_start(self, state: WarmStartState, new_params: dict) -> bool:
        if state.algorithm_id != self.definition.id:
            return False
        n_ok = int(state.params.get("n_dots", 0)) == int(new_params.get("n_dots", 0))
        w_ok = int(state.params.get("ssim_window", 7)) == int(
            new_params.get("ssim_window", 7)
        )
        return n_ok and w_ok

    def export_warm_state(self, ctx: RenderContext) -> WarmStartState:
        return WarmStartState(
            algorithm_id=self.definition.id,
            algorithm_version=self.definition.version,
            iteration=0,
            energy=self._energy,
            params=dict(ctx.params),
            payload={"points": self._points.tolist()},
        )

    def import_warm_state(self, ctx: RenderContext, state: WarmStartState) -> None:
        self._load_params(ctx)

        tone_full, orient_full = _compute_fields(ctx.input.image)
        tone_work, sx, sy = _resize_2d(tone_full, _MAX_DIM)
        nH, nW = tone_work.shape
        orient_work = _resize_orient(orient_full, nH, nW)

        self._tone_work = tone_work
        self._orient_work = orient_work
        self._density_full = tone_full
        self._sx = sx
        self._sy = sy

        self._points = np.array(state.payload["points"], dtype=np.float32)
        H, W = tone_work.shape
        self._dot_density = self._build_density_map(H, W)
        # Recompute energy from restored state (avoids floating-point drift).
        self._energy = self._compute_total_energy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_params(self, ctx: RenderContext) -> None:
        src = str(ctx.params.get("orientation_source", "internal")).strip()
        if src not in _INTERNAL_SOURCES:
            raise ValueError(
                f"orientation_source '{src}' is not supported. "
                "Use 'internal' to compute the orientation field from the input "
                "image.  Cross-run artifact borrowing is reserved for a future phase."
            )
        self._ssim_window = int(ctx.params.get("ssim_window", 7))
        self._gauss_sigma = max(0.5, self._ssim_window / 3.0)
        self._gauss_radius = int(np.ceil(3.0 * self._gauss_sigma))
        self._orient_radius = float(self._ssim_window * 2)
        self._w_tone = float(ctx.params.get("w_tone", 1.0))
        self._w_orient = float(ctx.params.get("w_orient", 0.5))
        self._T_start = float(ctx.params.get("temperature_start", 1.0))
        self._T_end = float(ctx.params.get("temperature_end", 0.01))
        self._max_iters = int(ctx.params.get("max_iterations", 50))

    def _temperature(self, it: int) -> float:
        if self._max_iters <= 1:
            return self._T_end
        frac = it / max(1, self._max_iters - 1)
        ratio = self._T_end / max(self._T_start, 1e-10)
        return float(self._T_start * (ratio ** frac))

    def _build_density_map(self, H: int, W: int) -> np.ndarray:
        d = np.zeros((H, W), dtype=np.float32)
        r = self._gauss_radius
        s2 = 2.0 * self._gauss_sigma ** 2
        for x, y in self._points:
            yi, xi = int(round(y)), int(round(x))
            y1, y2 = max(0, yi - r), min(H, yi + r + 1)
            x1, x2 = max(0, xi - r), min(W, xi + r + 1)
            yy = np.arange(y1, y2, dtype=np.float32) - y
            xx = np.arange(x1, x2, dtype=np.float32) - x
            d[y1:y2, x1:x2] += np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / s2)
        return d

    def _delta_tone(
        self, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> tuple[float, tuple]:
        """Compute delta E_tone when one dot moves; return (delta, patch_info)."""
        H, W = self._tone_work.shape
        r = self._gauss_radius
        s2 = 2.0 * self._gauss_sigma ** 2

        oyi, oxi = int(round(old_pos[1])), int(round(old_pos[0]))
        nyi, nxi = int(round(new_pos[1])), int(round(new_pos[0]))
        y1 = max(0, min(oyi, nyi) - r)
        y2 = min(H, max(oyi, nyi) + r + 1)
        x1 = max(0, min(oxi, nxi) - r)
        x2 = min(W, max(oxi, nxi) + r + 1)
        ys = slice(y1, y2)
        xs = slice(x1, x2)

        d_patch = self._dot_density[ys, xs].copy()
        t_patch = self._tone_work[ys, xs]

        yy_o = np.arange(y1, y2, dtype=np.float32) - old_pos[1]
        xx_o = np.arange(x1, x2, dtype=np.float32) - old_pos[0]
        g_old = np.exp(-(yy_o[:, None] ** 2 + xx_o[None, :] ** 2) / s2)

        yy_n = np.arange(y1, y2, dtype=np.float32) - new_pos[1]
        xx_n = np.arange(x1, x2, dtype=np.float32) - new_pos[0]
        g_new = np.exp(-(yy_n[:, None] ** 2 + xx_n[None, :] ** 2) / s2)

        e_before = float(np.sum((d_patch - t_patch) ** 2))
        e_after = float(np.sum((d_patch - g_old + g_new - t_patch) ** 2))
        return e_after - e_before, (ys, xs, g_old, g_new)

    def _delta_orient(
        self, idx: int, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> float:
        """Compute delta E_orient when dot idx moves from old_pos to new_pos."""
        if self._w_orient < 1e-10:
            return 0.0
        pts = self._points
        N = len(pts)
        H, W = self._orient_work.shape[:2]
        r = self._orient_radius

        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        other = pts[mask]
        if len(other) == 0:
            return 0.0

        d_old = np.hypot(other[:, 0] - old_pos[0], other[:, 1] - old_pos[1])
        d_new = np.hypot(other[:, 0] - new_pos[0], other[:, 1] - new_pos[1])
        affected = (d_old < r) | (d_new < r)
        if not np.any(affected):
            return 0.0

        nbrs = other[affected]

        def _pair(center: np.ndarray, nbrs_: np.ndarray) -> float:
            disps = nbrs_ - center
            dists = np.hypot(disps[:, 0], disps[:, 1])
            valid = dists > 1e-6
            if not np.any(valid):
                return 0.0
            ud = disps[valid] / dists[valid, None]
            mids = (center + nbrs_[valid]) / 2.0
            mxi = np.clip(mids[:, 0].astype(np.int32), 0, W - 1)
            myi = np.clip(mids[:, 1].astype(np.int32), 0, H - 1)
            ov = self._orient_work[myi, mxi]
            on = np.linalg.norm(ov, axis=-1)
            vo = on > 1e-6
            if not np.any(vo):
                return 0.0
            o = ov[vo] / on[vo, None]
            dot = np.einsum("ki,ki->k", ud[vo], o)
            return float(np.sum(1.0 - dot ** 2))

        return _pair(new_pos, nbrs) - _pair(old_pos, nbrs)

    def _compute_total_energy(self) -> float:
        e_tone = float(np.sum((self._dot_density - self._tone_work) ** 2))
        e_orient = self._compute_orient_energy_full()
        return self._w_tone * e_tone + self._w_orient * e_orient

    def _compute_orient_energy_full(self) -> float:
        """Sum sin^2(angle) over all pairs within orient_radius (pairs counted once)."""
        if self._w_orient < 1e-10:
            return 0.0
        N = len(self._points)
        if N < 2:
            return 0.0
        H, W = self._orient_work.shape[:2]
        r = self._orient_radius
        total = 0.0
        for i in range(N - 1):
            pos_i = self._points[i]
            pts_j = self._points[i + 1:]
            disps = pts_j - pos_i
            dists = np.hypot(disps[:, 0], disps[:, 1])
            close = (dists < r) & (dists > 1e-6)
            if not np.any(close):
                continue
            ud = disps[close] / dists[close, None]
            mids = (pos_i + pts_j[close]) / 2.0
            mxi = np.clip(mids[:, 0].astype(np.int32), 0, W - 1)
            myi = np.clip(mids[:, 1].astype(np.int32), 0, H - 1)
            ov = self._orient_work[myi, mxi]
            on = np.linalg.norm(ov, axis=-1)
            vo = on > 1e-6
            if np.any(vo):
                o = ov[vo] / on[vo, None]
                dot = np.einsum("ki,ki->k", ud[vo], o)
                total += float(np.sum(1.0 - dot ** 2))
        return total

    def _render_current(
        self, ctx: RenderContext, *, full_res: bool = False
    ) -> Image.Image:
        ink_rgb = _parse_hex(str(ctx.params.get("ink_color", "#1a1a1a")))
        paper_rgb = _parse_hex(str(ctx.params.get("paper_color", "#f4ebd9")))
        radius = float(ctx.params.get("dot_radius", 2.0))

        if full_res:
            W = ctx.input.substrate.width
            H = ctx.input.substrate.height
            pts = self._points.copy()
            pts[:, 0] /= self._sx
            pts[:, 1] /= self._sy
            pts = np.clip(pts, [0, 0], [W - 1, H - 1])
        else:
            H, W = self._tone_work.shape
            pts = np.clip(self._points, [0, 0], [W - 1, H - 1])
            radius = radius * min(self._sx, self._sy)

        return _rasterize(pts, radius, ink_rgb, paper_rgb, W, H)


registry.register(PangHalftoning())
