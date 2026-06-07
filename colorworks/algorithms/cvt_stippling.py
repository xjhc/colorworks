"""
CVT Stippling via Lloyd relaxation.

Phase 3 iterative algorithm (IterativeAlgorithm, AlgorithmRole.RENDERER).

Each step is one Lloyd iteration: for every pixel find its nearest stipple
point (Voronoi cell), then move each point to the density-weighted centroid
of its cell.  Convergence is declared when the maximum point displacement
falls below convergence_threshold pixels.

Warm-start contract:
  - can_warm_start: True when n_stipples is unchanged.
  - export / import: serialise point array via payload["points"].
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from colorworks.algorithms import IterativeAlgorithm, registry, RenderContext
from colorworks.domain import (
    AlgorithmDefinition,
    AlgorithmFamily,
    AlgorithmRole,
    ArtifactKindDef,
    ArtifactViewerSpec,
    ExecutionProfile,
    AlgorithmCapabilities,
    InputSpec,
    OutputSpec,
    IterationPreview,
    ParameterDef,
    ParameterType,
    PointSet,
    RasterGrid,
    RenderResult,
    WarmStartState,
)

# Max internal working resolution (resize larger images for speed)
_MAX_DIM = 256


DEFINITION = AlgorithmDefinition(
    id="cvt_stippling",
    version="1.1.0",
    family=AlgorithmFamily.STIPPLING,
    role=AlgorithmRole.RENDERER,
    name="CVT Stippling",
    description=(
        "Centroidal Voronoi Tessellation stippling via Lloyd relaxation. "
        "Produces a density-aware point cloud rendered as circular dots."
    ),
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="stipple_points",
        optional_artifacts=["final_raster"],
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "n_stipples", "Number of stipples", ParameterType.INT,
            default=2500, min=200, max=16000, step=100,
            group="stippling",
            invalidates=["stipple_points", "final_raster"],
        ),
        ParameterDef(
            "dot_radius", "Dot radius (px)", ParameterType.FLOAT,
            default=1.1, min=0.5, max=10.0, step=0.1,
            group="stippling",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "max_iterations", "Max iterations", ParameterType.INT,
            default=20, min=1, max=100,
            group="stippling",
            invalidates=["stipple_points", "final_raster"],
        ),
        ParameterDef(
            "convergence_threshold", "Convergence (px)", ParameterType.FLOAT,
            default=0.5, min=0.01, max=5.0, step=0.05,
            group="stippling",
            invalidates=["stipple_points", "final_raster"],
        ),
        ParameterDef(
            "ink_color", "Ink colour", ParameterType.STR,
            default="#1a1a1a", ui_hint="color",
            group="appearance",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "paper_color", "Paper colour", ParameterType.STR,
            default="#f4ebd9", ui_hint="color",
            group="appearance",
            invalidates=["final_raster"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="stipple_points",
            type="point_set",
            label="Stipple Points",
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


def _parse_hex(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _to_density(image: Image.Image) -> np.ndarray:
    """Return [H, W] float32 in [0,1] where 1.0 = full ink (dark pixel)."""
    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return 1.0 - arr   # invert: dark → high density


def _resize_density(density: np.ndarray, max_dim: int) -> tuple[np.ndarray, float, float]:
    H, W = density.shape
    if H <= max_dim and W <= max_dim:
        return density, 1.0, 1.0
    scale = max_dim / max(H, W)
    nH, nW = int(H * scale), int(W * scale)
    img = Image.fromarray((density * 255).astype(np.uint8))
    img = img.resize((nW, nH), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0, nW / W, nH / H


def _lloyd_step(
    points: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, float]:
    """One Lloyd iteration on the working-resolution density.

    points : (N, 2) in working-resolution pixel coords (x, y)
    Returns (new_points, max_displacement)
    """
    H, W = density.shape
    N = len(points)

    yy, xx = np.mgrid[0:H, 0:W]          # (H, W)
    px = points[:, 0]                      # (N,)
    py = points[:, 1]

    # Pairwise distances  (H, W, N)
    # For N=300, H=W=256: ≈ 60 M float32 = 240 MB — borderline.
    # Chunk over N to keep peak memory ~60 MB.
    chunk = 64
    nearest = np.empty((H, W), dtype=np.int32)
    best_d2 = np.full((H, W), np.inf, dtype=np.float64)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        dx = xx[:, :, np.newaxis] - px[np.newaxis, np.newaxis, start:end]
        dy = yy[:, :, np.newaxis] - py[np.newaxis, np.newaxis, start:end]
        d2 = dx**2 + dy**2           # (H, W, chunk)
        local_best = np.argmin(d2, axis=-1)   # (H, W)
        local_min  = d2[np.arange(H)[:, None], np.arange(W)[None, :], local_best]
        update = local_min < best_d2
        nearest[update] = (start + local_best)[update]
        best_d2[update] = local_min[update]

    new_points = np.zeros_like(points)
    for i in range(N):
        mask = nearest == i
        w = density[mask]
        total_w = w.sum()
        if total_w > 1e-10:
            ys_m, xs_m = np.where(mask)
            new_points[i, 0] = np.dot(xs_m, w) / total_w
            new_points[i, 1] = np.dot(ys_m, w) / total_w
        else:
            new_points[i] = points[i]

    disp = np.hypot(new_points[:, 0] - points[:, 0],
                    new_points[:, 1] - points[:, 1])
    return new_points, float(disp.max())


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
    for (x, y) in points:
        x0, y0 = x - radius, y - radius
        x1, y1 = x + radius, y + radius
        draw.ellipse([(x0, y0), (x1, y1)], fill=ink_rgb)
    return img


class CVTStippling(IterativeAlgorithm):
    definition = DEFINITION

    # ── state ─────────────────────────────────────────────────────────────────
    _points: np.ndarray
    _density_full: np.ndarray      # original resolution
    _density_work: np.ndarray      # working resolution
    _sx: float                     # scale from working → full (x)
    _sy: float                     # scale from working → full (y)
    _energy: float

    # ── IterativeAlgorithm hooks ──────────────────────────────────────────────

    def initialize(self, ctx: RenderContext) -> None:
        n = int(ctx.params["n_stipples"])
        density_full = _to_density(ctx.input.image)
        density_work, sx, sy = _resize_density(density_full, _MAX_DIM)

        H, W = density_work.shape
        flat = density_work.flatten()
        total = flat.sum()
        probs = flat / total if total > 1e-10 else np.ones(H * W, dtype=np.float32) / (H * W)

        indices = ctx.rng.choice(H * W, size=n, replace=(n > H * W), p=probs)
        ys = (indices // W).astype(np.float32)
        xs = (indices % W).astype(np.float32)

        self._points = np.stack([xs, ys], axis=1)     # working-res coords
        self._density_full = density_full
        self._density_work = density_work
        self._sx = sx
        self._sy = sy
        self._energy = float("inf")

    def step(self, ctx: RenderContext, iteration: int) -> float:
        new_pts, delta = _lloyd_step(self._points, self._density_work)
        self._points = new_pts
        self._energy = delta
        return delta

    def current_energy(self) -> float:
        return self._energy

    def should_stream_preview(self, it: int) -> bool:
        return it % 2 == 0   # every other iteration starting at 0

    def build_iteration_preview(self, ctx: RenderContext, iteration: int) -> IterationPreview:
        img = self._render_current(ctx)
        return IterationPreview(mode="direct_raster", direct_raster=img)

    def max_iterations(self, ctx: RenderContext) -> int:
        return int(ctx.params.get("max_iterations", 20))

    def convergence_threshold(self, ctx: RenderContext) -> float:
        return float(ctx.params.get("convergence_threshold", 0.5))

    def finalize(
        self,
        ctx: RenderContext,
        *,
        partial: bool,
        warm_state: WarmStartState | None,
    ) -> RenderResult:
        W_full = ctx.input.substrate.width
        H_full = ctx.input.substrate.height

        # Scale points back to full resolution
        full_pts = self._points.copy()
        full_pts[:, 0] /= self._sx
        full_pts[:, 1] /= self._sy
        full_pts = np.clip(full_pts, [0, 0], [W_full - 1, H_full - 1])

        substrate = ctx.input.substrate
        ps = PointSet(substrate=substrate, coords=full_pts)
        pt_id = ctx.store.publish("stipple_points", ps)

        final_img = self._render_current(ctx, full_res=True)
        final_id = ctx.store.publish("final_raster", final_img)

        return RenderResult(
            algorithm_primary_artifact_id=pt_id,
            final_artifact_id=final_id,
            partial=partial,
            warm_state=warm_state,
        )

    # ── warm-start ────────────────────────────────────────────────────────────

    def can_warm_start(self, state: WarmStartState, new_params: dict) -> bool:
        if state.algorithm_id != self.definition.id:
            return False
        return int(state.params.get("n_stipples", 0)) == int(new_params.get("n_stipples", 0))

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
        density_full = _to_density(ctx.input.image)
        density_work, sx, sy = _resize_density(density_full, _MAX_DIM)
        self._points = np.array(state.payload["points"], dtype=np.float32)
        self._density_full = density_full
        self._density_work = density_work
        self._sx = sx
        self._sy = sy
        self._energy = state.energy if state.energy is not None else float("inf")

    # ── private ───────────────────────────────────────────────────────────────

    def _render_current(self, ctx: RenderContext, *, full_res: bool = False) -> Image.Image:
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
            # Points were relaxed at a downsampled working resolution; grow the
            # dots by the same factor so coverage holds at full output size
            # (otherwise large exports look sparse and blank).
            upscale = 1.0 / max(min(self._sx, self._sy), 1e-6)
            if upscale > 1.0:
                radius *= upscale
        else:
            H, W = self._density_work.shape
            pts = np.clip(self._points, [0, 0], [W - 1, H - 1])
            # Scale radius to preview resolution
            radius = radius * min(self._sx, self._sy)

        return _rasterize(pts, radius, ink_rgb, paper_rgb, W, H)


registry.register(CVTStippling())
