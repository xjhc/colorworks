from __future__ import annotations

import hashlib
from typing import Any
import numpy as np

from colorworks.algorithms import (
    IterativeAlgorithm,
    registry,
    RenderContext,
    calibration_registry,
)
from colorworks.algorithms.image_ops import colorize_binary_ink_mask, to_gray
from colorworks.domain import (
    AlgorithmDefinition,
    AlgorithmFamily,
    AlgorithmRole,
    InputSpec,
    OutputSpec,
    ParameterDef,
    ParameterType,
    ArtifactKindDef,
    ArtifactViewerSpec,
    ExecutionProfile,
    AlgorithmCapabilities,
    RenderResult,
    CalibrationAssetRef,
    IterationPreview,
    WarmStartState,
)





def _compute_default_hvs() -> tuple[np.ndarray, dict[str, Any], str]:
    support = 11
    radius = 5
    sigma = 1.5
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    r_hh = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    r_hh /= np.sum(r_hh)

    # Cast to little-endian 32-bit float for stable, cross-platform checksums
    data = np.asarray(r_hh, dtype="<f4")
    checksum = hashlib.sha256(data.tobytes()).hexdigest()

    # Define calibration asset metadata.
    # Note: We use kind="lut" as the closest match for the HVS autocorrelation matrix
    # because it represents a lookup structure of spatial weights.
    metadata = {
        "id": "hvs_model",
        "algorithm_id": "dbs",
        "algorithm_version": "1.0.0",
        "kind": "lut",
        "storage_uri": f"calibration_assets/{checksum}.npy",
        "checksum": checksum,
        "size_bytes": data.nbytes,
        "metadata": {
            "sigma": sigma,
            "support": support,
        },
        "version": "1.0.0",
    }
    return data, metadata, checksum


DEFAULT_HVS_DATA, DEFAULT_HVS_META, DEFAULT_HVS_CHECKSUM = _compute_default_hvs()

# Register the default calibration asset with the global registry at import time
calibration_registry.register(DEFAULT_HVS_CHECKSUM, DEFAULT_HVS_DATA, DEFAULT_HVS_META)


DEFINITION = AlgorithmDefinition(
    id="dbs",
    version="1.0.0",
    family=AlgorithmFamily.HALFTONING,
    role=AlgorithmRole.RENDERER,
    name="Direct Binary Search",
    description="Model-based Direct Binary Search iterative halftoning (CPU reference).",
    input_spec=InputSpec(primary="raster", accepts_color=True, max_resolution=64),
    output_spec=OutputSpec(
        primary_artifact="final_raster",
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "max_iterations",
            "Max Iterations",
            ParameterType.INT,
            default=5,
            min=1,
            max=20,
            step=1,
            group="optimization",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "ink_color",
            "Ink Color",
            ParameterType.STR,
            default="#1a1a1a",
            group="palette",
            ui_hint="color",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "paper_color",
            "Paper Color",
            ParameterType.STR,
            default="#f4ebd9",
            group="palette",
            ui_hint="color",
            invalidates=["final_raster"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="final_raster",
            type="raster_image",
            label="Final Halftone",
            viewer=ArtifactViewerSpec(default_view="image"),
        ),
    ],
    calibration_assets=[
        CalibrationAssetRef(
            asset_id="hvs_model",
            checksum=DEFAULT_HVS_CHECKSUM,
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
        supports_progressive_refinement=False,
        deterministic=True,
        requires_gpu=False,
    ),
)


def _convolve2d_zero(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """NumPy-only 2D zero-padded convolution for small image sizes."""
    kh, kw = kernel.shape
    kh2, kw2 = kh // 2, kw // 2
    padded = np.pad(image, ((kh2, kh2), (kw2, kw2)), mode="constant", constant_values=0.0)

    h, w = image.shape
    out = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight == 0.0:
                continue
            out += weight * padded[i : i + h, j : j + w]
    return out




class DBSRenderer(IterativeAlgorithm):
    definition = DEFINITION

    def initialize(self, ctx: RenderContext) -> None:
        # Load the HVS autocorrelation matrix using the checksum from definition
        checksum = self.definition.calibration_assets[0].checksum
        self._r_hh = ctx.calibration.get_data(checksum)

        # Validate that the autocorrelation kernel is an odd square matrix
        if self._r_hh.ndim != 2 or self._r_hh.shape[0] != self._r_hh.shape[1] or self._r_hh.shape[0] % 2 != 1:
            raise ValueError("Calibration HVS matrix must be an odd-sized 2D square matrix.")
        self._radius = self._r_hh.shape[0] // 2

        # Get grayscale input and convert to density f = 1.0 - gray
        gray = to_gray(ctx.input.image)
        self._f = 1.0 - gray
        H, W = self._f.shape

        # Validate size limit using InputSpec helper
        self.definition.input_spec.validate_image_size(self.definition.id, W, H)

        # Seed initial binary halftone matrix b in {0.0, 1.0} using ctx.rng for determinism
        self._b = (ctx.rng.random(self._f.shape) < self._f).astype(np.float32)

        # Calculate initial error a = b - f
        self._a = self._b - self._f

        # Calculate initial g = a * R_hh using _convolve2d_zero
        self._g = _convolve2d_zero(self._a, self._r_hh)

    def step(self, ctx: RenderContext, iteration: int) -> float:
        # Perform one scan of the image using localized updates for g
        H, W = self._b.shape
        r_hh = self._r_hh
        radius = self._radius
        r_hh_center = r_hh[radius, radius]
        accepted_count = 0

        for y in range(H):
            for x in range(W):
                best_change = None
                best_delta_E = 0.0

                # 1. Evaluate Toggle
                # a0 is the change in b[y, x] if we toggle it
                a0_toggle = 1.0 - 2.0 * self._b[y, x]
                delta_E_toggle = 2.0 * a0_toggle * self._g[y, x] + r_hh_center

                if delta_E_toggle < best_delta_E:
                    best_delta_E = delta_E_toggle
                    best_change = (delta_E_toggle, "toggle", a0_toggle, None)

                # 2. Evaluate Swaps with 8-neighbors
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if self._b[y, x] != self._b[ny, nx]:
                                a0_swap = self._b[ny, nx] - self._b[y, x]
                                offset_y = radius + ny - y
                                offset_x = radius + nx - x
                                delta_E_swap = (
                                    2.0 * a0_swap * (self._g[y, x] - self._g[ny, nx]) +
                                    2.0 * (r_hh_center - r_hh[offset_y, offset_x])
                                )
                                if delta_E_swap < best_delta_E:
                                    best_delta_E = delta_E_swap
                                    best_change = (delta_E_swap, "swap", a0_swap, (ny, nx))

                # If a change reduces energy, apply it immediately (online greedy update)
                if best_change is not None:
                    _, change_type, a0, target = best_change
                    accepted_count += 1

                    if change_type == "toggle":
                        self._b[y, x] = 1.0 - self._b[y, x]

                        # Update g in-place in local neighborhood (NumPy slicing for speed)
                        y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
                        x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
                        ky1, ky2 = radius + y_min - y, radius + y_max - y
                        kx1, kx2 = radius + x_min - x, radius + x_max - x
                        self._g[y_min:y_max, x_min:x_max] += a0 * r_hh[ky1:ky2, kx1:kx2]

                    elif change_type == "swap":
                        ny, nx = target
                        # Swap binary values
                        val_x0 = self._b[y, x]
                        val_x1 = self._b[ny, nx]
                        self._b[y, x] = val_x1
                        self._b[ny, nx] = val_x0

                        # Update g in-place for neighborhood of (y, x) (NumPy slicing for speed)
                        y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
                        x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
                        ky1, ky2 = radius + y_min - y, radius + y_max - y
                        kx1, kx2 = radius + x_min - x, radius + x_max - x
                        self._g[y_min:y_max, x_min:x_max] += a0 * r_hh[ky1:ky2, kx1:kx2]

                        # Update g in-place for neighborhood of (ny, nx) (NumPy slicing for speed)
                        ny_min, ny_max = max(0, ny - radius), min(H, ny + radius + 1)
                        nx_min, nx_max = max(0, nx - radius), min(W, nx + radius + 1)
                        nky1, nky2 = radius + ny_min - ny, radius + ny_max - ny
                        nkx1, nkx2 = radius + nx_min - nx, radius + nx_max - nx
                        self._g[ny_min:ny_max, nx_min:nx_max] -= a0 * r_hh[nky1:nky2, nkx1:nkx2]

        return float(accepted_count)

    def current_energy(self) -> float:
        # E = sum e * g
        return float(np.sum((self._b - self._f) * self._g))

    def should_stream_preview(self, it: int) -> bool:
        return True

    def build_iteration_preview(self, ctx: RenderContext, iteration: int) -> IterationPreview:
        ink_color = str(ctx.params.get("ink_color", "#1a1a1a"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))
        img = colorize_binary_ink_mask(self._b, ink_color, paper_color)
        return IterationPreview(mode="direct_raster", direct_raster=img)

    def finalize(self, ctx: RenderContext, *, partial: bool, warm_state: WarmStartState | None) -> RenderResult:
        ink_color = str(ctx.params.get("ink_color", "#1a1a1a"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))
        final_img = colorize_binary_ink_mask(self._b, ink_color, paper_color)
        final_id = ctx.store.publish("final_raster", final_img)

        return RenderResult(
            algorithm_primary_artifact_id=final_id,
            final_artifact_id=final_id,
            partial=partial,
            warm_state=warm_state,
        )

    def max_iterations(self, ctx: RenderContext) -> int:
        return int(ctx.params.get("max_iterations", 5))

    def convergence_threshold(self, ctx: RenderContext) -> float:
        # Zero accepted improvements = converged.
        return 0.5

    def can_warm_start(self, state: WarmStartState, new_params: dict[str, Any]) -> bool:
        return False


registry.register(DBSRenderer())
