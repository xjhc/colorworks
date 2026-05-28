from __future__ import annotations

import hashlib
import re
from typing import Any
import numpy as np
from PIL import Image, ImageOps

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext, calibration_registry
from colorworks.algorithms.structure_analyzer import (
    to_gray,
    remap_tone,
    convolve2d_nearest,
    gaussian_blur,
    etf_smooth,
)
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
)

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}$|^#[0-9a-fA-F]{6}$")


def validate_color(hex_str: str) -> None:
    if not HEX_COLOR_RE.match(hex_str):
        raise ValueError(f"Invalid hex color format: {hex_str}. Must be #RGB or #RRGGBB.")


def parse_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def _compute_saed_lut() -> tuple[np.ndarray, dict[str, Any], str]:
    n_angles = 180
    support = 11
    radius = 5
    sigma_g = 2.0
    frequency = 1.0

    lut = np.zeros((n_angles, support, support), dtype=np.float32)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    for i in range(n_angles):
        theta = float(i) * np.pi / float(n_angles)

        # Rotated coordinates:
        # x_rot is along the normal (theta + pi/2), y_rot is along the flow (theta)
        x_rot = -x * np.sin(theta) + y * np.cos(theta)
        y_rot = -x * np.cos(theta) - y * np.sin(theta)

        # Gabor filter equation: exp(-(x'^2 + y'^2) / (2 * sigma_g^2)) * cos(frequency * x')
        g = np.exp(-(x_rot**2 + y_rot**2) / (2.0 * sigma_g**2)) * np.cos(frequency * x_rot)
        g -= np.mean(g)
        lut[i] = g

    # Cast to little-endian 32-bit float for stable, cross-platform checksums
    data = np.asarray(lut, dtype="<f4")
    checksum = hashlib.sha256(data.tobytes()).hexdigest()

    metadata = {
        "id": "saed_gabor_lut",
        "algorithm_id": "saed",
        "algorithm_version": "1.0.0",
        "kind": "lut",
        "storage_uri": f"calibration_assets/{checksum}.npy",
        "checksum": checksum,
        "size_bytes": data.nbytes,
        "metadata": {
            "n_angles": n_angles,
            "kernel_size": support,
            "sigma_g": sigma_g,
            "frequency": frequency,
        },
        "version": "1.0.0",
    }
    return data, metadata, checksum


DEFAULT_SAED_DATA, DEFAULT_SAED_META, DEFAULT_SAED_CHECKSUM = _compute_saed_lut()
calibration_registry.register(DEFAULT_SAED_CHECKSUM, DEFAULT_SAED_DATA, DEFAULT_SAED_META)


DEFINITION = AlgorithmDefinition(
    id="saed",
    version="1.0.0",
    family=AlgorithmFamily.HALFTONING,
    role=AlgorithmRole.RENDERER,
    name="Structure-Aware Error Diffusion",
    description="CPU-only structure-aware error diffusion dither using orientation-guided Gabor thresholds and anisotropic coefficients.",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="final_raster",
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "contrast",
            "Contrast",
            ParameterType.FLOAT,
            default=1.0,
            min=0.1,
            max=3.0,
            step=0.05,
            group="tone",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "midpoint",
            "Midpoint",
            ParameterType.FLOAT,
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            group="tone",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "sigma",
            "Tensor Blur (sigma)",
            ParameterType.FLOAT,
            default=3.0,
            min=0.5,
            max=10.0,
            step=0.1,
            group="structure",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "etf_iterations",
            "ETF Iterations",
            ParameterType.INT,
            default=3,
            min=0,
            max=10,
            step=1,
            group="etf",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "etf_radius",
            "ETF Radius",
            ParameterType.INT,
            default=5,
            min=1,
            max=15,
            step=1,
            group="etf",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "gabor_amplitude",
            "Gabor Amplitude",
            ParameterType.FLOAT,
            default=0.2,
            min=0.0,
            max=1.0,
            step=0.05,
            group="saed",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "anisotropy_alpha",
            "Anisotropy Alpha",
            ParameterType.FLOAT,
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            group="saed",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "edge_scaling",
            "Edge Scaling",
            ParameterType.FLOAT,
            default=5.0,
            min=0.1,
            max=20.0,
            step=0.5,
            group="saed",
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
            asset_id="saed_gabor_lut",
            checksum=DEFAULT_SAED_CHECKSUM,
        ),
    ],
    execution_profile=ExecutionProfile(
        typical_runtime="seconds",
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


class SAEDRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return

        # 1. Load parameters
        contrast = float(ctx.params.get("contrast", 1.0))
        midpoint = float(ctx.params.get("midpoint", 0.5))
        sigma = float(ctx.params.get("sigma", 3.0))
        etf_iterations = int(ctx.params.get("etf_iterations", 3))
        etf_radius = int(ctx.params.get("etf_radius", 5))
        gabor_amplitude = float(ctx.params.get("gabor_amplitude", 0.2))
        anisotropy_alpha = float(ctx.params.get("anisotropy_alpha", 0.5))
        edge_scaling = float(ctx.params.get("edge_scaling", 5.0))
        ink_color = str(ctx.params.get("ink_color", "#1a1a1a"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))

        validate_color(ink_color)
        validate_color(paper_color)

        H = ctx.input.image.height
        W = ctx.input.image.width
        if H > 256 or W > 256:
            raise ValueError("SAED input dimensions exceed the 256x256 pixel limit for the CPU-only reference renderer.")

        # 2. Load calibration asset
        checksum = self.definition.calibration_assets[0].checksum
        if ctx.calibration is not None:
            lut = ctx.calibration.get_data(checksum)
        else:
            lut = DEFAULT_SAED_DATA

        # Validate that the Gabor LUT is a 3D array with an odd square kernel shape
        if lut.ndim != 3 or lut.shape[0] == 0 or lut.shape[1] != lut.shape[2] or lut.shape[1] % 2 != 1:
            raise ValueError(
                "Calibration SAED Gabor LUT must be a 3D array with shape "
                "(angles, kernel_size, kernel_size) where kernel_size is an odd square."
            )

        n_angles = lut.shape[0]
        support = lut.shape[1]
        radius = support // 2

        # 3. Grayscale conversion & Tone mapping (Density space: 1.0 = ink, 0.0 = paper)
        gray = to_gray(ctx.input.image)
        adjusted_gray = remap_tone(gray, contrast, midpoint)
        f = 1.0 - adjusted_gray

        # 4. Structure Tensor & Orientation Field (Phase 2 helpers)
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

        Gx = convolve2d_nearest(gray, Kx)
        Gy = convolve2d_nearest(gray, Ky)

        Jxx = Gx * Gx
        Jxy = Gx * Gy
        Jyy = Gy * Gy

        Jxx_smooth = gaussian_blur(Jxx, sigma)
        Jxy_smooth = gaussian_blur(Jxy, sigma)
        Jyy_smooth = gaussian_blur(Jyy, sigma)

        theta = 0.5 * np.arctan2(2.0 * Jxy_smooth, Jxx_smooth - Jyy_smooth) + np.pi / 2.0
        tx = np.cos(theta)
        ty = np.sin(theta)
        t = np.stack([tx, ty], axis=-1)

        if etf_iterations > 0:
            t = etf_smooth(t, Jxx_smooth, Jyy_smooth, etf_iterations, etf_radius)

        # 5. Spatially-Varying Gabor Threshold
        tx_vals = t[:, :, 0]
        ty_vals = t[:, :, 1]
        angles = np.arctan2(ty_vals, tx_vals) % np.pi
        indices = np.round(angles / np.pi * n_angles).astype(np.int32) % n_angles

        f_padded = np.pad(f, radius, mode="edge")
        I_G = np.zeros_like(f)
        for y in range(H):
            for x in range(W):
                kernel = lut[indices[y, x]]
                window = f_padded[y : y + support, x : x + support]
                I_G[y, x] = np.sum(window * kernel)

        T = np.clip(0.5 + gabor_amplitude * I_G, 0.01, 0.99)

        # 6. Anisotropic Error Diffusion
        neighbors = [(0, 1), (1, -1), (1, 0), (1, 1)]
        w_FS = np.array([7/16, 3/16, 5/16, 1/16], dtype=np.float32)

        v_neighbors = np.array([
            [1.0, 0.0],
            [-1.0/np.sqrt(2), 1.0/np.sqrt(2)],
            [0.0, 1.0],
            [1.0/np.sqrt(2), 1.0/np.sqrt(2)]
        ], dtype=np.float32)

        mag = np.sqrt(np.maximum(Jxx_smooth + Jyy_smooth, 0.0))
        alpha = anisotropy_alpha * np.minimum(1.0, mag * edge_scaling)

        # Project unit neighbor vectors onto the local ETF vector field
        dots = np.abs(np.tensordot(t, v_neighbors, axes=([2], [1]))) # (H, W, 4)
        dots_sum = np.sum(dots, axis=-1, keepdims=True)
        dots_sum = np.where(dots_sum < 1e-6, 1.0, dots_sum)
        A = dots / dots_sum

        arr = f.copy()
        out = np.zeros((H, W), dtype=bool)

        for y in range(H):
            for x in range(W):
                old_val = arr[y, x]
                thresh = T[y, x]
                new_val = 1.0 if old_val >= thresh else 0.0
                out[y, x] = (new_val == 1.0)
                err = old_val - new_val

                a_val = alpha[y, x]
                if dots_sum[y, x, 0] < 1e-6:
                    weights = w_FS
                else:
                    weights = (1.0 - a_val) * w_FS + a_val * A[y, x]

                for k, (dy, dx) in enumerate(neighbors):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        arr[ny, nx] += err * weights[k]

        # 7. Colorization and Publishing (1.0 = ink_color, 0.0 = paper_color)
        ink_rgb = parse_color(ink_color)
        paper_rgb = parse_color(paper_color)
        canvas = np.empty((H, W, 3), dtype=np.uint8)
        canvas[out] = ink_rgb
        canvas[~out] = paper_rgb

        img = Image.fromarray(canvas)
        ctx.store.publish("final_raster", img)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        art = ctx.store.get_by_name("final_raster")
        return RenderResult(
            algorithm_primary_artifact_id=art.id,
            final_artifact_id=art.id,
            default_composition=None,
        )


registry.register(SAEDRenderer())
