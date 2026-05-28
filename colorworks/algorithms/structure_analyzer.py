from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from colorworks.algorithms import StagedAlgorithm, registry
from colorworks.algorithms.image_ops import (
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
    ScalarField,
    StructureTensorField,
    VectorField2D,
    Composition,
    InkLayerSpec,
    PaletteColor,
    PatternSpec,
    PatternCoordinateSpec,
    RenderResult,
)

DEFINITION = AlgorithmDefinition(
    id="structure_analyzer",
    version="1.0.0",
    family=AlgorithmFamily.STRUCTURE_ANALYSIS,
    role=AlgorithmRole.ANALYZER,
    name="Structure Analyzer",
    description="Extract structure tensor and ETF orientation vector field.",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="orientation_field",
        optional_artifacts=["tone_map", "structure_tensor"],
        produces_composition=True,
    ),
    parameters=[
        ParameterDef(
            "contrast",
            "Tone contrast",
            ParameterType.FLOAT,
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            group="tone",
            invalidates=["tone_map"],
        ),
        ParameterDef(
            "midpoint",
            "Tone midpoint",
            ParameterType.FLOAT,
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            group="tone",
            invalidates=["tone_map"],
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
            invalidates=["structure_tensor", "orientation_field"],
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
            invalidates=["orientation_field"],
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
            invalidates=["orientation_field"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="orientation_field",
            type="vector_field_2d",
            label="Orientation Field",
            suitable_as=["orientation_source"],
            viewer=ArtifactViewerSpec(default_view="orientation_hsv"),
        ),
        ArtifactKindDef(
            name="tone_map",
            type="scalar_field",
            label="Tone Map",
            suitable_as=["density_source"],
            viewer=ArtifactViewerSpec(default_view="heatmap", colormap="gray"),
        ),
        ArtifactKindDef(
            name="structure_tensor",
            type="structure_tensor_field",
            label="Structure Tensor",
            suitable_as=["structure_tensor"],
            viewer=ArtifactViewerSpec(default_view="heatmap"),
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
        supports_raster_output=False,
        supports_vector_output=True,
        supports_multi_class=False,
        supports_interactive_preview=True,
        supports_progressive_refinement=False,
        deterministic=True,
        requires_gpu=False,
    ),
)


class StructureAnalyzer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["tone_map", "structure_tensor", "orientation_field"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        gray = None

        # 1. Tone Map
        if not ctx.store.has("tone_map"):
            gray = to_gray(ctx.input.image)
            tone = remap_tone(
                gray,
                ctx.params.get("contrast", 1.0),
                ctx.params.get("midpoint", 0.5)
            )
            tone_id = ctx.store.publish(
                "tone_map",
                ScalarField(ctx.input.substrate, tone, "float32"),
            )
            ctx.working.put("tone_id", tone_id)
        else:
            tone_id = ctx.store.get_by_name("tone_map").id
            ctx.working.put("tone_id", tone_id)

        # 2. Structure Tensor
        if not ctx.store.has("structure_tensor"):
            if gray is None:
                gray = to_gray(ctx.input.image)

            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
            Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

            Gx = convolve2d_nearest(gray, Kx)
            Gy = convolve2d_nearest(gray, Ky)

            Jxx = Gx * Gx
            Jxy = Gx * Gy
            Jyy = Gy * Gy

            sigma = ctx.params.get("sigma", 3.0)
            Jxx_smooth = gaussian_blur(Jxx, sigma)
            Jxy_smooth = gaussian_blur(Jxy, sigma)
            Jyy_smooth = gaussian_blur(Jyy, sigma)

            tensor_data = np.stack([Jxx_smooth, Jxy_smooth, Jyy_smooth], axis=-1)
            tensor_id = ctx.store.publish(
                "structure_tensor",
                StructureTensorField(ctx.input.substrate, tensor_data),
            )
            ctx.working.put("tensor_id", tensor_id)
        else:
            tensor_id = ctx.store.get_by_name("structure_tensor").id
            ctx.working.put("tensor_id", tensor_id)

        # 3. Orientation Field
        if not ctx.store.has("orientation_field"):
            tensor_art = ctx.store.get_by_name("structure_tensor")
            tensor_data = tensor_art.value.data
            Jxx = tensor_data[:, :, 0]
            Jxy = tensor_data[:, :, 1]
            Jyy = tensor_data[:, :, 2]

            theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0
            tx = np.cos(theta)
            ty = np.sin(theta)
            t = np.stack([tx, ty], axis=-1)

            iterations = int(ctx.params.get("etf_iterations", 3))
            radius = int(ctx.params.get("etf_radius", 5))
            if iterations > 0:
                t = etf_smooth(t, Jxx, Jyy, iterations, radius)

            orientation_id = ctx.store.publish(
                "orientation_field",
                VectorField2D(ctx.input.substrate, t, is_bidirectional=True),
            )
            ctx.working.put("orientation_id", orientation_id)
        else:
            orientation_id = ctx.store.get_by_name("orientation_field").id
            ctx.working.put("orientation_id", orientation_id)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        default = Composition(
            paper_color=PaletteColor("#f4ebd9", "paper"),
            layers=[
                InkLayerSpec(
                    name="hatch_layer",
                    color=PaletteColor("#1a1a1a", "ink"),
                    role="shadow",
                    density_source="tone_map",
                    pattern=PatternSpec(
                        kind="hatch",
                        params={
                            "frequency": 8.0,
                            "angle_deg": 45.0,
                            "phase": 0.0,
                        },
                        orientation_source="orientation_field",
                        coordinates=PatternCoordinateSpec(
                            space="image_px",
                            seed=ctx.seed,
                        ),
                    ),
                )
            ],
        )
        return RenderResult(
            algorithm_primary_artifact_id=ctx.working.get("orientation_id"),
            default_composition=default,
        )

    def is_artifact_enabled(self, name: str, params: dict[str, Any]) -> bool:
        return True

    def load_from_cache(self, ctx: RenderContext, artifacts: dict[str, str]) -> None:
        if "tone_map" in artifacts:
            ctx.working.put("tone_id", artifacts["tone_map"])
        if "structure_tensor" in artifacts:
            ctx.working.put("tensor_id", artifacts["structure_tensor"])
        if "orientation_field" in artifacts:
            ctx.working.put("orientation_id", artifacts["orientation_field"])

# Register StructureAnalyzer
registry.register(StructureAnalyzer())
