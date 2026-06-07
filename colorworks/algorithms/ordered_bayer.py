from __future__ import annotations

import numpy as np
from PIL import Image

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext
from colorworks.algorithms.image_ops import (
    to_gray,
    colorize_binary_ink_mask,
)
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
from colorworks.renderers.bayer import bayer_matrix

DEFINITION = AlgorithmDefinition(
    id="ordered_bayer",
    version="1.0.0",
    family=AlgorithmFamily.DITHERING,
    role=AlgorithmRole.RENDERER,
    name="Ordered (Bayer)",
    description="Crisp threshold grid ordered dither.",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="final_raster",
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "matrix_size",
            "Bayer Matrix Size",
            ParameterType.INT,
            default=8,
            options=[
                OptionDef(2, "2x2"),
                OptionDef(4, "4x4"),
                OptionDef(8, "8x8"),
                OptionDef(16, "16x16"),
            ],
            group="tone",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "threshold",
            "Threshold",
            ParameterType.FLOAT,
            default=0.0,
            min=-0.5,
            max=0.5,
            step=0.01,
            group="tone",
            invalidates=["final_raster"],
        ),
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
            "ink_color",
            "Ink Color",
            ParameterType.STR,
            default="#121212",
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
            label="Bayer Dither",
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


class OrderedBayerRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return

        matrix_size = int(ctx.params.get("matrix_size", 8))
        threshold = float(ctx.params.get("threshold", 0.0))
        contrast = float(ctx.params.get("contrast", 1.0))
        ink_color = str(ctx.params.get("ink_color", "#121212"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))

        gray = to_gray(ctx.input.image)
        adjusted = np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)
        density = np.clip(1.0 - adjusted + threshold, 0.0, 1.0)

        threshold_matrix = bayer_matrix(matrix_size)
        height, width = density.shape
        repeats_y = (height + matrix_size - 1) // matrix_size
        repeats_x = (width + matrix_size - 1) // matrix_size
        tiled_thresholds = np.tile(threshold_matrix, (repeats_y, repeats_x))[:height, :width]

        ink_mask = density >= tiled_thresholds
        
        img = colorize_binary_ink_mask(ink_mask, ink_color, paper_color)
        ctx.store.publish("final_raster", img)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        art = ctx.store.get_by_name("final_raster")
        return RenderResult(
            algorithm_primary_artifact_id=art.id,
            default_composition=None,
        )


# Register the algorithm
registry.register(OrderedBayerRenderer())
