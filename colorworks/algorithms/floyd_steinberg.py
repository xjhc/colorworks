from __future__ import annotations

import numpy as np

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
    ArtifactKindDef,
    ArtifactViewerSpec,
    ExecutionProfile,
    AlgorithmCapabilities,
    RenderResult,
)

DEFINITION = AlgorithmDefinition(
    id="floyd_steinberg",
    version="1.0.0",
    family=AlgorithmFamily.HALFTONING,
    role=AlgorithmRole.RENDERER,
    name="Floyd-Steinberg",
    description="Direct Floyd-Steinberg error diffusion dither. Bypasses the Compositor.",
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
            label="Final Dither",
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


class FloydSteinbergRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return

        contrast = float(ctx.params.get("contrast", 1.0))
        midpoint = float(ctx.params.get("midpoint", 0.5))
        ink_color = str(ctx.params.get("ink_color", "#1a1a1a"))
        paper_color = str(ctx.params.get("paper_color", "#f4ebd9"))

        gray = to_gray(ctx.input.image)
        adjusted = np.clip((gray - midpoint) * contrast + 0.5, 0.0, 1.0)

        h, w = adjusted.shape
        # standard 2D Floyd-Steinberg scanline error diffusion dither
        out = np.zeros((h, w), dtype=bool)
        arr = adjusted.copy()

        for y in range(h):
            for x in range(w):
                old_val = arr[y, x]
                new_val = 1.0 if old_val >= 0.5 else 0.0
                out[y, x] = (new_val == 1.0)
                err = old_val - new_val

                # Distribute error to 4 neighbors:
                if x + 1 < w:
                    arr[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x > 0:
                        arr[y + 1, x - 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (1.0 / 16.0)

        # Map binary output to ink/paper colors using shared helper.
        # Note Floyd-Steinberg's inverted out meaning: out is True when new_val is 1.0 (paper),
        # so ~out represents ink.
        img = colorize_binary_ink_mask(~out, ink_color, paper_color)
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
registry.register(FloydSteinbergRenderer())
