from __future__ import annotations

from PIL import Image

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext
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

DEFINITION = AlgorithmDefinition(
    id="palette_quantize",
    version="1.1.0",
    family=AlgorithmFamily.DITHERING,
    role=AlgorithmRole.RENDERER,
    name="Palette Quantize",
    description="Flat N-color pixel-art style palette quantization",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="final_raster",
        produces_composition=False,
    ),
    parameters=[
        ParameterDef(
            "colors",
            "Colors",
            ParameterType.INT,
            default=4,
            min=2,
            max=8,
            step=1,
            group="palette",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "palette",
            "Palette",
            ParameterType.STR,
            default="adaptive",
            options=[
                OptionDef(value="adaptive", label="Adaptive"),
                OptionDef(value="grayscale", label="Grayscale"),
            ],
            group="palette",
            invalidates=["final_raster"],
        ),
        ParameterDef(
            "dither",
            "Dither",
            ParameterType.BOOL,
            default=True,
            group="palette",
            invalidates=["final_raster"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="final_raster",
            type="raster_image",
            label="Quantized Image",
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


class PaletteQuantizeRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return

        colors = int(ctx.params.get("colors", 4))
        palette_mode = str(ctx.params.get("palette", "adaptive"))
        dither = bool(ctx.params.get("dither", True))

        img = ctx.input.image

        base = img.convert("L").convert("RGB") if palette_mode == "grayscale" else img.convert("RGB")

        # First derive the N-colour adaptive palette with no dithering.
        pal_img = base.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

        if dither:
            # PIL ignores `dither=` when `method=` is passed, so re-map the image
            # onto the fixed palette with explicit Floyd–Steinberg error diffusion.
            quantized = base.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG)
        else:
            quantized = pal_img

        rgb_out = quantized.convert("RGB")
        ctx.store.publish("final_raster", rgb_out)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        art = ctx.store.get_by_name("final_raster")
        return RenderResult(
            algorithm_primary_artifact_id=art.id,
            default_composition=None,
        )


# Register the algorithm
registry.register(PaletteQuantizeRenderer())
