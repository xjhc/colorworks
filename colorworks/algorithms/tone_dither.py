"""Multi-tone dither renderer — the primary path to the N-colour dithered look.

One algorithm, three knobs that matter: how many `colors`, which `palette`, and
which dither `method`. The method selector spans crisp ordered (Bayer), organic
blue-noise, error-diffused Floyd-Steinberg, and the "artistic" flow/wave mask —
for flowing, non-photographic texture.
"""
from __future__ import annotations

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext
from colorworks.algorithms.dither import render_tone_dither
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
    Eq,
)

DEFINITION = AlgorithmDefinition(
    id="tone_dither",
    version="1.2.0",
    family=AlgorithmFamily.DITHERING,
    role=AlgorithmRole.RENDERER,
    name="Dither (Multi-tone)",
    description="N-colour dithering: choose colours, palette, and dither texture "
    "(ordered, blue-noise, Floyd-Steinberg, or wave).",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(primary_artifact="final_raster", produces_composition=False),
    parameters=[
        ParameterDef(
            "colors", "Colors", ParameterType.INT,
            default=4, min=2, max=8, step=1,
            group="palette", invalidates=["final_raster"],
        ),
        ParameterDef(
            "palette", "Palette", ParameterType.STR,
            default="adaptive",
            options=[
                OptionDef(value="adaptive", label="Adaptive (from image)"),
                OptionDef(value="grayscale", label="Grayscale"),
                OptionDef(value="duotone", label="Duotone (ink → paper)"),
            ],
            group="palette", invalidates=["final_raster"],
        ),
        ParameterDef(
            "method", "Dither Method", ParameterType.STR,
            default="floyd_steinberg",
            options=[
                OptionDef(value="bayer", label="Ordered (Bayer)"),
                OptionDef(value="blue_noise", label="Blue Noise"),
                OptionDef(value="floyd_steinberg", label="Floyd–Steinberg"),
                OptionDef(value="flow", label="Flow (waves)"),
            ],
            group="pattern", invalidates=["final_raster"],
        ),
        ParameterDef(
            "matrix_size", "Bayer Matrix Size", ParameterType.INT,
            default=8,
            options=[OptionDef(2, "2×2"), OptionDef(4, "4×4"), OptionDef(8, "8×8"), OptionDef(16, "16×16")],
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "bayer"),
        ),
        ParameterDef(
            "noise_size", "Blue Noise Size", ParameterType.INT,
            default=64,
            options=[OptionDef(16, "16×16"), OptionDef(32, "32×32"), OptionDef(64, "64×64"), OptionDef(128, "128×128")],
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "blue_noise"),
        ),
        ParameterDef(
            "frequency", "Wave Density", ParameterType.FLOAT,
            default=5.0, min=1.0, max=24.0, step=0.5,
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "flow"),
        ),
        ParameterDef(
            "warp", "Flow Strength", ParameterType.FLOAT,
            default=7.0, min=0.0, max=20.0, step=0.5,
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "flow"),
        ),
        ParameterDef(
            "angle_deg", "Flow Angle (deg)", ParameterType.FLOAT,
            default=45.0, min=0.0, max=180.0, step=1.0,
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "flow"),
        ),
        ParameterDef(
            "detail", "Flow Detail", ParameterType.FLOAT,
            default=2.5, min=0.5, max=8.0, step=0.5,
            group="pattern", invalidates=["final_raster"],
            visible_when=Eq("method", "flow"),
        ),
        ParameterDef(
            "contrast", "Contrast", ParameterType.FLOAT,
            default=1.0, min=0.1, max=3.0, step=0.05,
            group="tone", invalidates=["final_raster"],
        ),
        ParameterDef(
            "midpoint", "Midpoint", ParameterType.FLOAT,
            default=0.5, min=0.0, max=1.0, step=0.01,
            group="tone", invalidates=["final_raster"],
        ),
        ParameterDef(
            "ink_color", "Ink Color (duotone)", ParameterType.STR,
            default="#161616", group="palette", ui_hint="color", invalidates=["final_raster"],
            visible_when=Eq("palette", "duotone"),
        ),
        ParameterDef(
            "paper_color", "Paper Color (duotone)", ParameterType.STR,
            default="#f4ebd9", group="palette", ui_hint="color", invalidates=["final_raster"],
            visible_when=Eq("palette", "duotone"),
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="final_raster",
            type="raster_image",
            label="Dithered Image",
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


class ToneDitherRenderer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["final_raster"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        if ctx.store.has("final_raster"):
            return
        p = ctx.params
        img = render_tone_dither(
            ctx.input.image,
            colors=int(p.get("colors", 4)),
            palette_mode=str(p.get("palette", "adaptive")),
            method=str(p.get("method", "floyd_steinberg")),
            contrast=float(p.get("contrast", 1.0)),
            midpoint=float(p.get("midpoint", 0.5)),
            ink_color=str(p.get("ink_color", "#161616")),
            paper_color=str(p.get("paper_color", "#f4ebd9")),
            params={
                "matrix_size": int(p.get("matrix_size", 8)),
                "noise_size": int(p.get("noise_size", 64)),
                "frequency": float(p.get("frequency", 5.0)),
                "warp": float(p.get("warp", 7.0)),
                "angle_deg": float(p.get("angle_deg", 45.0)),
                "detail": float(p.get("detail", 2.5)),
            },
            seed=ctx.seed,
        )
        ctx.store.publish("final_raster", img)

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        art = ctx.store.get_by_name("final_raster")
        return RenderResult(algorithm_primary_artifact_id=art.id, default_composition=None)


registry.register(ToneDitherRenderer())
