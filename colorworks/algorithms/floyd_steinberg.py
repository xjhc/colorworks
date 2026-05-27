from __future__ import annotations

import re
import numpy as np
from PIL import Image, ImageOps

from colorworks.algorithms import StagedAlgorithm, registry, RenderContext
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

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}$|^#[0-9a-fA-F]{6}$")

def validate_color(hex_str: str) -> None:
    if not HEX_COLOR_RE.match(hex_str):
        raise ValueError(f"Invalid hex color format: {hex_str}. Must be #RGB or #RRGGBB.")

def parse_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)

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

def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0

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

        validate_color(ink_color)
        validate_color(paper_color)

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

        # Map binary output to ink/paper colors
        ink_rgb = parse_color(ink_color)
        paper_rgb = parse_color(paper_color)

        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[out] = paper_rgb
        canvas[~out] = ink_rgb

        img = Image.fromarray(canvas)
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
