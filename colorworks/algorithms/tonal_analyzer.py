from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from colorworks.algorithms import StagedAlgorithm, registry
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
    Eq,
    ScalarField,
    BinaryMask,
    Composition,
    InkLayerSpec,
    PaletteColor,
    PatternSpec,
    PatternCoordinateSpec,
    RenderResult,
    PatternKindDef,
)

DEFINITION = AlgorithmDefinition(
    id="tonal_analyzer",
    version="1.0.0",
    family=AlgorithmFamily.TONE_ANALYSIS,
    role=AlgorithmRole.ANALYZER,
    name="Tonal Analyzer",
    description="Tone map + edge mask. Compose with ink layers and pattern kinds.",
    input_spec=InputSpec(primary="raster", accepts_color=True),
    output_spec=OutputSpec(
        primary_artifact="tone_map",
        optional_artifacts=["edge_mask"],
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
            "preserve_edges",
            "Edge preservation",
            ParameterType.BOOL,
            default=True,
            group="structure",
            invalidates=["edge_mask"],
        ),
        ParameterDef(
            "edge_threshold",
            "Edge threshold",
            ParameterType.FLOAT,
            default=0.15,
            min=0.0,
            max=1.0,
            step=0.01,
            group="structure",
            visible_when=Eq("preserve_edges", True),
            invalidates=["edge_mask"],
        ),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="tone_map",
            type="scalar_field",
            label="Tone Map",
            suitable_as=["density_source"],
            viewer=ArtifactViewerSpec(default_view="heatmap", colormap="gray"),
        ),
        ArtifactKindDef(
            name="edge_mask",
            type="binary_mask",
            label="Edges",
            suitable_as=["edge_mask", "mask_source"],
            viewer=ArtifactViewerSpec(default_view="mask"),
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

def remap_tone(gray: np.ndarray, contrast: float, midpoint: float) -> np.ndarray:
    return np.clip((gray - midpoint) * contrast + 0.5, 0.0, 1.0)

def convolve2d_nearest(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    output = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            output += padded[i : i + H, j : j + W] * kernel[i, j]
    return output

def sobel_edge_mask(gray: np.ndarray, threshold: float) -> np.ndarray:
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    
    Gx = convolve2d_nearest(gray, Kx)
    Gy = convolve2d_nearest(gray, Ky)
    magnitude = np.hypot(Gx, Gy)
    return magnitude >= threshold

class TonalAnalyzer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["tone_map", "edge_mask"]
    produced_in_synthesize = []

    def analyze(self, ctx: RenderContext) -> None:
        gray = None

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

        if ctx.params.get("preserve_edges", True):
            if not ctx.store.has("edge_mask"):
                if gray is None:
                    gray = to_gray(ctx.input.image)
                edges = sobel_edge_mask(
                    gray,
                    threshold=ctx.params.get("edge_threshold", 0.15)
                )
                ctx.store.publish(
                    "edge_mask",
                    BinaryMask(ctx.input.substrate, edges),
                )

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        default = Composition(
            paper_color=PaletteColor("#f4ebd9", "paper"),
            layers=[
                InkLayerSpec(
                    name="ink",
                    color=PaletteColor("#1a1a1a", "ink"),
                    role="shadow",
                    density_source="tone_map",
                    pattern=PatternSpec(
                        kind="wave",
                        params={
                            "frequency": 8.0,
                            "angle_deg": 45.0,
                            "phase": 0.0,
                        },
                        mask_source="edge_mask" if ctx.params.get("preserve_edges", True) else None,
                        coordinates=PatternCoordinateSpec(
                            space="image_px",
                            seed=ctx.seed,
                        ),
                    ),
                ),
            ],
        )
        return RenderResult(
            algorithm_primary_artifact_id=ctx.working.get("tone_id"),
            default_composition=default,
        )

    def is_artifact_enabled(self, name: str, params: dict[str, Any]) -> bool:
        if name == "edge_mask":
            return params.get("preserve_edges", True)
        return True

    def load_from_cache(self, ctx: RenderContext, artifacts: dict[str, str]) -> None:
        if "tone_map" in artifacts:
            ctx.working.put("tone_id", artifacts["tone_map"])

# Register the algorithm
registry.register(TonalAnalyzer())

# Register the wave pattern kind
registry.register_pattern(PatternKindDef(
    kind="wave",
    name="Wave",
    description="Sinusoidal pattern modulated by density.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef(
            "frequency",
            "Frequency (cycles / 100 image_px)",
            ParameterType.FLOAT,
            default=8.0,
            min=0.5,
            max=64.0,
            step=0.5,
        ),
        ParameterDef(
            "angle_deg",
            "Angle (deg)",
            ParameterType.FLOAT,
            default=45.0,
            min=0.0,
            max=180.0,
            step=1.0,
        ),
        ParameterDef(
            "phase",
            "Phase",
            ParameterType.FLOAT,
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
        ),
    ],
))
