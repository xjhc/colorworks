from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal
import numpy as np
from PIL import Image

# Enums and Types
class AlgorithmFamily(str, Enum):
    TONE_ANALYSIS = "tone_analysis"
    STRUCTURE_ANALYSIS = "structure_analysis"
    DENSITY_FIELD = "density_field"
    DITHERING = "dithering"
    HALFTONING = "halftoning"
    STIPPLING = "stippling"
    FLOW_LINE = "flow_line"

class AlgorithmRole(str, Enum):
    ANALYZER = "analyzer"
    RENDERER = "renderer"

class ParameterType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STR = "str"

class RecomputeScope(str, Enum):
    FULL = "full"
    SYNTHESIS = "synthesis"
    COMPOSITE = "composite"
    DISPLAY = "display"

PatternKind = str

# Predicates for visibility / enabling
class Predicate:
    def evaluate(self, params: dict[str, Any]) -> bool:
        return True

@dataclass(frozen=True)
class Eq(Predicate):
    param_key: str
    value: Any

    def evaluate(self, params: dict[str, Any]) -> bool:
        return params.get(self.param_key) == self.value

# Inputs and Outputs Specs
@dataclass(frozen=True)
class InputSpec:
    primary: str = "raster"
    accepts_color: bool = True

@dataclass(frozen=True)
class OutputSpec:
    primary_artifact: str
    optional_artifacts: list[str] = field(default_factory=list)
    produces_composition: bool = False

@dataclass(frozen=True)
class ExecutionProfile:
    typical_runtime: str = "sub_second"
    is_iterative: bool = False
    is_streamable: bool = False
    is_cancellable: bool = False
    parallelism: str = "serial"
    memory_class: str = "small"

@dataclass(frozen=True)
class AlgorithmCapabilities:
    supports_raster_output: bool = False
    supports_vector_output: bool = False
    supports_multi_class: bool = False
    supports_interactive_preview: bool = True
    supports_progressive_refinement: bool = False
    deterministic: bool = True
    requires_gpu: bool = False

@dataclass(frozen=True)
class OptionDef:
    value: Any
    label: str

@dataclass(frozen=True)
class ParameterDef:
    key: str
    label: str
    type: ParameterType
    default: Any
    group: str = "general"
    description: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[OptionDef] | None = None
    ui_hint: str | None = None
    visible_when: Predicate | None = None
    enabled_when: Predicate | None = None
    invalidates: list[str] = field(default_factory=list)
    recompute_scope: RecomputeScope = RecomputeScope.FULL

@dataclass(frozen=True)
class ArtifactViewerSpec:
    default_view: str
    colormap: str | None = None
    value_range: tuple[float, float] | None = None

@dataclass(frozen=True)
class ArtifactKindDef:
    name: str
    type: str
    label: str
    suitable_as: list[str] = field(default_factory=list)
    viewer: ArtifactViewerSpec | None = None

@dataclass(frozen=True)
class CalibrationAssetRef:
    asset_id: str
    checksum: str

@dataclass(frozen=True)
class AlgorithmDefinition:
    id: str
    version: str
    family: AlgorithmFamily
    role: AlgorithmRole
    name: str
    description: str
    input_spec: InputSpec
    output_spec: OutputSpec
    parameters: list[ParameterDef]
    artifact_kinds: list[ArtifactKindDef]
    calibration_assets: list[CalibrationAssetRef] = field(default_factory=list)
    execution_profile: ExecutionProfile = field(default_factory=ExecutionProfile)
    capabilities: AlgorithmCapabilities = field(default_factory=AlgorithmCapabilities)

# Pattern Kind Def
@dataclass(frozen=True)
class PatternKindDef:
    kind: PatternKind
    name: str
    description: str
    parameters: list[ParameterDef]
    generation: Literal["procedural", "field", "geometry"] = "procedural"
    requires_density: bool = True
    requires_orientation: bool = False
    requires_warp: bool = False

# Coordinate Frame & Pattern Spec
@dataclass(frozen=True)
class PatternCoordinateSpec:
    space: Literal["image_px", "normalized", "output_px"] = "image_px"
    origin: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    rotation_deg: float = 0.0
    seed: int | None = None

@dataclass(frozen=True)
class PatternSpec:
    kind: PatternKind
    params: dict[str, Any] = field(default_factory=dict)
    field_source: str | None = None
    orientation_source: str | None = None
    warp_source: str | None = None
    mask_source: str | None = None
    coordinates: PatternCoordinateSpec = field(default_factory=PatternCoordinateSpec)

# Palette & Ink layer
@dataclass(frozen=True)
class PaletteColor:
    hex: str
    name: str | None = None

@dataclass(frozen=True)
class InkLayerSpec:
    name: str
    color: PaletteColor
    role: str
    density_source: str
    pattern: PatternSpec
    threshold: float | None = None
    blend_mode: Literal["normal", "multiply", "overprint", "screen"] = "normal"
    opacity: float = 1.0
    priority: int = 0

@dataclass(frozen=True)
class Composition:
    paper_color: PaletteColor
    layers: list[InkLayerSpec]
    output_size: tuple[int, int] | None = None

@dataclass(frozen=True)
class RenderResult:
    algorithm_primary_artifact_id: str
    default_composition: Composition | None = None

# Substrate, Fields, and Masks
@dataclass(frozen=True)
class RasterGrid:
    width: int
    height: int
    pixel_size: float = 1.0

Substrate = RasterGrid

@dataclass
class ScalarField:
    substrate: Substrate
    data: np.ndarray
    dtype: Literal["float32", "uint8", "uint16"]
    range: tuple[float, float] = (0.0, 1.0)

@dataclass
class BinaryMask:
    substrate: Substrate
    data: np.ndarray  # bool

# WorkingSet and ArtifactStore
class WorkingSet:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def put(self, name: str, value: Any) -> None:
        self._data[name] = value

    def has(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str) -> Any:
        return self._data[name]

    def get_scalar(self, name: str) -> ScalarField:
        val = self.get(name)
        if not isinstance(val, ScalarField):
            raise TypeError(f"{name} is not a ScalarField")
        return val

    def get_mask(self, name: str) -> BinaryMask:
        val = self.get(name)
        if not isinstance(val, BinaryMask):
            raise TypeError(f"{name} is not a BinaryMask")
        return val

@dataclass
class Artifact:
    id: str
    name: str
    type: str  # "scalar_field", "binary_mask", "raster_image", etc.
    value: Any
    checksum: str
    viewer: ArtifactViewerSpec | None = None

def compute_array_checksum(arr: np.ndarray) -> str:
    # Use array byte copy or view to ensure stable byte representation
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

class ArtifactStore:
    def __init__(self, output_dir: Path | None = None) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._name_to_id: dict[str, str] = {}
        self.output_dir = output_dir

    def has(self, name: str) -> bool:
        return name in self._name_to_id

    def publish(
        self,
        name: str,
        value: Any,
        viewer: ArtifactViewerSpec | None = None,
        iteration: int | None = None,
    ) -> str:
        if isinstance(value, ScalarField):
            checksum = compute_array_checksum(value.data)
            type_name = "scalar_field"
        elif isinstance(value, BinaryMask):
            checksum = compute_array_checksum(value.data)
            type_name = "binary_mask"
        elif isinstance(value, Image.Image):
            buf = io.BytesIO()
            value.save(buf, format="PNG")
            checksum = hashlib.sha256(buf.getvalue()).hexdigest()
            type_name = "raster_image"
        elif hasattr(value, "data") and isinstance(value.data, np.ndarray):
            checksum = compute_array_checksum(value.data)
            type_name = "raster_image"
        else:
            checksum = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            type_name = "other"

        artifact_id = f"{name}_{checksum[:16]}"
        artifact = Artifact(
            id=artifact_id,
            name=name,
            type=type_name,
            value=value,
            checksum=checksum,
            viewer=viewer,
        )
        self._artifacts[artifact_id] = artifact
        self._name_to_id[name] = artifact_id

        if self.output_dir:
            self.save_preview(artifact)

        return artifact_id

    def list(self) -> list[str]:
        return list(self._artifacts.keys())

    def get(self, id: str) -> Artifact:
        if id not in self._artifacts:
            raise KeyError(f"Artifact {id} not found")
        return self._artifacts[id]

    def get_by_name(self, name: str) -> Artifact:
        if name not in self._name_to_id:
            raise KeyError(f"Artifact name {name} not found")
        return self._artifacts[self._name_to_id[name]]

    def save_preview(self, artifact: Artifact) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = self.output_dir / f"{artifact.id}.png"
        
        # Don't recreate if already exists
        if preview_path.exists():
            return

        if artifact.type == "scalar_field":
            arr = artifact.value.data
            arr_uint8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(arr_uint8, mode="L")
            img.save(preview_path, format="PNG")
        elif artifact.type == "binary_mask":
            arr = artifact.value.data
            arr_uint8 = np.where(arr, 255, 0).astype(np.uint8)
            img = Image.fromarray(arr_uint8, mode="L")
            img.save(preview_path, format="PNG")
        elif artifact.type == "raster_image":
            if isinstance(artifact.value, Image.Image):
                artifact.value.save(preview_path, format="PNG")
            elif hasattr(artifact.value, "data"):
                img = Image.fromarray(artifact.value.data)
                img.save(preview_path, format="PNG")
