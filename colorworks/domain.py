from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
class CalibrationAsset:
    id: str
    algorithm_id: str
    algorithm_version: str
    kind: Literal["lut", "tonal_art_map", "neural_weights", "reference_mask", "hvs_model"]
    storage_uri: str
    checksum: str
    size_bytes: int
    metadata: dict[str, Any]

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
    accepts_orientation: bool = False
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
    blend_mode: Literal["normal", "multiply"] = "normal"
    opacity: float = 1.0
    priority: int = 0

@dataclass(frozen=True)
class Composition:
    paper_color: PaletteColor
    layers: list[InkLayerSpec]
    output_size: tuple[int, int] | None = None

@dataclass
class WarmStartState:
    algorithm_id: str
    algorithm_version: str
    iteration: int
    energy: float | None
    params: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RenderResult:
    algorithm_primary_artifact_id: str | None = None
    default_composition: Composition | None = None
    final_artifact_id: str | None = None
    partial: bool = False
    warm_state: WarmStartState | None = None

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

@dataclass
class VectorField2D:
    substrate: Substrate
    data: np.ndarray  # [H, W, 2]
    is_bidirectional: bool = False

@dataclass
class StructureTensorField:
    substrate: Substrate
    data: np.ndarray  # [H, W, 3] = (Jxx, Jxy, Jyy)

@dataclass
class Polyline:
    points: np.ndarray  # [P, 2]
    closed: bool = False

@dataclass
class Stroke:
    path: Polyline
    width_profile: np.ndarray | None = None
    color_index: int = 0
    opacity_profile: np.ndarray | None = None

@dataclass
class StrokeSet:
    substrate: Substrate
    strokes: list[Stroke]

@dataclass
class PointSet:
    substrate: Substrate
    coords: np.ndarray          # [N, 2]  (x, y)
    radii: np.ndarray | None = None
    color_index: np.ndarray | None = None
    attributes: dict[str, np.ndarray] = field(default_factory=dict)

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

    def get_vector(self, name: str) -> VectorField2D:
        val = self.get(name)
        if not isinstance(val, VectorField2D):
            raise TypeError(f"{name} is not a VectorField2D")
        return val

    def get_strokes(self, name: str) -> StrokeSet:
        val = self.get(name)
        if not isinstance(val, StrokeSet):
            raise TypeError(f"{name} is not a StrokeSet")
        return val

@dataclass
class Artifact:
    id: str
    name: str
    type: str  # "scalar_field", "binary_mask", "raster_image", "vector_field_2d", etc.
    value: Any
    checksum: str
    viewer: ArtifactViewerSpec | None = None

def compute_array_checksum(arr: np.ndarray) -> str:
    # Use array byte copy or view to ensure stable byte representation
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def compute_strokeset_checksum(stroke_set: StrokeSet) -> str:
    h = hashlib.sha256()
    for stroke in stroke_set.strokes:
        h.update(np.ascontiguousarray(stroke.path.points).tobytes())
        h.update(bytes([1 if stroke.path.closed else 0]))
        if stroke.width_profile is not None:
            h.update(np.ascontiguousarray(stroke.width_profile).tobytes())
        h.update(bytes([stroke.color_index]))
    return h.hexdigest()

def render_vector_hsv(field: VectorField2D) -> Image.Image:
    arr = field.data
    H, W, _ = arr.shape
    theta = np.arctan2(arr[:, :, 1], arr[:, :, 0])
    theta_sym = np.mod(theta, np.pi)

    h_arr = (theta_sym / np.pi * 255.0).astype(np.uint8)
    s_arr = np.full((H, W), 255, dtype=np.uint8)
    v_arr = np.full((H, W), 255, dtype=np.uint8)

    h_img = Image.fromarray(h_arr, mode="L")
    s_img = Image.fromarray(s_arr, mode="L")
    v_img = Image.fromarray(v_arr, mode="L")

    hsv_img = Image.merge("HSV", (h_img, s_img, v_img))
    return hsv_img.convert("RGB")

def render_vector_glyphs(field: VectorField2D) -> Image.Image:
    arr = field.data
    H, W, _ = arr.shape
    img = Image.new("RGB", (W, H), (30, 30, 36))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    grid_size = 16
    for y in range(grid_size // 2, H, grid_size):
        for x in range(grid_size // 2, W, grid_size):
            vx, vy = arr[y, x]
            mag = np.hypot(vx, vy)
            if mag < 1e-5:
                continue
            dx = vx / mag
            dy = vy / mag
            length = 10.0
            x1 = x - dx * (length / 2.0)
            y1 = y - dy * (length / 2.0)
            x2 = x + dx * (length / 2.0)
            y2 = y + dy * (length / 2.0)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 255), width=2)
    return img

def render_tensor_anisotropy(field: StructureTensorField) -> Image.Image:
    arr = field.data
    Jxx = arr[:, :, 0]
    Jxy = arr[:, :, 1]
    Jyy = arr[:, :, 2]

    trace = Jxx + Jyy
    diff = Jxx - Jyy
    sqrt_term = np.sqrt(diff**2 + 4.0 * Jxy**2)

    lambda1 = 0.5 * (trace + sqrt_term)
    lambda2 = 0.5 * (trace - sqrt_term)

    denom = lambda1 + lambda2
    aniso = np.zeros_like(denom)
    mask = denom > 1e-6
    aniso[mask] = (lambda1[mask] - lambda2[mask]) / denom[mask]

    arr_uint8 = np.clip(aniso * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L")

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
        elif isinstance(value, VectorField2D):
            checksum = compute_array_checksum(value.data)
            type_name = "vector_field_2d"
        elif isinstance(value, StructureTensorField):
            checksum = compute_array_checksum(value.data)
            type_name = "structure_tensor_field"
        elif isinstance(value, StrokeSet):
            checksum = compute_strokeset_checksum(value)
            type_name = "stroke_set"
        elif isinstance(value, PointSet):
            checksum = compute_array_checksum(value.coords)
            type_name = "point_set"
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
        elif artifact.type == "vector_field_2d":
            img = render_vector_hsv(artifact.value)
            img.save(preview_path, format="PNG")
        elif artifact.type == "structure_tensor_field":
            img = render_tensor_anisotropy(artifact.value)
            img.save(preview_path, format="PNG")
        elif artifact.type == "stroke_set":
            # For preview of StrokeSet, draw black strokes on a transparent canvas
            H, W = artifact.value.substrate.height, artifact.value.substrate.width
            img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for stroke in artifact.value.strokes:
                pts = stroke.path.points
                if len(pts) < 2:
                    continue
                w = 1.0
                if stroke.width_profile is not None:
                    # draw segments
                    for i in range(len(pts) - 1):
                        p1 = pts[i]
                        p2 = pts[i + 1]
                        seg_w = float((stroke.width_profile[i] + stroke.width_profile[i + 1]) / 2.0)
                        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=(26, 26, 26, 255), width=max(1, int(seg_w)))
                else:
                    draw.line([(p[0], p[1]) for p in pts], fill=(26, 26, 26, 255), width=int(w))
            img.save(preview_path, format="PNG")
        elif artifact.type == "point_set":
            ps = artifact.value
            H, W = ps.substrate.height, ps.substrate.width
            img = Image.new("L", (W, H), 255)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for i in range(len(ps.coords)):
                x, y = ps.coords[i]
                r = float(ps.radii[i]) if ps.radii is not None else 2.0
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=0)
            img.save(preview_path, format="PNG")
        elif artifact.type == "raster_image":
            if isinstance(artifact.value, Image.Image):
                artifact.value.save(preview_path, format="PNG")
            elif hasattr(artifact.value, "data"):
                img = Image.fromarray(artifact.value.data)
                img.save(preview_path, format="PNG")


# ── Phase 3: cancellation, warm-start, iteration preview ──────────────────────

class CancelToken:
    def __init__(self) -> None:
        self._requested = False

    @property
    def requested(self) -> bool:
        return self._requested

    def cancel(self) -> None:
        self._requested = True


@dataclass(frozen=True)
class IterationPreview:
    mode: Literal["compose", "direct_raster", "inspector"]
    changed_artifact_ids: list[str] = field(default_factory=list)
    direct_raster: Any = None       # PIL Image for mode="direct_raster"
    inspector_artifact_id: str | None = None


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class PreviewRun:
    id: str
    session_id: str
    asset_id: str
    algorithm_id: str
    algorithm_version: str
    params: dict[str, Any]
    composition: Composition | None
    seed: int
    quality: str
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    superseded_at: datetime | None = None
    primary_artifact_id: str | None = None
    final_artifact_id: str | None = None
    error: str | None = None
    calibration_checksum: str | None = None
    calibration_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "asset_id": self.asset_id,
            "algorithm_id": self.algorithm_id,
            "status": self.status.value,
            "quality": self.quality,
            "primary_artifact_id": self.primary_artifact_id,
            "final_artifact_id": self.final_artifact_id,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "calibration_checksum": self.calibration_checksum,
            "calibration_version": self.calibration_version,
        }


@dataclass
class RenderRun:
    id: str
    asset_id: str
    algorithm_id: str
    algorithm_version: str
    params: dict[str, Any]
    composition: Composition | None
    seed: int
    quality: str
    status: RunStatus = RunStatus.QUEUED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    primary_artifact_id: str | None = None
    final_artifact_id: str | None = None
    error: str | None = None
    promoted_from_preview_id: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    recipe_id: str | None = None
    calibration_checksum: str | None = None
    calibration_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "algorithm_id": self.algorithm_id,
            "status": self.status.value,
            "quality": self.quality,
            "primary_artifact_id": self.primary_artifact_id,
            "final_artifact_id": self.final_artifact_id,
            "error": self.error,
            "promoted_from_preview_id": self.promoted_from_preview_id,
            "artifact_ids": self.artifact_ids,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "calibration_checksum": self.calibration_checksum,
            "calibration_version": self.calibration_version,
        }
