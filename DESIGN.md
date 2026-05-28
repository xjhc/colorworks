# Colorworks — System Design

## 0. Reading guide

Two framings up front:

1. **Colorworks is an image-processing tool, not a research platform.** The product loop is image input -> print-process-inspired raster/vector output -> export. Pang halftoning, CVT stippling, ETF/FDoG line work, SAED, and DBS shape the core roadmap; mesh, video, and neural-heavy substrate work are deferred research, not implied product phases. §14 is the binding plan; everything earlier is the shape the implementation should grow into without rewrites.
2. **DDD is a tool, not a religion.** Bounded contexts and aggregates appear where they prevent real coupling. No application/infrastructure layering — premature at this scale.

Internal package name: **`colorworks`**. Use it everywhere.

---

## 1. Two theses

The whole design rests on two load-bearing decisions. Read these first; the rest follows.

### 1.1 The product is ink layers, not pixels

The domain center is:

```
source image → structural fields → ink layers (color + pattern + density source) → composition
```

A `Palette` as "list of colors" is too thin. The right abstraction is closer to risograph / screenprint: each ink layer carries a role (shadow / midtone / highlight / edge / paper), a color, a pattern (wave / hatch / dither / stipple), and references to the structural fields driving it (tone map, orientation field, edge mask).

This lets the same underlying pattern field drive multiple ink layers, lets palettes change without re-running analysis, and matches what users actually want to control.

### 1.2 Iteration is first-class

Half the eventual algorithms iterate (Pang annealing, Lloyd relaxation, ETF smoothing, electrostatic N-body, DBS). Forcing them into `analyze / synthesize / compose` would hide progress, block streaming previews, and prevent warm starts.

The contract is `async def render(ctx) -> AsyncIterator[RenderProgress]`. `StagedAlgorithm` and `IterativeAlgorithm` are opt-in helper base classes, not the contract itself.

---

## 2. Bounded contexts

```
┌─────────────────┐     ┌──────────────────────┐
│  Asset          │     │  Algorithm Catalog   │
│  - RasterImage  │     │  - AlgorithmDef      │
│                 │     │  - ParameterSchema   │
│  (deferred:     │     │  - PatternKindDef    │
│   Mesh, Video)  │     │  - CalibrationAsset  │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────┐
│  Workspace & Recipe                         │
│  - Project, Recipe, Preset                  │
│  - Composition (ink layers + patterns)      │
└────────────────────┬────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  Rendering & Artifacts                      │
│  - PreviewRun, RenderRun, RenderProgress    │
│  - WorkingSet, ArtifactStore                │
│  - Scheduler, Cache, SSE bus, Compositor    │
└─────────────────────────────────────────────┘
```

*Asset* and *Catalog* never depend on user state — calibration LUTs, pattern kinds, algorithm metadata, and source media are versioned and deployed independently of any project. *Workspace* references both and adds user-controlled `Composition` (ink layers). *Rendering* depends on all three; it's where you'll iterate fastest, so we keep upstream contexts stable.

This is the only context split that pays off. Don't sub-layer inside each context until you feel real pain.

---

## 3. Core types

### 3.1 Substrate

The main product ships raster-image workflows. Mesh and Video are declared so the type system has a place for future experiments, but they are deferred research sketches, not binding roadmap deliverables.

```python
@dataclass(frozen=True)
class RasterGrid:
    width: int
    height: int
    pixel_size: float = 1.0       # source units; matters for print/HVS modeling

# Deferred research substrates -- not part of the main roadmap:
@dataclass(frozen=True)
class MeshSurface:
    mesh_id: str
    elements: Literal["vertex", "face", "halfedge"]

@dataclass(frozen=True)
class VideoVolume:
    width: int
    height: int
    n_frames: int
    fps: float

Substrate = RasterGrid | MeshSurface | VideoVolume
SubstrateKind = Literal["raster", "mesh", "video"]
```

### 3.2 Fields and masks

```python
@dataclass
class ScalarField:
    substrate: Substrate
    data: np.ndarray
    dtype: Literal["float32", "uint8", "uint16"]
    range: tuple[float, float] = (0.0, 1.0)

@dataclass
class VectorField2D:
    substrate: Substrate
    data: np.ndarray              # raster: [H, W, 2]
    is_bidirectional: bool = False   # ETF tangents True; flow fields False

@dataclass
class StructureTensorField:
    substrate: Substrate
    data: np.ndarray              # [H, W, 3] = (Jxx, Jxy, Jyy)

@dataclass
class BinaryMask:
    substrate: Substrate
    data: np.ndarray              # bool

@dataclass
class LabelField:
    substrate: Substrate
    data: np.ndarray              # int32, region/superpixel labels

# Deferred research / substrate expansion:
@dataclass
class CrossField:                 # N-RoSy on a mesh
    substrate: MeshSurface
    coefficients: np.ndarray      # complex, [F]
    N: int                        # 2 line field, 4 cross-hatch
```

### 3.3 Geometry

```python
@dataclass
class PointSet:
    substrate: Substrate
    coords: np.ndarray            # [N, 2]
    radii: np.ndarray | None = None
    color_index: np.ndarray | None = None    # multi-class stippling
    attributes: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class Polyline:
    points: np.ndarray            # [P, 2]
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
class PolygonSet:                 # for graph-based vector stylization
    substrate: Substrate
    polygons: list[np.ndarray]
    fill_index: list[int]
```

### 3.4 WorkingSet vs. ArtifactStore

The single most important change versus a naive `AnalysisBundle`-style design: **separate transient computation state from persisted, viewable outputs**. Not every intermediate is a stored artifact, and the algorithm author should choose explicitly.

```python
class WorkingSet:
    """In-memory typed artifacts used during a single render. Cheap, transient.
    Algorithms accumulate scratch fields here; nothing leaves the process unless
    explicitly published to the store."""

    def put(self, name: str, value: Any) -> None: ...
    def has(self, name: str) -> bool: ...
    def get(self, name: str) -> Any: ...

    # Typed accessors — runtime-checked against declared artifact kinds:
    def get_scalar(self, name: str) -> ScalarField: ...
    def get_vector(self, name: str) -> VectorField2D: ...
    def get_mask(self, name: str) -> BinaryMask: ...
    def get_points(self, name: str) -> PointSet: ...
    def get_strokes(self, name: str) -> StrokeSet: ...


class ArtifactStore:
    """Persisted, content-addressed artifacts visible in the inspector and
    re-usable by the compositor. Each publish() yields a stable ArtifactId."""

    def publish(
        self,
        name: str,                # matches an ArtifactKindDef.name
        value: Any,
        viewer: ArtifactViewerSpec | None = None,
        iteration: int | None = None,    # for streamed iteration previews
    ) -> ArtifactId: ...

    def list(self) -> list[ArtifactId]: ...
    def get(self, id: ArtifactId) -> Artifact: ...
```

The algorithm author keeps a working `WorkingSet` cheaply. When something is worth persisting (because the inspector should show it, the cache should keep it, or the compositor will consume it), the author calls `ArtifactStore.publish`. Large transient buffers (Pang's swap proposals, Lloyd's intermediate cell rasters) stay in `WorkingSet` and are dropped at run end.

`RenderContext` (§6.1) carries both.

### 3.5 ArtifactViewerSpec

The UI can't render generically across artifact types without hints. A vector field could be shown as arrows, as an HSV angle map, as an LIC texture, or as downsampled glyphs — and the algorithm author knows which is right.

```python
@dataclass(frozen=True)
class ArtifactViewerSpec:
    default_view: Literal[
        "raster", "heatmap", "mask", "labelmap",
        "glyph_field", "orientation_hsv", "lic",
        "points", "strokes", "svg", "composition",
    ]
    downsample_policy: Literal["block_max", "block_mean", "subsample"] | None = None
    value_range: tuple[float, float] | None = None
    colormap: str | None = None                # "viridis", "magma", "diverging", ...
    overlay_on: str | None = None              # composite atop another artifact in inspector
```

Default views by type are filled in if the algorithm doesn't specify, but every algorithm can override per artifact. `/artifacts/{id}/preview` (§9.1) uses the viewer to materialize a thumbnail.

---

## 4. Ink layers, patterns, composition

This is the product-shaped layer. **Algorithms produce structural fields. Ink layers + patterns turn those fields into pixels.** The split lets palettes and patterns change without re-running analysis, and lets one analysis drive multiple expressive renderings.

### 4.1 Ink layers

```python
@dataclass(frozen=True)
class PaletteColor:
    hex: str
    name: str | None = None

@dataclass(frozen=True)
class InkLayerSpec:
    name: str                              # "shadow_ink", "highlight_ink"
    color: PaletteColor
    role: Literal["shadow", "midtone", "highlight", "edge", "accent", "paper"]

    density_source: str                    # ArtifactStore name (e.g. "tone_map")
    pattern: PatternSpec
    threshold: float | None = None         # density cutoff before pattern applies

    blend_mode: Literal["normal", "multiply", "overprint", "screen"] = "normal"
    opacity: float = 1.0
    priority: int = 0                      # render order (lower first)
```

### 4.2 Pattern specs

Patterns are the language for "use this micro-structure as a tone carrier." **The Compositor owns procedural pattern generation by default.** Algorithms produce structural fields (tone, edges, orientation, density); ink layers reference those fields as `density_source` / `orientation_source`, and patterns are synthesized in the Compositor from `PatternSpec.params`.

The escape hatch — `kind="field"` — lets an algorithm publish a pre-computed pattern as a `ScalarField` artifact and have the ink layer use it directly. Reserved for algorithm-specific or expensive patterns (e.g., a learned pattern field from a neural component).

```python
@dataclass(frozen=True)
class PatternSpec:
    kind: PatternKind                      # see registry below
    params: dict[str, Any]                 # validated against PatternKindDef.parameters

    # Optional artifact sources (names in ArtifactStore):
    field_source: str | None = None        # required if kind == "field"
    orientation_source: str | None = None  # for kinds that consume a VectorField2D
    warp_source: str | None = None         # for kinds that consume a warp VectorField2D
    mask_source: str | None = None         # optional BinaryMask clip

    # Coordinate frame for procedural patterns — see PatternCoordinateSpec:
    coordinates: PatternCoordinateSpec = field(default_factory=PatternCoordinateSpec)


PatternKind = Literal[
    "solid",                # no pattern; flat ink wherever density > threshold
    "field",                # use PatternSpec.field_source ScalarField as the pattern
    "ordered_dither",       # Bayer / classic threshold matrix (procedural)
    "blue_noise",           # void-and-cluster threshold mask (procedural)
    "wave",                 # sine/cosine field, frequency + angle + phase (procedural)
    "zigzag",               # triangle wave (procedural)
    "maze",                 # warped maze pattern (procedural)
    "hatch",                # parallel lines, orientation-driven (procedural)
    "crosshatch",           # two-direction hatching (procedural)
    "stipple",              # point cloud (geometry — from PointSet artifact)
    "stroke_set",           # vector strokes (geometry — from StrokeSet artifact)
]
```

Pattern kinds are declared in a `PatternKindDef` registry parallel to `AlgorithmDef`. The schema-driven UI uses these for per-layer pattern controls.

```python
@dataclass(frozen=True)
class PatternKindDef:
    kind: PatternKind
    name: str
    description: str
    parameters: list[ParameterDef]

    generation: Literal["procedural", "field", "geometry"]
    # procedural — Compositor synthesizes from params (wave, maze, dither, hatch, ...)
    # field      — Pattern is a ScalarField artifact; PatternSpec.field_source required
    # geometry   — Pattern is a PointSet / StrokeSet artifact rendered by the Compositor

    requires_density: bool = True
    requires_orientation: bool = False
    requires_warp: bool = False
```

#### Pattern coordinate frame

Without an explicit coordinate spec, the same recipe rendered at preview (e.g., 512px) vs. full (e.g., 2048px) vs. export DPI will produce visibly different patterns — frequencies drift, dot sizes shift, seeds re-shuffle. Make the frame explicit:

```python
@dataclass(frozen=True)
class PatternCoordinateSpec:
    space: Literal["image_px", "normalized", "output_px"] = "image_px"
    # image_px   — tied to source image pixels; same density across zoom levels (default)
    # normalized — [0, 1] canvas-relative; resolution-independent stylization
    # output_px  — tied to render output pixels; DPI-aware print

    origin: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0                     # multiplier on pattern frequency
    rotation_deg: float = 0.0              # extra rotation on top of pattern's own angle
    seed: int | None = None                # for stochastic patterns; falls back to run seed
```

Default `space="image_px"` means a wave's frequency in cycles-per-100-pixels is interpreted against the source image, so a 1× preview and a 4× export show the same number of wave cycles per region. For print output where DPI matters, switch to `output_px`. For resolution-independent stylization (rare but possible), `normalized`.

Seed precedence is explicit: `PatternCoordinateSpec.seed` wins when set; otherwise the pattern uses the run seed from `RenderContext`. Presets that should reproduce an exact stochastic texture set per-pattern seeds. Presets that should re-roll texture on each new render leave them null.

Preview fidelity rule: preview renders may use a lower output resolution, but pattern coordinates are still evaluated in their declared space. For `image_px`, the pattern frequency is derived from source-image coordinates and then sampled to the preview canvas; do not reinterpret "cycles per 100 image pixels" against the reduced preview grid. If this is too slow for a pattern kind, mark the preview as approximate in `ArtifactViewerSpec` rather than silently changing the coordinate meaning.

### 4.3 Composition

```python
@dataclass(frozen=True)
class Composition:
    paper_color: PaletteColor
    layers: list[InkLayerSpec]
    output_size: tuple[int, int] | None = None    # None = inherit from source
```

### 4.4 The Compositor service

Built-in. Not user-extensible in v0. After an algorithm completes with a `Composition` attached:

1. For each layer (in priority order), read its source artifacts from the `ArtifactStore`:
   - `density_source` (required) — a `ScalarField`.
   - `orientation_source`, `warp_source`, `mask_source` (optional, kind-dependent).
   - `field_source` (when `pattern.kind == "field"`) — a `ScalarField`.
2. Generate the layer's pattern at the canvas resolution honoring `pattern.coordinates`:
   - procedural kinds → synthesized from `pattern.params`
   - field kind → resampled from the source ScalarField
   - geometry kinds → rasterized from the source PointSet / StrokeSet
3. Threshold the pattern against the density field (with optional `mask_source` clip).
4. Composite onto the canvas using `blend_mode`.
5. Publish the user-visible output as `final_raster` (or `final_svg` for vector-only compositions).

If an algorithm has `role=RENDERER` and `Composition` is null, the algorithm's primary artifact is the final output and the Compositor is skipped. If a `Composition` is attached (either as the algorithm's `default_composition` or supplied by the recipe), the algorithm's published artifacts are inputs to the Compositor and its output is the final.

**Final artifact rule:** `RenderRun.primary_artifact_id`, export endpoints, and history thumbnails always point at the user-visible final artifact. With a `Composition`, that is the Compositor-published `final_raster` / `final_svg`. Without a `Composition`, it is the renderer algorithm's direct output. Structural artifacts such as `tone_map`, `edge_mask`, and `orientation_field` stay inspectable but are never the run primary unless the user explicitly exports that inspector tab.

This is the canonical interactive case: one tone-map analyzer re-rendered with different inks/patterns without re-running the analyzer.

### 4.5 Suitability hints

`ArtifactKindDef.suitable_as` declares what UI dropdowns should offer. When a user picks an ink layer's `density_source`, the UI offers artifacts where `suitable_as` includes `"density_source"`.

```python
@dataclass(frozen=True)
class ArtifactKindDef:
    name: str
    type: ArtifactDataType
    label: str
    persist: bool = True
    suitable_as: list[Literal[
        "density_source", "orientation_source", "warp_source",
        "mask_source", "edge_mask", "region_mask", "final",
    ]] = field(default_factory=list)
    viewer: ArtifactViewerSpec | None = None
```

---

## 5. Algorithm catalog

### 5.1 AlgorithmDefinition

```python
@dataclass(frozen=True)
class AlgorithmDefinition:
    id: str                              # "tonal_analyzer"
    version: str                         # semver — breaking changes bump major
    family: AlgorithmFamily
    role: AlgorithmRole
    name: str
    description: str

    input_spec: InputSpec
    output_spec: OutputSpec
    parameters: list[ParameterDef]
    artifact_kinds: list[ArtifactKindDef]
    calibration_assets: list[CalibrationAssetRef]

    execution_profile: ExecutionProfile
    capabilities: AlgorithmCapabilities


class AlgorithmFamily(str, Enum):
    """Describes the mathematical / artistic method."""
    TONE_ANALYSIS = "tone_analysis"            # tone map, contrast, midpoint
    STRUCTURE_ANALYSIS = "structure_analysis"  # edges, orientation, structure tensor
    DENSITY_FIELD = "density_field"            # density / saliency / importance maps
    DITHERING = "dithering"                    # error-diffusion or threshold-matrix renderers
    HALFTONING = "halftoning"                  # structure-aware halftone renderers
    STIPPLING = "stippling"                    # point-set renderers (CVT, electrostatic)
    FLOW_LINE = "flow_line"                    # ETF/FDoG/streamline-based line renderers
    HATCHING_2D = "hatching_2d"                # orientation-driven hatch renderers
    HATCHING_SURFACE = "hatching_surface"      # deferred mesh/surface research
    NEURAL_HYBRID = "neural_hybrid"            # deferred learned components


class AlgorithmRole(str, Enum):
    """Describes how the framework executes the algorithm."""
    ANALYZER = "analyzer"                # publishes fields; Compositor runs after
    RENDERER = "renderer"                # produces final raster/vector; Compositor skipped
```

`family` is the method; `role` is the framework's execution contract. A `TONE_ANALYSIS` algorithm has `role=ANALYZER`; a `HALFTONING` algorithm typically has `role=RENDERER` (Pang produces a direct binary output); a `STRUCTURE_ANALYSIS` algorithm has `role=ANALYZER` (publishes edges/orientation for downstream ink layers).

### 5.2 Input / output specs

```python
@dataclass(frozen=True)
class InputSpec:
    primary: SubstrateKind               # "raster" only in v0
    accepts_color: bool
    min_resolution: tuple[int, int] | None = None
    max_resolution: tuple[int, int] | None = None

@dataclass(frozen=True)
class OutputSpec:
    primary_artifact: str                # algorithm-owned primary; not necessarily run primary
    optional_artifacts: list[str]
    produces_composition: bool = False   # True if algorithm emits a default Composition
```

### 5.3 ParameterDef

```python
@dataclass(frozen=True)
class ParameterDef:
    key: str
    label: str
    type: ParameterType
    default: ParameterValue | DefaultExpr     # may be computed from input dims
    group: str = "general"
    description: str = ""

    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[OptionDef] | None = None
    ui_hint: Literal["slider", "input", "toggle", "select",
                     "color", "vector2", "curve", "palette"] | None = None

    visible_when: Predicate | None = None
    enabled_when: Predicate | None = None
    validates_with: list[CrossParamValidator] = field(default_factory=list)

    # Invalidation — see §5.4:
    invalidates: list[str] = field(default_factory=list)
    recompute_scope: RecomputeScope = RecomputeScope.FULL    # shorthand fallback
```

### 5.4 Invalidation as a dependency graph, not an enum

`RecomputeScope` is too coarse once an algorithm has more than a handful of intermediate artifacts. Move the truth to **per-artifact dependency keys**:

```python
class RecomputeScope(str, Enum):
    FULL        = "full"          # invalidates everything; shorthand
    SYNTHESIS   = "synthesis"     # analyze-stage artifacts survive
    COMPOSITE   = "composite"     # synthesis artifacts survive; only re-composite
    DISPLAY     = "display"       # client-side only (zoom, pan, opacity)
```

```python
ParameterDef(
    key="contrast",
    invalidates=["tone_map"],                # downstream nodes drop automatically
)
ParameterDef(
    key="frequency",                          # in PatternKindDef("wave")
    invalidates=["pattern_field_<layer>"],
)
ParameterDef(
    key="ink_color",                          # in InkLayerSpec
    invalidates=["composition"],
)
ParameterDef(
    key="zoom",
    invalidates=[],
)
```

The cache keys each *artifact* separately (not just the whole run, §8.5). When a parameter changes, the framework drops invalidated artifacts and **everything downstream in the DAG**, then re-runs only the stages that depend on them.

`recompute_scope` stays as a convenient shorthand for declarations. Simple algorithms can rely on it; precise algorithms supply explicit `invalidates` lists. If `invalidates` is non-empty, it is authoritative and `recompute_scope` is ignored for that parameter. Registry validation should reject parameters that try to use both as independent sources of truth.

### 5.4.1 The dependency DAG

`invalidates=[...]` entries must reference declared nodes in a static dependency graph. Without that constraint, `invalidates` strings become silently-broken keys that the cache can't reason about.

```python
@dataclass(frozen=True)
class DependencyNode:
    key: str
    kind: Literal["artifact", "stage", "compositor_internal"]
    depends_on: list[str]
```

Algorithms contribute nodes for each `ArtifactKindDef`. The Compositor contributes its own internal nodes. The framework refuses to register an algorithm whose `ParameterDef.invalidates` references an undeclared key.

Pattern kinds are the exception that proves the rule: they are registered as templates because they do not know their layer name yet. A pattern parameter may reference templated nodes such as `pattern_field_<layer>` or `ink_mask_<layer>`; those placeholders are validated against the Compositor template at pattern-kind registration time, then expanded to concrete keys (`pattern_field_shadow`, `ink_mask_shadow`, etc.) when a `Composition` is instantiated. Recipe save/preview validation is the point where concrete layer names, source artifacts, and expanded DAG keys must all resolve.

V0 compositor DAG (illustrative — per-layer keys are templated):

```
tone_map                  artifact            (algorithm-provided)
edge_mask                 artifact            (algorithm-provided)
orientation_field         artifact            (algorithm-provided)
pattern_field_<layer>     compositor_internal depends_on: [orientation_source?, warp_source?]
ink_mask_<layer>          compositor_internal depends_on: [density_source, pattern_field_<layer>, mask_source?]
composition               compositor_internal depends_on: [ink_mask_<every-layer>]
final_raster              artifact            depends_on: [composition]
```

When a parameter changes, the framework collects its `invalidates` set, transitively closes downstream nodes through the DAG, and drops their cache entries. Everything upstream of the invalidated set hits cache as before.

### 5.5 Predicate AST for conditional UI

```python
Predicate = Eq | NotEq | In | And | Or | Not | GreaterThan | LessThan
# Each carries a `key` ref and either a literal or another Predicate.
```

Serialize to JSON. One evaluator on each side. No per-algorithm DSL.

### 5.6 Calibration assets

SAED/DBS LUTs (Phase 4), blue-noise reference masks, and deferred research assets such as Tonal Art Maps or neural weights are owned by an algorithm version, not by a recipe.

```python
@dataclass(frozen=True)
class CalibrationAsset:
    id: str
    algorithm_id: str
    algorithm_version: str
    kind: Literal["lut", "tonal_art_map", "neural_weights", "reference_mask"]
    storage_uri: str
    checksum: str
    size_bytes: int
    metadata: dict[str, Any]
```

**Versioning rule:** re-cooking a calibration asset is a breaking change → algorithm version bump. Keeps reproducibility honest.

This is deliberately strict for v0. If calibration iteration becomes operationally painful, split `algorithm_version` and `calibration_version` in `RenderRun` snapshots, but do not silently replace an asset behind an existing algorithm version.

### 5.7 Pattern kind catalog

`PatternKindDef` is defined in §4.2. Each pattern kind ships with its own parameter schema, used by the UI for the per-layer pattern panel. The Compositor (§4.4) dispatches on `PatternSpec.kind`.

Pattern kinds are registered the same way algorithms are:

```python
registry.register_pattern(PatternKindDef(
    kind="wave",
    name="Wave",
    description="Sinusoidal pattern modulated by density.",
    generation="procedural",
    requires_density=True,
    parameters=[
        ParameterDef("frequency", "Frequency (cycles / 100 image_px)",
                     ParameterType.FLOAT, default=8.0, min=0.5, max=64.0, step=0.5),
        ParameterDef("angle_deg", "Angle (deg)", ParameterType.FLOAT,
                     default=45.0, min=0.0, max=180.0, step=1.0),
        ParameterDef("phase", "Phase", ParameterType.FLOAT,
                     default=0.0, min=0.0, max=1.0, step=0.01),
    ],
))
```

Pattern parameters participate in the same `invalidates` / DAG model as algorithm parameters. Each layer instantiates per-layer DAG nodes; the Compositor templates them by layer name.

---

## 6. The algorithm contract

```python
class RenderAlgorithm(Protocol):
    definition: AlgorithmDefinition

    async def render(self, ctx: RenderContext) -> AsyncIterator[RenderProgress]:
        """Yields progress events. Final yield must be kind='completed', 'cancelled', or 'failed'."""
```

That's the entire contract. Helper base classes follow.

### 6.1 RenderContext

```python
@dataclass
class RenderContext:
    input: MediaAsset
    params: ParameterSet
    composition: Composition | None
    seed: int
    quality: QualityMode                  # "preview" | "draft" | "full"
    working: WorkingSet                   # transient
    store: ArtifactStore                  # persisted
    cancel: CancelToken                   # cooperative
    log: RenderLogger
    calibration: CalibrationAccessor
    preview: PreviewCompositor            # framework-owned preview materializer
    rng: np.random.Generator              # seeded from `seed`
    warm_start: WarmStartState | None = None    # see §6.5
```

### 6.2 Progress events

```python
@dataclass
class RenderProgress:
    kind: Literal["started", "artifact", "iteration", "completed", "cancelled", "failed"]

    iteration: int | None = None
    total_iterations: int | None = None
    energy: float | None = None
    delta: float | None = None
    preview_artifact_id: ArtifactId | None = None   # downsampled for live UI

    artifact_kind: str | None = None
    artifact_id: ArtifactId | None = None

    result: RenderResult | None = None
    error: RenderError | None = None
```

Patterns:
- **Cheap algorithms** (`TonalAnalyzer`, ordered-dither renderer): single `started → completed`.
- **Iterative** (Pang, Lloyd, ETF smoothing): throttled `iteration` events (≤4 Hz) with optional preview thumbnails.
- **Staged**: `artifact` events after each stage so the inspector populates progressively.

### 6.3 Execution profile

```python
@dataclass(frozen=True)
class ExecutionProfile:
    typical_runtime: Literal["sub_second", "seconds", "minutes", "long"]
    is_iterative: bool
    is_streamable: bool
    is_cancellable: bool
    parallelism: Literal["serial", "threaded", "gpu", "distributed"]
    memory_class: Literal["small", "moderate", "large"]
```

Scheduler reads this to decide inline vs. worker dispatch, SSE wiring, GPU slot allocation, and cancellation timeout.

### 6.4 Helper bases

Iteration previews are framework-composed when possible. Algorithms publish partial fields, point sets, or direct rasters; they do **not** invoke the Compositor themselves.

```python
@dataclass
class IterationPreview:
    mode: Literal["compose", "direct_raster", "inspector"]
    changed_artifact_ids: list[ArtifactId] = field(default_factory=list)
    direct_raster: RasterImage | None = None
    inspector_artifact_id: ArtifactId | None = None
```

Modes:
- `compose` — analyzer/geometry algorithm has published partial artifacts; the framework runs the Compositor at preview quality using the current/default composition and returns the composed preview artifact.
- `direct_raster` — renderer algorithm provides a direct preview raster; the framework publishes it as the preview artifact.
- `inspector` — no meaningful composition exists yet; show a raw artifact tab preview only, labeled as an intermediate.

**StagedAlgorithm** (fits tone/structure analyzers, renderer wrappers like Floyd-Steinberg, ETF/FDoG, SAED):

```python
class StagedAlgorithm(RenderAlgorithm):
    async def render(self, ctx):
        yield RenderProgress(kind="started")
        self.analyze(ctx)
        for name in self.produced_in_analyze:
            yield RenderProgress(kind="artifact", artifact_kind=name, ...)
        self.synthesize(ctx)
        for name in self.produced_in_synthesize:
            yield RenderProgress(kind="artifact", artifact_kind=name, ...)
        result = self.compose(ctx)
        yield RenderProgress(kind="completed", result=result)

    def analyze(self, ctx): ...
    def synthesize(self, ctx): ...
    def compose(self, ctx) -> RenderResult: ...
```

**IterativeAlgorithm** (fits Pang, Lloyd, electrostatic, DBS, differentiable search):

```python
class IterativeAlgorithm(RenderAlgorithm):
    async def render(self, ctx):
        if ctx.warm_start is not None and self.can_warm_start(ctx.warm_start, ctx.params):
            self.import_warm_state(ctx, ctx.warm_start)
        else:
            self.initialize(ctx)

        for it in range(self.max_iterations(ctx)):
            if ctx.cancel.requested:
                state = self.export_warm_state(ctx)
                yield RenderProgress(
                    kind="cancelled",
                    result=self.finalize(ctx, partial=True, warm_state=state),
                )
                return
            delta = self.step(ctx, it)
            if self.should_stream_preview(it):
                preview = self.build_iteration_preview(ctx, it)
                preview_id = ctx.preview.materialize(preview, ctx)
                yield RenderProgress(
                    kind="iteration", iteration=it,
                    delta=delta, energy=self.current_energy(),
                    preview_artifact_id=preview_id,
                )
            if delta < self.convergence_threshold(ctx):
                break

        yield RenderProgress(
            kind="completed",
            result=self.finalize(ctx, partial=False, warm_state=None),
        )

    # Subclasses implement:
    def initialize(self, ctx) -> None: ...
    def step(self, ctx, iteration: int) -> float: ...
    def current_energy(self) -> float: ...
    def should_stream_preview(self, it: int) -> bool: ...
    def build_iteration_preview(self, ctx, iteration: int) -> IterationPreview: ...
    def finalize(self, ctx, *, partial: bool,
                 warm_state: WarmStartState | None) -> RenderResult: ...

    # Warm-start contract — see §6.5:
    def can_warm_start(self, state: WarmStartState, new_params: ParameterSet) -> bool: ...
    def export_warm_state(self, ctx) -> WarmStartState: ...
    def import_warm_state(self, ctx, state: WarmStartState) -> None: ...
```

**StreamingAlgorithm** (deferred research; sketched for completeness): video frame-by-frame with optical-flow state propagation. Not part of the main roadmap.

### 6.5 Warm-start contract

The scheduler retains the most recent iterative run's state when a slider drag cancels it. If the next run is on the same algorithm/asset and the changed parameters allow it, the new run starts from that state instead of re-initializing.

```python
@dataclass
class WarmStartState:
    algorithm_id: str
    algorithm_version: str
    iteration: int
    energy: float | None
    params: ParameterSet                  # what was running when state was captured
    payload: dict[str, Any]               # algorithm-specific serialized state
```

Algorithm authors implement three methods:

- `export_warm_state(ctx) -> WarmStartState`: serialize current iterative state.
- `import_warm_state(ctx, state) -> None`: restore it.
- `can_warm_start(state, new_params) -> bool`: judge whether resuming is valid given the new parameter set.

Default `can_warm_start` returns True iff no parameter with `recompute_scope=FULL` changed. Algorithms override for finer judgment — Pang annealing can warm-start when `w_t` shifts by 0.05 but not when `ssim_window` changes.

Warm-start state lives in the scheduler's process memory only. Not persisted across restarts; not in cache. The cache holds completed results; warm-start holds in-progress momentum.

### 6.6 RenderResult

The terminal value returned by every algorithm. Different shapes for analyzers vs renderers, but the same type so the framework treats them uniformly.

```python
@dataclass
class RenderResult:
    algorithm_primary_artifact_id: ArtifactId | None = None
    default_composition: Composition | None = None
    final_artifact_id: ArtifactId | None = None     # set by Compositor or direct renderer
    partial: bool = False                           # True only for cancelled / intentionally partial output
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warm_state: WarmStartState | None = None        # carried back to scheduler on cancel
```

By role:

- **ANALYZER** — `algorithm_primary_artifact_id` is the field of headline interest (e.g., `tone_map`); `default_composition` is non-null (an opinionated default the recipe may override); `final_artifact_id` is filled in by the Compositor after `render()` returns.
- **RENDERER** — `algorithm_primary_artifact_id` is the direct raster or vector output; `default_composition` is None; `final_artifact_id == algorithm_primary_artifact_id`; the Compositor is skipped.

---

## 7. Workspace, recipes, presets

```python
@dataclass
class Project:
    id: str
    name: str
    primary_asset_id: str
    created_at: datetime

@dataclass
class Recipe:
    id: str
    project_id: str
    algorithm_id: str
    algorithm_version: str
    params: ParameterSet
    composition: Composition | None       # may be None for renderer algorithms
    name: str | None
    parent_recipe_id: str | None          # fork-from-preset provenance
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Preset:
    id: str
    algorithm_id: str
    algorithm_version_range: str          # semver range, e.g. "^1.0.0"
    params: ParameterSet
    composition: Composition | None
    name: str
    description: str
    tags: list[str]
    thumbnail_artifact_id: ArtifactId | None
```

Recipes are mutable user state. Presets are immutable shareable snapshots (export to JSON for sharing).

---

## 8. Rendering: preview vs. durable runs

The biggest behavioral change from the earlier design: **slider drags produce ephemeral preview runs, not history entries**. A user explicitly pins or exports to create a durable `RenderRun`.

### 8.1 PreviewRun

```python
@dataclass
class PreviewRun:
    id: str
    session_id: str                       # groups slider-drag activity
    project_id: str
    asset_id: str
    algorithm_id: str
    algorithm_version: str
    params: ParameterSet
    composition: Composition | None
    seed: int
    quality: QualityMode                  # "preview" or "draft"
    cache_key: str
    progress: RunProgress
    created_at: datetime
    superseded_at: datetime | None        # set when a later preview in the same session arrived
```

Properties:
- Cancellable; superseded by the next preview in the same session.
- Cached (so re-visiting a parameter snapshot is instant), but cache entries TTL after ~1 hour.
- Not shown in user-visible history by default.
- Eligible to be warm-start source for the next preview.

### 8.2 RenderRun

```python
@dataclass
class RenderRun:
    id: str
    project_id: str
    recipe_id: str | None                 # null if promoted from anonymous preview
    asset_id: str
    algorithm_id: str
    algorithm_version: str
    params: ParameterSet                  # snapshot — recipes are mutable, runs aren't
    composition: Composition | None
    seed: int
    quality: QualityMode                  # typically "full"
    status: RunStatus
    cache_key: str

    started_at: datetime
    completed_at: datetime | None

    progress: RunProgress
    artifact_ids: list[ArtifactId]
    primary_artifact_id: ArtifactId | None          # user-visible final artifact
    error: RenderError | None

    promoted_from_preview_id: str | None  # provenance trail
```

Properties:
- Created when the user clicks **Pin**, **Save**, or **Export**.
- Appears in project history.
- Artifacts kept indefinitely (until project deletion).
- Reproducible bit-identically from `(asset_checksum, algorithm_id, version, params, composition, seed, calibration_checksum)`.

### 8.3 UI flow

```
slider drag
  → PreviewRun (ephemeral)
  → cancellable; superseded by next drag

slider release + user clicks "Pin to history"
  → promote latest PreviewRun → RenderRun (or re-run at full quality if preview was "draft")

user clicks "Save as preset"
  → write Preset from current params + composition

user clicks "Export PNG/SVG"
  → ensure full-quality RenderRun exists; serve its user-visible final artifact
```

### 8.4 Artifact

```python
@dataclass
class Artifact:
    id: ArtifactId
    run_id: str                           # PreviewRun or RenderRun id
    run_kind: Literal["preview", "render"]
    kind: str                             # ArtifactKindDef.name
    type: ArtifactDataType
    substrate_ref: SubstrateRef
    storage_uri: str                      # content-addressed
    format: Literal["png", "exr", "npy", "json", "svg"]
    checksum: str
    size_bytes: int
    iteration: int | None
    metadata: dict[str, Any]
    viewer: ArtifactViewerSpec | None = None
```

### 8.5 Cache

The cache is keyed *per artifact*, not per run:

```
artifact_cache_key = sha256(
    asset_checksum
  | algorithm_id | algorithm_version
  | artifact_kind                          # "tone_map", "edge_mask", "final_raster", etc.
  | canonical_json(params_in_scope)        # only params that affect THIS artifact
  | str(seed)
  | str(quality)
  | calibration_assets_checksum
)
```

`params_in_scope` is computed from the dependency graph: only parameters whose `invalidates` list includes the artifact kind or an upstream-of-it artifact kind. This is the payoff for §5.4 and §5.4.1: changing `ink_color` invalidates `composition` only, so `tone_map` and `edge_mask` hit cache.

Compositor-generated artifacts (`pattern_field_<layer>`, `ink_mask_<layer>`, `composition`, `final_raster`, `final_svg`) extend that base key with their own inputs:

```
compositor_artifact_cache_key = sha256(
    base artifact fields above
  | compositor_version
  | canonical_json(composition_snapshot)
  | canonical_json(layer_spec_in_scope)       # color, role, opacity, blend mode, priority
  | pattern_kind | pattern_impl_version
  | canonical_json(pattern_params_in_scope)
  | canonical_json(pattern_coordinates)
  | output_size
  | source_artifact_checksums                 # density/orientation/warp/mask/field sources
)
```

This prevents the dangerous case where a palette, layer order, blend mode, pattern parameter, or source artifact changes but the final raster reuses an old cache hit. Structural artifacts still stay cheap: the `tone_map` key does not include ink colors or pattern parameters.

Run-level cache_key is the composition of all artifact keys; a "complete run hit" means all artifacts hit.

### 8.6 Scheduler

```python
class RenderScheduler:
    def submit(self, run: PreviewRun | RenderRun) -> RunHandle:
        # 1. Compute per-artifact cache keys. Reuse what hits.
        # 2. If everything hits → mark completed, return.
        # 3. Look up execution_profile.
        # 4. If sub_second and not GPU → run inline, return synchronously.
        # 5. Else enqueue (interactive for previews, batch for renders). Return SSE handle.
```

Two queues:
- **interactive** — preview runs, FIFO, low concurrency cap, eager preemption.
- **batch** — durable render runs at full quality, worker pool, GPU-aware.

Cancellation is cooperative (`ctx.cancel`). When superseded, an iterative preview exports its warm-start state to the scheduler for the next preview.

---

## 9. API surface

### 9.1 REST

```
GET    /algorithms                              # list, filter by family/role/capability
GET    /algorithms/{id}/{version}               # full definition + parameter schema
GET    /pattern_kinds                           # pattern catalog with parameter schemas

POST   /assets                                  # upload raster (v0)
GET    /assets/{id}

POST   /projects
GET    /projects/{id}

POST   /projects/{id}/recipes
PATCH  /recipes/{id}

POST   /preview_runs                            # ephemeral slider-drag render
DELETE /preview_runs/{id}                       # cancel
GET    /preview_runs/{id}
POST   /preview_runs/{id}/promote               # → RenderRun

POST   /render_runs                             # durable, full-quality
GET    /render_runs/{id}
DELETE /render_runs/{id}                        # cancel

GET    /runs/{id}/artifacts                     # works for both kinds
GET    /artifacts/{id}                          # binary or JSON depending on type
GET    /artifacts/{id}/preview                  # thumbnail honoring ArtifactViewerSpec

GET    /presets?algorithm_id=...
POST   /presets                                 # save current params+composition as preset
```

### 9.2 Streaming (SSE)

```
GET /preview_runs/{id}/events    →  text/event-stream
GET /render_runs/{id}/events     →  text/event-stream
```

```
event: iteration
data: {"iteration": 42, "energy": 0.0127, "preview_artifact_id": "art_..."}

event: artifact
data: {"artifact_kind": "tone_map", "artifact_id": "art_..."}

event: completed
data: {"run_id": "...", "primary_artifact_id": "art_...", "final_artifact_id": "art_..."}

event: cancelled
data: {"run_id": "...", "partial": true, "warm_start_available": true}
```

UI flow: debounced param change → POST `/preview_runs` → open SSE → render iteration previews to canvas → swap to final on `completed`. On `cancelled`, keep the last displayed preview but do not promote/export it as final; the scheduler may carry its `warm_state` into the next preview.

### 9.3 Schema-driven UI

The frontend fetches `AlgorithmDefinition` and `PatternKindDef` lists once per version and caches them. Parameter controls and ink-layer pattern panels are generated entirely from `parameters[]` + `Predicate` evaluator. **No per-algorithm UI code.**

Boundary: schema-driven controls stop at parameter panels. Composition editing is product UI, not just schema rendering: adding/removing ink layers, choosing `density_source` / `orientation_source` from `suitable_as`-filtered artifact lists, dragging layer order, previewing blend modes, and managing presets are first-class UI workflows with their own components.

---

## 10. Persistence

| Data                                    | Where                                                            | Notes                          |
| --------------------------------------- | ---------------------------------------------------------------- | ------------------------------ |
| Algorithm + pattern definitions         | Python modules; JSON snapshot served to frontend                 | Versioned with code            |
| Calibration assets                      | Object store (S3) or local disk, content-addressed               | Immutable per version          |
| Projects, recipes, presets              | SQLite (v0) → Postgres (multi-user)                              |                                |
| RenderRun metadata + artifact metadata  | SQLite/Postgres                                                  |                                |
| PreviewRun metadata                     | In-memory + Redis (when multi-user); TTL ~1 hour                 | Lost on restart is fine        |
| Artifact blobs                          | Object store, content-addressed by checksum                      | Deduped across runs            |
| Cache (per-artifact)                    | `artifact_cache_key → ArtifactId` table; blobs in object store   | GC by LRU and TTL              |
| Run progress (live)                     | Redis pub/sub for SSE fan-out                                    | Lost on restart is fine        |
| Warm-start state                        | In-process memory only                                           | Not persisted; not in cache    |

v0 single-user: SQLite + local disk. No Postgres, no Redis until concurrent users force it.

---

## 11. Reproducibility

1. Every `RenderRun` records `(asset_checksum, algorithm_id, algorithm_version, params_snapshot, composition_snapshot, seed, quality, calibration_checksum)`. That tuple re-executes bit-identically.
2. `capabilities.deterministic` is enforced by a CI fixture: re-run, hash output, compare.
3. Non-deterministic algorithms (some GPU reductions) are marked explicitly. UI shows a "non-deterministic" badge.
4. On major version bump, existing `RenderRun`s remain queryable (immutable history) but require explicit migration to re-execute. `RunRebuildRequest(target_version, param_map)` is the affordance.
5. `PreviewRun`s make no reproducibility promise — they exist only to drive interactive UI.

---

## 12. Extensibility — worked example

A Phase 1A analyzer: tone analysis only. Pattern generation lives in the Compositor; the algorithm just produces structural fields and an opinionated default composition.

```python
# colorworks/algorithms/tonal_analyzer.py
from colorworks.algorithms import StagedAlgorithm, registry
from colorworks.domain import (
    AlgorithmDefinition, AlgorithmFamily, AlgorithmRole, AlgorithmCapabilities,
    InputSpec, OutputSpec, ExecutionProfile,
    ParameterDef, ParameterType, ArtifactKindDef, ArtifactViewerSpec,
    Composition, InkLayerSpec, PaletteColor, PatternSpec, PatternCoordinateSpec,
    RenderResult, ScalarField, BinaryMask, Eq,
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
        ParameterDef("contrast", "Tone contrast", ParameterType.FLOAT,
                     default=1.0, min=0.0, max=3.0, step=0.05,
                     group="tone",
                     invalidates=["tone_map"]),
        ParameterDef("midpoint", "Tone midpoint", ParameterType.FLOAT,
                     default=0.5, min=0.0, max=1.0, step=0.01,
                     group="tone",
                     invalidates=["tone_map"]),
        ParameterDef("preserve_edges", "Edge preservation", ParameterType.BOOL,
                     default=True, group="structure",
                     invalidates=["edge_mask"]),
        ParameterDef("edge_threshold", "Edge threshold", ParameterType.FLOAT,
                     default=0.15, min=0.0, max=1.0, step=0.01,
                     group="structure",
                     visible_when=Eq("preserve_edges", True),
                     invalidates=["edge_mask"]),
    ],
    artifact_kinds=[
        ArtifactKindDef(
            name="tone_map", type="scalar_field", label="Tone Map",
            suitable_as=["density_source"],
            viewer=ArtifactViewerSpec(default_view="heatmap", colormap="gray"),
        ),
        ArtifactKindDef(
            name="edge_mask", type="binary_mask", label="Edges",
            suitable_as=["edge_mask", "mask_source"],
            viewer=ArtifactViewerSpec(default_view="mask"),
        ),
    ],
    calibration_assets=[],
    execution_profile=ExecutionProfile(
        typical_runtime="sub_second", is_iterative=False, is_streamable=False,
        is_cancellable=False, parallelism="serial", memory_class="small",
    ),
    capabilities=AlgorithmCapabilities(
        supports_raster_output=False, supports_vector_output=False,
        supports_multi_class=False, supports_interactive_preview=True,
        supports_progressive_refinement=False, deterministic=True, requires_gpu=False,
    ),
)


class TonalAnalyzer(StagedAlgorithm):
    definition = DEFINITION
    produced_in_analyze = ["tone_map", "edge_mask"]
    produced_in_synthesize: list[str] = []

    def analyze(self, ctx):
        gray = to_gray(ctx.input)
        tone = remap_tone(gray, ctx.params["contrast"], ctx.params["midpoint"])
        tone_id = ctx.store.publish(
            "tone_map",
            ScalarField(ctx.input.substrate, tone, "float32"),
        )
        ctx.working.put("tone_id", tone_id)

        if ctx.params["preserve_edges"]:
            edges = sobel_edge_mask(gray, threshold=ctx.params["edge_threshold"])
            ctx.store.publish(
                "edge_mask",
                BinaryMask(ctx.input.substrate, edges),
            )

    def synthesize(self, ctx):
        pass  # no synthesis stage for a pure analyzer

    def compose(self, ctx):
        # Opinionated default composition. The recipe may override.
        # The wave is generated by the Compositor from PatternSpec params —
        # this algorithm has no opinion about wave frequency or phase.
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
                        mask_source="edge_mask" if ctx.params["preserve_edges"] else None,
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


registry.register(TonalAnalyzer())
```

**One-click looks are presets, not algorithms.** "Wave Halftone," "Maze Halftone," "Crosshatch" all use the same `TonalAnalyzer` with different `Composition`s:

```python
# colorworks/presets/wave_halftone.py
WAVE_HALFTONE_PRESET = Preset(
    id="wave_halftone_v1",
    algorithm_id="tonal_analyzer",
    algorithm_version_range="^1.0.0",
    params={"contrast": 1.2, "midpoint": 0.5,
            "preserve_edges": True, "edge_threshold": 0.15},
    composition=Composition(
        paper_color=PaletteColor("#f4ebd9", "paper"),
        layers=[
            InkLayerSpec(
                name="ink",
                color=PaletteColor("#1a1a1a", "ink"),
                role="shadow",
                density_source="tone_map",
                pattern=PatternSpec(
                    kind="wave",
                    params={"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                    mask_source="edge_mask",
                    coordinates=PatternCoordinateSpec(space="image_px"),
                ),
            ),
        ],
    ),
    name="Wave Halftone",
    description="Sinusoidal halftone with edge preservation",
    tags=["halftone", "wave"],
)
```

What the author **did not write**:
- HTTP handlers
- UI controls
- SSE wiring
- Cache keys / dependency invalidation
- Cancellation plumbing
- Artifact storage
- The Compositor itself
- Pattern generation
- History display
- Schema validation

For **iterative algorithms** (Pang, CVT, electrostatic — Phase 3), substitute `IterativeAlgorithm` and implement `step`, `current_energy`, `build_iteration_preview`, `finalize`, `can_warm_start`, `export_warm_state`, `import_warm_state`. The framework handles everything else identically.

For **renderer algorithms** (Pang halftone, Floyd-Steinberg — direct binary output), set `role=RENDERER`, leave `default_composition=None` in the `RenderResult`, and publish the final raster as the primary artifact. The Compositor is skipped.

---

## 13. Frontend layout

A short note on the UI side because it constrains a few backend decisions.

```
┌──────────────┬────────────────────────────────────┬──────────────────┐
│              │                                    │                  │
│  Left rail   │       Canvas (main view)           │   Right rail     │
│              │                                    │                  │
│  - asset     │  - source / output split           │  - algorithm     │
│  - algorithm │  - zoom, pan                       │    params        │
│  - presets   │  - artifact selector (tabs)        │    (grouped)     │
│  - history   │  - non-destructive overlay         │  - ink layers    │
│              │                                    │    panel         │
│              │                                    │  - per-layer     │
│              │                                    │    pattern       │
│              │                                    │    params        │
│              │                                    │  - palette       │
│              │                                    │                  │
└──────────────┴────────────────────────────────────┴──────────────────┘
                          ↓
                  Inspector strip:
                  [Source] [Tone Map] [Pattern Field] [Edges] [Composition]
                  ↑ tabs auto-populated from ArtifactStore + ArtifactViewerSpec
```

Two interaction modes:
- **Auto preview** for algorithms with `typical_runtime="sub_second"` — slider drag → live update.
- **Manual re-render** for iterative algorithms — slider drag updates params; user clicks Render or it auto-fires on release with a short debounce.

---

## 14. Phasing — the binding plan

Build in this order. Each phase ends with a usable product, not just plumbing. Every phase must leave behind agent-verifiable evidence: tests, a smoke path, and a short note on what was intentionally deferred.

The binding roadmap stops at Phase 5. Phase 4 completes the core image-algorithm expansion; Phase 5 is product finish, real-image validation, preset tuning, export hardening, and usability. There is no Phase 6/7 in the main plan.

Research map: Phase 2 is motivated by `RESEARCH.md` §4.1–§4.2, Phase 3 by §2.2 and §3.1, and Phase 4 by §2.1–§2.3 and §5. Phase 5 is driven by product evidence, not a new research family. Phase 0–1B are product-loop scaffolding plus cheap procedural patterns.

Agent handoff template for every phase:

- **Deliverables:** concrete files/features the coding agent is expected to add.
- **Acceptance checks:** commands or UI/API flows another agent can run locally.
- **Evidence:** checksums, screenshots, timings, exported files, or test names to report.
- **Deferred:** explicit non-goals so later phases do not mistake omissions for bugs.

### Phase 0 — Prove the loop (1 week)

Deliverables:
- One self-contained renderer: **ordered dither** (Bayer matrix).
- Local web UI with raster upload, source/output preview, controls for matrix size, threshold, and contrast.
- Sync render API only. No DB, no SSE, no auth, no compositor, no ink layers.
- JSON recipe save/load on disk.
- PNG export of the displayed output.
- Focused package/tests around Bayer rendering and recipe round-trip.

Acceptance checks:
- Load one local raster from disk.
- Adjust Bayer matrix size, threshold, and contrast.
- Render a 1MP image in under 200 ms on a typical laptop.
- Save recipe JSON and reload it to reproduce the same output checksum.
- Export a PNG matching the displayed canvas.
- Run unit tests for renderer and recipe serialization.

Evidence to report:
- Test command and pass/fail count.
- Render timing for a roughly 1MP image.
- Original render checksum, reloaded-recipe render checksum, and exported PNG checksum.
- Local URL or command used to launch the tool.

Deferred:
- Algorithm catalog, compositor, ink layers, artifact DAG, SSE, background workers.

### Phase 1A — Tone analyzer + Compositor with one pattern (1–1.5 weeks)

The architecture-proving milestone. **One analyzer, one pattern kind, one ink layer composition.**

Deliverables:
- `TonalAnalyzer` publishing `tone_map` and optional `edge_mask`.
- Minimal `AlgorithmDefinition`, `PatternKindDef`, `ParameterDef`, `WorkingSet`, `ArtifactStore`, and `ArtifactViewerSpec` sufficient for one analyzer.
- One compositor-owned pattern kind: **wave**.
- One-layer `Composition` with paper + ink, density source, optional mask source, and `PatternCoordinateSpec`.
- Schema-generated controls for analyzer params and wave pattern params.
- Inspector tabs: source, tone map, edge mask, final.

Acceptance checks:
- Rendering `TonalAnalyzer` with the default wave composition produces `tone_map`, `edge_mask`, and `final_raster` artifacts.
- Changing ink color or wave params changes only compositor/final cache keys; `tone_map` checksum remains unchanged.
- Changing contrast or midpoint changes `tone_map` and downstream final output.
- A saved preset/recipe reloads to the same final checksum.
- Adding a second procedural pattern in a test fixture requires no algorithm code changes.

Evidence to report:
- Artifact IDs/checksums for source, tone map, edge mask, final.
- Cache-hit/miss log for palette-only vs tone-param changes.
- UI screenshot or browser smoke output showing inspector tabs.

Deferred:
- Multi-layer editing polish, vector output, iterative previews, background queues.

### Phase 1B — Pattern catalog + a renderer (1 week)

Deliverables:
- Pattern kinds: **maze**, **blue_noise**, **ordered_dither**, **hatch**.
- One direct renderer algorithm: **Floyd-Steinberg** (`HALFTONING` family, `RENDERER` role), bypassing the Compositor.
- File-based per-artifact cache implementing the §8.5 key rules.
- Multi-layer composition support: paper + at least two ink layers.
- Right-rail layer UI for add/remove/reorder, color, blend mode, density source, and per-layer pattern params.
- Preset CRUD with composition snapshots.
- Built-in presets: "Wave Halftone," "Maze Halftone," "Hatch."

Acceptance checks:
- One uploaded raster can render the three presets from the same cached `tone_map`.
- Palette-only or layer-order changes re-render final output without re-running `TonalAnalyzer`.
- Floyd-Steinberg produces a final raster with no compositor artifacts.
- Multi-layer composition order changes the output checksum predictably.
- Preset save/load round-trips params, layer order, colors, pattern params, and source refs.

Evidence to report:
- Cache logs proving analyzer reuse across presets.
- Checksums for wave/maze/hatch outputs from the same source/tone map.
- Test proving renderer role skips compositor.

Deferred:
- Orientation fields, vector/SVG export, iterative algorithms.

### Phase 2 — Orientation-aware + vector output (2–3 weeks)

Deliverables:
- `StructureAnalyzer` publishing structure tensor and ETF/orientation `VectorField2D` artifacts.
- Orientation-driven **hatch** and **crosshatch** pattern kinds consuming `orientation_source`.
- Vector-field viewers: at least HSV angle view and downsampled glyph view; LIC may be optional behind a flag.
- First-class `Polyline`, `Stroke`, and `StrokeSet` types.
- SVG export for hatch/crosshatch output.

Acceptance checks:
- A source image produces an orientation artifact visible in the inspector.
- Hatch orientation changes when `orientation_source` is swapped or disabled.
- SVG export opens as valid XML and contains expected path/stroke elements.
- Raster preview and SVG export agree on layer count, colors, and output bounds.
- Missing or wrong-type `orientation_source` is rejected with a user-safe validation error.

Evidence to report:
- Screenshot/browser output for orientation viewer and hatch output.
- SVG file path, size, and validation command.
- Tests covering vector-field type checks and SVG bounds.

Deferred:
- Lloyd/Pang iteration, mesh cross-fields, temporal coherence.

### Phase 3 — Iteration + streaming (2–3 weeks)

Deliverables:
- `IterativeAlgorithm`, `RenderProgress`, `IterationPreview`, and `PreviewCompositor`.
- SSE event endpoints for preview/render runs.
- `PreviewRun` cancellation, supersession, and `RenderRun` promotion.
- Warm-start state for at least one iterative algorithm.
- **CVT stippling** (Lloyd relaxation) and a first **Pang-style structure-aware halftoning** implementation.
- In-process asyncio background worker.

Acceptance checks:
- Iterative run emits `started`, multiple `iteration`, and terminal `completed` events over SSE.
- Cancelling emits `cancelled`, does not create history, and makes warm-start state available.
- A subsequent compatible parameter change warm-starts from prior state and reaches first preview faster than cold start.
- `IterationPreview(mode=\"compose\")` produces a composed ink-layer preview from partial artifacts.
- Promoting a completed preview creates a durable `RenderRun` whose artifacts survive restart.

Evidence to report:
- SSE transcript for completed and cancelled runs.
- Timing comparison: cold first preview vs warm-start first preview.
- Test proving cancelled partial output cannot be exported as final.
- Stored `RenderRun` metadata/artifact listing after restart.

Deferred:
- Redis/RQ, multi-user scheduling, GPU slots.

### Phase 4 — Advanced image halftoning quality (long)

Deliverables:
- **DBS** as a deterministic quality-reference renderer with immutable HVS calibration assets.
- **SAED** as the final core halftoning renderer, consuming tone/orientation structure without adding a new substrate.
- Calibration metadata persisted in every durable run that uses calibration assets.
- Algorithm-specific guardrails for expensive renderers: input-size limits, iteration limits, clear 4xx errors, and UI disablement where needed.
- A quality comparison harness covering the core output families: Floyd-Steinberg, Pang, CVT stippling, DBS, and SAED.
- Explicit closure note: after SAED, do not add another main-roadmap algorithm family unless Phase 5 real-image testing proves a concrete product gap.

Acceptance checks:
- Calibration asset checksum/version is recorded in every run using it.
- Replacing/re-cooking a calibration asset requires either algorithm version bump or explicit `calibration_version`.
- DBS reference output is deterministic under fixed seed and small fixture input.
- SAED output is deterministic under fixed seed, reacts to orientation structure on a directional fixture, and rejects unsupported inputs with a user-safe validation error.
- Expensive renderers stay within documented CPU/time guardrails on fixture inputs.
- The UI/API routes render DBS and SAED through the same preview/render-run paths users will exercise.
- The comparison harness produces nonblank outputs and stable evidence for the selected real-image fixtures.

Evidence to report:
- Run snapshot showing calibration checksum/version.
- Determinism hashes for DBS and SAED fixtures.
- Quality comparison fixture: Floyd-Steinberg/Pang/CVT/DBS/SAED outputs with metric table.
- Browser or API smoke proof showing user-facing DBS/SAED behavior.
- Guardrail test names and error payload examples.

Deferred:
- Electrostatic halftoning, neural analyzer hooks, GPU scheduling, mesh substrate, video, full plugin system, production multi-tenant GPU cluster.

### Phase 5 — Product finish + real-image validation

Deliverables:
- Curated real-image fixture set: portrait, landscape, line art, noisy scan, high-contrast graphic, low-contrast photo, and small icon/illustration.
- Tuned built-in presets for the core algorithms and common image types.
- Side-by-side comparison UI for source, selected artifacts, and multiple algorithm outputs.
- Export hardening for final PNG and SVG outputs: dimensions, color metadata where available, filenames, and repeatable checksums.
- Batch or gallery workflow for trying several presets against the same source without losing context.
- Usability pass over controls, run status, cancellation, errors, empty states, and inspector navigation.
- User-facing examples/docs generated from the fixture gallery.

Acceptance checks:
- Every fixture renders through the selected core presets without crashes, blank outputs, or stale inspector state.
- At least one fixture exercises each output path: compositor raster, direct renderer raster, iterative renderer, and SVG export.
- Exported PNG/SVG files match the displayed output bounds, layer/color expectations, and repeatable checksum rules.
- Switching algorithms or presets never leaves stale SSE progress, stale artifacts, or hidden controls in the UI.
- Guardrails for oversized inputs are clear, actionable, and covered by tests.
- A browser-backed smoke test captures the comparison/gallery workflow on desktop and a narrow viewport.

Evidence to report:
- Fixture list, fixture dimensions, and selected presets.
- Output gallery paths/checksums for each fixture/preset pair.
- Browser screenshots or Playwright evidence for the comparison/gallery workflow.
- Exported PNG/SVG paths and validation commands.
- Performance table for representative fixture renders.

Deferred:
- New algorithm families, mesh substrate, video, neural/GPU research, cloud rendering, collaboration, plugin marketplace.

### Deferred research appendix — not the product roadmap

These topics can stay in the type system and research notes as sketches, but they are not Phase 6/7 and should not be prompted to a coding agent as normal next work. Promote one only with a separate product decision and a new acceptance plan.

- `MeshSurface`, N-RoSy cross-fields, TAM surface hatching, Three.js mesh viewers.
- `VideoVolume`, `StreamingAlgorithm`, optical-flow state propagation, frame scrubbers, video export.
- Neural analyzer hooks, learned ETF, differentiable algorithms, GPU scheduling.
- Electrostatic halftoning beyond a bounded product need.
- User-loaded compositor plugins or a plugin marketplace.

---

## 15. Deliberately out of scope

- **Multi-tenancy / auth.** Single-user assumption through the product-finish roadmap. Add at the API edge later; no domain changes needed.
- **Node-graph editor.** Atomic algorithms + composition is enough. If you want "ETF → SAED-with-orientation," register it as a new composite algorithm, not a generic DAG. Reconsider only after ≥3 such combinations.
- **Print device profiles.** Add `device_profile: DeviceProfile | None` to `RenderContext` when DBS-class device modeling matters. No domain restructure.
- **Collaboration / sharing.** Recipes export to JSON. That's enough until users ask.
- **A second user-extensible Compositor.** The product ships one built-in compositor. Custom compositors as plugins are a deferred research/plugin question.
- **Mesh, video, live camera, and cloud rendering.** These are separate product directions, not follow-on phases after Phase 5.

---

## 16. Open questions

Worth tracking; don't block Phase 0–1B on them.

1. **Composite algorithms vs. runtime composition.** Default: register as a new algorithm. Revisit when there are 3+ similar combinations.
2. **Multi-class output shape.** `PointSet[]` with shared substrate, or a `MultiClassPointSet` with explicit inter-class coupling metadata? The latter is more honest about Wasserstein-barycenter / multi-class stippling but adds a type. Defer until multi-class stippling actually lands.
3. **Differentiable algorithms.** First-class capability (so optimizers can compose) vs. wrapped inside specific algorithms. Deferred research decision.
4. **Online video state-passing.** `StreamingAlgorithm` is sketched; the state-passing protocol for mid-stream parameter changes is deferred unless video becomes a separate product direction.
5. **Plugin pattern kinds.** Should `PatternKindDef` be code-only (compiled into the binary) or loaded from a user directory? Code-only is simpler; user-defined patterns is a real eventual want after the core product is stable.
6. **Hybrid algorithms.** A renderer can already publish intermediate artifacts. Reintroduce `role=HYBRID` only when there is a concrete algorithm that both needs automatic composition and has a distinct direct final output.

---

## 17. Type glossary

Names used above but not fully defined in the design snippets:

| Type | Meaning |
| ---- | ------- |
| `MediaAsset` | Uploaded source media plus checksum, metadata, and substrate reference; v0 only stores raster images. |
| `RasterImage` | Pixel buffer plus `RasterGrid`; used for source rasters, previews, and direct renderer outputs. |
| `ArtifactId` | Stable ID returned by `ArtifactStore.publish`; resolves to persisted artifact metadata and blob storage. |
| `ArtifactDataType` | Runtime kind tag such as `raster_image`, `scalar_field`, `vector_field_2d`, `binary_mask`, `point_set`, `stroke_set`, `svg`, `json`. |
| `ParameterSet` | Canonical JSON-compatible dict of parameter keys to validated values. |
| `ParameterValue` | JSON scalar/list/object accepted by `ParameterDef.type`. |
| `QualityMode` | `"preview"`, `"draft"`, or `"full"`; affects resolution, iteration budget, and cache key. |
| `RunProgress` | Persisted aggregate of the latest progress event, percent estimate, status, and timestamps. |
| `RunStatus` | `queued`, `running`, `completed`, `cancelled`, or `failed`. |
| `RenderError` | User-safe error code/message plus internal diagnostic reference. |
| `CalibrationAssetRef` | Reference from an algorithm definition to an immutable calibration asset and checksum. |
| `PreviewCompositor` | Framework service that turns an `IterationPreview` into a preview artifact, composing partial fields when possible. |

---

## 18. Theses, restated

1. **Ink layers, not pixels.** The product talks in terms of paper + inks + patterns + tone references, not in terms of generic image processing. `InkLayerSpec` + `PatternSpec` + `Composition` are first-class; the Compositor is the final stage of every render.
2. **Iteration is first-class.** `async render → AsyncIterator[RenderProgress]` is the contract. `StagedAlgorithm` and `IterativeAlgorithm` are opt-in helpers, not the protocol. Warm starts, cancellation, and SSE streaming are uniform mechanisms across families.

Everything else — bounded contexts, the WorkingSet / ArtifactStore split, viewer specs, per-artifact cache keys, preview-vs-render runs, the dependency-keyed invalidation model — exists to make those two theses cheap to honor as the algorithm zoo grows.
