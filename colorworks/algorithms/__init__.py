from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from PIL import Image

from colorworks.domain import (
    AlgorithmDefinition,
    PatternKindDef,
    Composition,
    WorkingSet,
    ArtifactStore,
    RenderResult,
    Substrate,
    RasterGrid,
    CancelToken,
    WarmStartState,
    IterationPreview,
)

@dataclass(frozen=True)
class MediaAsset:
    id: str
    checksum: str
    image: Image.Image
    substrate: Substrate

@dataclass
class RenderContext:
    input: MediaAsset
    params: dict[str, Any]
    composition: Composition | None
    seed: int
    quality: str = "full"
    working: WorkingSet = field(default_factory=WorkingSet)
    store: ArtifactStore = field(default_factory=ArtifactStore)
    cancel: CancelToken = field(default_factory=CancelToken)
    warm_start: WarmStartState | None = None
    rng: Any = None  # np.random.Generator, lazy-initialised

    def __post_init__(self) -> None:
        if self.rng is None:
            import numpy as np
            self.rng = np.random.default_rng(self.seed)

@dataclass
class RenderProgress:
    kind: str  # "started", "artifact", "iteration", "completed", "cancelled", "failed"
    artifact_kind: str | None = None
    artifact_id: str | None = None
    result: RenderResult | None = None
    iteration: int | None = None
    total_iterations: int | None = None
    delta: float | None = None
    energy: float | None = None
    preview_artifact_id: str | None = None


class StagedAlgorithm:
    definition: AlgorithmDefinition
    produced_in_analyze: list[str] = []
    produced_in_synthesize: list[str] = []

    async def render(self, ctx: RenderContext) -> AsyncIterator[RenderProgress]:
        yield RenderProgress(kind="started")
        self.analyze(ctx)
        for name in self.produced_in_analyze:
            try:
                art = ctx.store.get_by_name(name)
                yield RenderProgress(kind="artifact", artifact_kind=name, artifact_id=art.id)
            except KeyError:
                pass
        self.synthesize(ctx)
        for name in self.produced_in_synthesize:
            try:
                art = ctx.store.get_by_name(name)
                yield RenderProgress(kind="artifact", artifact_kind=name, artifact_id=art.id)
            except KeyError:
                pass
        result = self.compose(ctx)
        yield RenderProgress(kind="completed", result=result)

    def analyze(self, ctx: RenderContext) -> None:
        pass

    def synthesize(self, ctx: RenderContext) -> None:
        pass

    def compose(self, ctx: RenderContext) -> RenderResult:
        raise NotImplementedError()

    def is_artifact_enabled(self, name: str, params: dict[str, Any]) -> bool:
        return True

    def load_from_cache(self, ctx: RenderContext, artifacts: dict[str, str]) -> None:
        pass


class IterativeAlgorithm:
    """Base class for iterative algorithms (Lloyd, Pang, etc.)."""
    definition: AlgorithmDefinition

    async def render(self, ctx: RenderContext) -> AsyncIterator[RenderProgress]:
        yield RenderProgress(kind="started")

        if ctx.warm_start is not None and self.can_warm_start(ctx.warm_start, ctx.params):
            self.import_warm_state(ctx, ctx.warm_start)
        else:
            self.initialize(ctx)

        max_iters = self.max_iterations(ctx)
        for it in range(max_iters):
            if ctx.cancel.requested:
                warm_state = self.export_warm_state(ctx)
                result = self.finalize(ctx, partial=True, warm_state=warm_state)
                yield RenderProgress(kind="cancelled", result=result)
                return

            delta = self.step(ctx, it)

            if self.should_stream_preview(it):
                preview = self.build_iteration_preview(ctx, it)
                preview_id = _materialize_preview(preview, ctx)
                yield RenderProgress(
                    kind="iteration",
                    iteration=it,
                    total_iterations=max_iters,
                    delta=delta,
                    energy=self.current_energy(),
                    preview_artifact_id=preview_id,
                )

            if delta < self.convergence_threshold(ctx):
                break

        result = self.finalize(ctx, partial=False, warm_state=None)
        yield RenderProgress(kind="completed", result=result)

    # ── Subclasses must implement ─────────────────────────────────────────────
    def initialize(self, ctx: RenderContext) -> None:
        raise NotImplementedError()

    def step(self, ctx: RenderContext, iteration: int) -> float:
        raise NotImplementedError()

    def current_energy(self) -> float:
        return float("inf")

    def should_stream_preview(self, it: int) -> bool:
        return it % 5 == 4

    def build_iteration_preview(self, ctx: RenderContext, iteration: int) -> IterationPreview:
        return IterationPreview(mode="inspector")

    def finalize(self, ctx: RenderContext, *, partial: bool,
                 warm_state: WarmStartState | None) -> RenderResult:
        raise NotImplementedError()

    def max_iterations(self, ctx: RenderContext) -> int:
        return int(ctx.params.get("max_iterations", 30))

    def convergence_threshold(self, ctx: RenderContext) -> float:
        return float(ctx.params.get("convergence_threshold", 0.5))

    # ── Warm-start contract ───────────────────────────────────────────────────
    def can_warm_start(self, state: WarmStartState, new_params: dict[str, Any]) -> bool:
        return state.algorithm_id == self.definition.id

    def export_warm_state(self, ctx: RenderContext) -> WarmStartState:
        return WarmStartState(
            algorithm_id=self.definition.id,
            algorithm_version=self.definition.version,
            iteration=0,
            energy=self.current_energy(),
            params=dict(ctx.params),
        )

    def import_warm_state(self, ctx: RenderContext, state: WarmStartState) -> None:
        pass


def _materialize_preview(preview: IterationPreview, ctx: RenderContext) -> str | None:
    """Publish an iteration preview to ctx.store; return artifact_id or None."""
    if preview.mode == "direct_raster" and preview.direct_raster is not None:
        return ctx.store.publish("iteration_preview", preview.direct_raster)
    if preview.mode == "compose" and ctx.composition is not None:
        try:
            from colorworks.compositor import Compositor
            comp = Compositor(ctx.store)
            w = ctx.input.substrate.width
            h = ctx.input.substrate.height
            img = comp.composite(ctx.composition, w, h, ctx.seed)
            return ctx.store.publish("iteration_preview", img)
        except Exception:
            return None
    if preview.mode == "inspector" and preview.inspector_artifact_id is not None:
        return preview.inspector_artifact_id
    return None


class PreviewCompositor:
    """Framework service: materialise an IterationPreview into a preview artifact."""

    def materialize(self, preview: IterationPreview, ctx: RenderContext) -> str | None:
        return _materialize_preview(preview, ctx)


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._algorithms: dict[str, Any] = {}
        self._patterns: dict[str, PatternKindDef] = {}
        self._pattern_generators: dict[str, Any] = {}

    def register(self, algorithm: Any) -> None:
        self._algorithms[algorithm.definition.id] = algorithm

    def get(self, algorithm_id: str) -> Any:
        if algorithm_id not in self._algorithms:
            raise KeyError(f"Algorithm {algorithm_id} not registered")
        return self._algorithms[algorithm_id]

    def list_algorithms(self) -> list[Any]:
        return list(self._algorithms.values())

    def register_pattern(self, pattern: PatternKindDef) -> None:
        self._patterns[pattern.kind] = pattern

    def get_pattern(self, kind: str) -> PatternKindDef:
        if kind not in self._patterns:
            raise KeyError(f"Pattern kind {kind} not registered")
        return self._patterns[kind]

    def list_patterns(self) -> list[PatternKindDef]:
        return list(self._patterns.values())

    def register_pattern_generator(self, kind: str, generator_fn: Any) -> None:
        self._pattern_generators[kind] = generator_fn

    def get_pattern_generator(self, kind: str) -> Any:
        return self._pattern_generators.get(kind)

    def unregister_pattern(self, kind: str) -> None:
        if kind in self._patterns:
            del self._patterns[kind]
        if kind in self._pattern_generators:
            del self._pattern_generators[kind]

    def save_state(self) -> dict[str, Any]:
        return {
            "algorithms": dict(self._algorithms),
            "patterns": dict(self._patterns),
            "pattern_generators": dict(self._pattern_generators),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._algorithms = dict(state["algorithms"])
        self._patterns = dict(state["patterns"])
        self._pattern_generators = dict(state["pattern_generators"])

registry = AlgorithmRegistry()
