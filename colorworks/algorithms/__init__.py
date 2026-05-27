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

@dataclass
class RenderProgress:
    kind: str  # "started", "artifact", "completed", "cancelled", "failed"
    artifact_kind: str | None = None
    artifact_id: str | None = None
    result: RenderResult | None = None
    iteration: int | None = None
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
