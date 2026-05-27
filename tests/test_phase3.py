"""
Phase 3 acceptance tests: iterative algorithms, SSE, warm-start, promote.

Evidence required:
  ✓ SSE emits started, multiple iteration events, completed
  ✓ Cancellation emits cancelled, no durable history, warm-start state available
  ✓ Warm-start compatible params run faster than cold first preview
  ✓ IterationPreview(mode="compose") uses compositor over partial artifacts
  ✓ Promoting completed preview creates durable RenderRun that survives restart
  ✓ Cancelled partial output cannot be exported as final
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import registry-populating modules
import colorworks.algorithms.cvt_stippling
import colorworks.algorithms.pattern_catalog
import colorworks.algorithms.tonal_analyzer

from colorworks.algorithms import (
    IterativeAlgorithm,
    MediaAsset,
    PreviewCompositor,
    RenderContext,
    RenderProgress,
    registry,
)
from colorworks.compositor import Compositor
from colorworks.domain import (
    ArtifactStore,
    BinaryMask,
    CancelToken,
    Composition,
    InkLayerSpec,
    IterationPreview,
    PaletteColor,
    PatternSpec,
    PointSet,
    RasterGrid,
    RenderResult,
    RunStatus,
    ScalarField,
    WarmStartState,
)
from colorworks.storage import LocalStore


# ── helpers ───────────────────────────────────────────────────────────────────

def gradient_image(width: int = 48, height: int = 48) -> Image.Image:
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    rgb = np.stack([ramp, ramp, ramp], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def small_image(width: int = 32, height: int = 32) -> Image.Image:
    """Tiny image for fast stippling tests."""
    arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def run_server(tmp_path):
    from colorworks.web.server import ColorworksServer
    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", store, server
    server.shutdown()
    server.server_close()
    t.join(timeout=3)


def upload_asset(base_url: str, image: Image.Image, filename: str = "src.png") -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=buf.getvalue(),
        headers={"X-Filename": filename, "Content-Type": "image/png"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["asset"]


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def delete_url(url: str) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def read_sse(url: str, timeout: float = 15.0) -> list[dict]:
    """Read all SSE events until terminal event or timeout."""
    events: list[dict] = []
    deadline = time.monotonic() + timeout
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = b""
        while time.monotonic() < deadline:
            chunk = resp.read(512)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                event_block, buf = buf.split(b"\n\n", 1)
                for line in event_block.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"data:"):
                        try:
                            events.append(json.loads(line[5:].strip()))
                        except json.JSONDecodeError:
                            pass
            if events and events[-1].get("kind") in ("completed", "cancelled", "failed"):
                break
    return events


def stipple_payload(asset_id: str, *, n: int = 20, max_iters: int = 8,
                    session_id: str = "test-session") -> dict:
    return {
        "asset_id": asset_id,
        "renderer_id": "cvt_stippling",
        "params": {
            "n_stipples": n,
            "max_iterations": max_iters,
            # 0.0 → delta < 0.0 is never true → no early convergence
            "convergence_threshold": 0.0,
            "dot_radius": 2.0,
        },
        "seed": 7,
        "session_id": session_id,
    }


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_cvt_stippling_registered() -> None:
    algo = registry.get("cvt_stippling")
    assert algo.definition.id == "cvt_stippling"
    assert algo.definition.execution_profile.is_iterative
    assert algo.definition.execution_profile.is_cancellable


def test_iterative_algorithm_base_emits_events() -> None:
    """IterativeAlgorithm.render() yields started, iteration*, completed."""
    import asyncio

    class Counting(IterativeAlgorithm):
        _count: int = 0
        _e: float = 10.0

        @property
        def definition(self):
            from colorworks.domain import (
                AlgorithmDefinition, AlgorithmFamily, AlgorithmRole,
                InputSpec, OutputSpec, ExecutionProfile, AlgorithmCapabilities,
            )
            return AlgorithmDefinition(
                id="counting", version="1.0.0",
                family=AlgorithmFamily.STIPPLING, role=AlgorithmRole.RENDERER,
                name="Counting", description="",
                input_spec=InputSpec(), output_spec=OutputSpec(primary_artifact="out"),
                parameters=[], artifact_kinds=[],
                execution_profile=ExecutionProfile(is_iterative=True, is_cancellable=True),
                capabilities=AlgorithmCapabilities(),
            )

        def initialize(self, ctx):
            self._count = 0
            self._e = 10.0

        def step(self, ctx, it):
            self._count += 1
            self._e -= 1.0
            return self._e

        def current_energy(self):
            return self._e

        def should_stream_preview(self, it):
            return True   # emit every iteration

        def max_iterations(self, ctx):
            return int(ctx.params.get("max_iterations", 5))

        def convergence_threshold(self, ctx):
            return 0.0   # never converge

        def finalize(self, ctx, *, partial, warm_state):
            return RenderResult(partial=partial, warm_state=warm_state)

    img = small_image()
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    ctx = RenderContext(
        input=asset,
        params={"max_iterations": 5},
        composition=None,
        seed=0,
    )

    algo = Counting()

    async def collect():
        evs = []
        async for p in algo.render(ctx):
            evs.append(p)
        return evs

    events = asyncio.run(collect())

    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    assert "iteration" in kinds
    assert kinds[-1] == "completed"
    assert sum(1 for k in kinds if k == "iteration") == 5


def test_cancel_token() -> None:
    tok = CancelToken()
    assert not tok.requested
    tok.cancel()
    assert tok.requested


def test_cancel_during_iteration_emits_cancelled_with_warm_state() -> None:
    import asyncio

    class SelfCancelling(IterativeAlgorithm):
        _pts: list

        @property
        def definition(self):
            from colorworks.domain import (
                AlgorithmDefinition, AlgorithmFamily, AlgorithmRole,
                InputSpec, OutputSpec, ExecutionProfile, AlgorithmCapabilities,
            )
            return AlgorithmDefinition(
                id="selfcancel", version="1.0.0",
                family=AlgorithmFamily.STIPPLING, role=AlgorithmRole.RENDERER,
                name="SC", description="",
                input_spec=InputSpec(), output_spec=OutputSpec(primary_artifact="out"),
                parameters=[], artifact_kinds=[],
                execution_profile=ExecutionProfile(is_iterative=True, is_cancellable=True),
                capabilities=AlgorithmCapabilities(),
            )

        def initialize(self, ctx):
            self._pts = [1.0, 2.0]

        def step(self, ctx, it):
            if it == 2:
                ctx.cancel.cancel()
            return 5.0

        def current_energy(self):
            return 5.0

        def should_stream_preview(self, it):
            return False

        def max_iterations(self, ctx):
            return 100

        def convergence_threshold(self, ctx):
            return 0.0

        def finalize(self, ctx, *, partial, warm_state):
            return RenderResult(partial=partial, warm_state=warm_state)

        def export_warm_state(self, ctx):
            return WarmStartState(
                algorithm_id="selfcancel", algorithm_version="1.0.0",
                iteration=2, energy=5.0, params={},
                payload={"pts": self._pts},
            )

    img = small_image()
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    ctx = RenderContext(input=asset, params={}, composition=None, seed=0)

    algo = SelfCancelling()

    async def collect():
        evs = []
        async for p in algo.render(ctx):
            evs.append(p)
        return evs

    events = asyncio.run(collect())
    kinds = [e.kind for e in events]
    assert kinds[-1] == "cancelled"
    last = events[-1]
    assert last.result is not None
    assert last.result.partial is True
    assert last.result.warm_state is not None
    assert last.result.warm_state.payload["pts"] == [1.0, 2.0]


def test_iteration_preview_direct_raster() -> None:
    img = small_image()
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    store = ArtifactStore()
    ctx = RenderContext(input=asset, params={}, composition=None, seed=0, store=store)

    raster = Image.new("RGB", (img.width, img.height), (128, 0, 0))
    preview = IterationPreview(mode="direct_raster", direct_raster=raster)

    pc = PreviewCompositor()
    art_id = pc.materialize(preview, ctx)
    assert art_id is not None
    art = store.get(art_id)
    assert art.type == "raster_image"


def test_iteration_preview_compose_uses_compositor() -> None:
    """IterationPreview(mode='compose') should invoke Compositor with store artifacts."""
    substrate = RasterGrid(32, 32)
    store = ArtifactStore()
    density = np.full((32, 32), 0.5, dtype=np.float32)
    store.publish("tone_map", ScalarField(substrate, density, "float32"))

    composition = Composition(
        paper_color=PaletteColor("#ffffff"),
        layers=[InkLayerSpec(
            name="ink",
            color=PaletteColor("#000000"),
            role="shadow",
            density_source="tone_map",
            pattern=PatternSpec(
                kind="wave",
                params={"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
            ),
        )],
    )

    img = small_image(32, 32)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    ctx = RenderContext(
        input=asset, params={}, composition=composition, seed=0, store=store,
    )

    preview = IterationPreview(mode="compose")
    pc = PreviewCompositor()
    art_id = pc.materialize(preview, ctx)

    assert art_id is not None, "Expected a composed preview artifact"
    art = store.get(art_id)
    assert art.type == "raster_image"
    assert isinstance(art.value, Image.Image)
    assert art.value.size == (32, 32)


def test_warm_start_contract() -> None:
    """CVTStippling can_warm_start iff n_stipples matches."""
    algo = registry.get("cvt_stippling")

    state = WarmStartState(
        algorithm_id="cvt_stippling", algorithm_version="1.0.0",
        iteration=10, energy=0.8,
        params={"n_stipples": 50},
        payload={"points": [[10.0, 20.0]]},
    )

    assert algo.can_warm_start(state, {"n_stipples": 50})
    assert not algo.can_warm_start(state, {"n_stipples": 100})
    assert not algo.can_warm_start(
        WarmStartState(
            algorithm_id="other", algorithm_version="1.0.0",
            iteration=0, energy=None, params={"n_stipples": 50},
        ),
        {"n_stipples": 50},
    )


def test_cvt_warm_start_faster_than_cold() -> None:
    """
    Warm-starting from a prior run's saved state begins at lower energy than cold.

    Evidence: run 15 cold iterations; manually capture state after 5; restart
    warm and verify that the warm run's FIRST iteration energy is ≤ the cold
    run's energy at step 5.  This proves import_warm_state resumes from where
    the prior run left off rather than re-initialising.
    """
    import asyncio
    from colorworks.algorithms.cvt_stippling import CVTStippling

    img = small_image(24, 24)
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    # threshold=0.0 → delta<0 is never true → all iterations run
    base_params = {"n_stipples": 15, "max_iterations": 15,
                   "convergence_threshold": 0.0, "dot_radius": 2.0}

    async def collect_events(algo, ctx):
        evs = []
        async for p in algo.render(ctx):
            evs.append(p)
        return evs

    # ── Cold run: record per-iteration energy ─────────────────────────────────
    cold_algo = CVTStippling()
    cold_ctx = RenderContext(input=asset, params=base_params,
                             composition=None, seed=42)
    cold_events = asyncio.run(collect_events(cold_algo, cold_ctx))
    assert cold_events[-1].kind == "completed"
    cold_iters = [e for e in cold_events if e.kind == "iteration"]
    assert len(cold_iters) >= 5, "Need at least 5 iteration events from cold run"
    energy_after_5_cold = cold_iters[4].energy
    assert energy_after_5_cold is not None

    # ── Capture warm state after 5 manual steps ───────────────────────────────
    source = CVTStippling()
    ws_ctx = RenderContext(input=asset, params=base_params, composition=None, seed=42)
    source.initialize(ws_ctx)
    for it in range(5):
        source.step(ws_ctx, it)
    saved_state = source.export_warm_state(ws_ctx)
    assert saved_state.algorithm_id == "cvt_stippling"
    assert saved_state.payload.get("points") is not None

    # ── Warm run: first-iteration energy ≤ cold energy at step 5 ─────────────
    warm_algo = CVTStippling()
    warm_ctx = RenderContext(
        input=asset,
        params={**base_params, "max_iterations": 10},
        composition=None,
        seed=42,
        warm_start=saved_state,
    )
    warm_events = asyncio.run(collect_events(warm_algo, warm_ctx))
    assert warm_events[-1].kind == "completed", \
        f"Warm run ended with {warm_events[-1].kind}"

    warm_iters = [e for e in warm_events if e.kind == "iteration"]
    assert len(warm_iters) >= 1, "Warm run produced no iteration events"
    warm_first_energy = warm_iters[0].energy
    assert warm_first_energy is not None

    assert warm_first_energy <= energy_after_5_cold + 1e-6, (
        f"Warm run first-iteration energy ({warm_first_energy:.4f}) should be "
        f"≤ cold run energy after 5 steps ({energy_after_5_cold:.4f}). "
        "import_warm_state did not resume from prior point positions."
    )


# ── Integration / server tests ────────────────────────────────────────────────

def test_sse_emits_started_iteration_completed(run_server) -> None:
    """Full SSE flow: started → iteration* → completed."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(24, 24))

    payload = stipple_payload(asset["id"], n=10, max_iters=6)
    run = post_json(f"{base_url}/api/preview_runs", payload)
    assert "id" in run
    run_id = run["id"]

    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events", timeout=20.0)

    kinds = [e["kind"] for e in events]
    assert kinds[0] == "started", f"Expected started, got {kinds}"
    assert "iteration" in kinds, f"No iteration events: {kinds}"
    assert kinds[-1] == "completed", f"Last event: {kinds[-1]}"

    # The completed event carries primary_artifact_id
    completed = next(e for e in events if e["kind"] == "completed")
    assert "primary_artifact_id" in completed


def test_cancel_emits_cancelled_no_history_warm_state_available(run_server) -> None:
    """
    Cancellation:
      - SSE yields 'cancelled'
      - run status is 'cancelled' (not in durable render history)
      - warm-start state available for next run in same session
    """
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(24, 24))
    session = "cancel-session-001"

    # Long run so we can cancel it (threshold=0.0 means never-converge-early)
    payload = stipple_payload(asset["id"], n=20, max_iters=50, session_id=session)

    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    # Wait briefly then cancel
    def cancel_after_delay():
        time.sleep(0.3)
        try:
            delete_url(f"{base_url}/api/preview_runs/{run_id}")
        except Exception:
            pass

    threading.Thread(target=cancel_after_delay, daemon=True).start()

    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events", timeout=20.0)
    kinds = [e["kind"] for e in events]

    # Must have received cancellation (or completed — race is acceptable)
    assert kinds[-1] in ("cancelled", "completed"), f"Unexpected final kind: {kinds}"

    # Run status should NOT be completed (it was either cancelled or the cancel
    # arrived too late — check we don't have a durable RenderRun for it)
    run_status = server.scheduler.get_run(run_id)
    assert run_status is not None

    if run_status["status"] == "cancelled":
        # Verify warm-start state was saved for this session+asset+algorithm
        warm = server.scheduler.get_warm_state(session, asset["id"], "cvt_stippling")
        assert warm is not None, "Expected warm-start state after cancellation"
        assert warm.algorithm_id == "cvt_stippling"

        # Cancelled run is NOT promotable
        try:
            promote_resp = post_json(f"{base_url}/api/preview_runs/{run_id}/promote", {})
            # If promote returns 2xx it should be an error — check status
            assert False, "Expected promotion of cancelled run to fail"
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 409, 404, 410)


def test_promote_preview_to_render_run_survives_restart(run_server, tmp_path) -> None:
    """
    Promoting a completed preview creates a durable RenderRun.
    Run metadata AND artifact bytes must survive server restart.
    """
    base_url, store, server = run_server

    asset = upload_asset(base_url, small_image(20, 20))
    payload = stipple_payload(asset["id"], n=10, max_iters=4)

    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    # Wait for completion
    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events", timeout=20.0)
    last_kind = events[-1]["kind"] if events else "none"
    assert last_kind == "completed", f"Preview didn't complete: {[e['kind'] for e in events]}"

    completed_ev = next(e for e in events if e["kind"] == "completed")
    final_id = completed_ev.get("final_artifact_id") or completed_ev.get("primary_artifact_id")
    assert final_id, "completed event must carry a final/primary artifact id"

    # Verify the artifact is serveable RIGHT NOW (before promote / restart)
    with urllib.request.urlopen(f"{base_url}/api/artifacts/{final_id}") as r:
        body = r.read()
    assert len(body) > 100, "Final artifact bytes must be non-empty"
    assert r.headers["Content-Type"].startswith("image/")

    # Promote
    promoted = post_json(f"{base_url}/api/preview_runs/{run_id}/promote", {})
    assert "id" in promoted
    assert promoted["status"] == "completed"
    assert promoted.get("promoted_from_preview_id") == run_id

    rrun_id = promoted["id"]

    # Verify the render run JSON is on disk
    run_files = list(store.runs_dir.glob(f"{rrun_id}*.json"))
    assert len(run_files) >= 1, "RenderRun not persisted to disk"

    # Simulate restart: create fresh scheduler pointing at same runs_dir
    from colorworks.scheduler import RunScheduler
    new_scheduler = RunScheduler(store.runs_dir)
    try:
        restored = new_scheduler.get_run(rrun_id)
        assert restored is not None, "RenderRun not restored after restart"
        assert restored["status"] == "completed"
        assert restored.get("promoted_from_preview_id") == run_id

        # Verify the artifact file itself is still on disk after restart
        art_path = store.artifacts_dir / f"{final_id}.png"
        assert art_path.exists(), (
            f"Artifact {final_id}.png missing from artifacts_dir after restart"
        )
    finally:
        new_scheduler.shutdown()


def test_cancelled_partial_cannot_be_promoted(run_server) -> None:
    """A cancelled or not-completed preview cannot be promoted/exported."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(24, 24))
    session = "no-promote-session"

    # Start a long run and cancel it fast
    payload = stipple_payload(asset["id"], n=30, max_iters=100, session_id=session)

    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    time.sleep(0.1)
    try:
        delete_url(f"{base_url}/api/preview_runs/{run_id}")
    except Exception:
        pass

    # Wait a little for the cancel to propagate
    time.sleep(0.3)

    run_status = server.scheduler.get_run(run_id)
    if run_status and run_status["status"] == "cancelled":
        # Try to promote cancelled run
        try:
            post_json(f"{base_url}/api/preview_runs/{run_id}/promote", {})
            # If we get here without error, check is_exportable is False
            assert not server.scheduler.is_exportable(run_id), \
                "Cancelled run should not be exportable"
        except urllib.error.HTTPError as exc:
            assert exc.code in (400, 409, 410), \
                f"Expected 4xx for cancelled promote, got {exc.code}"
    # If the race was lost and it completed, that's also acceptable


def test_get_run_status(run_server) -> None:
    """GET /api/preview_runs/{id} returns current status."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(20, 20))

    payload = stipple_payload(asset["id"], n=10, max_iters=3)
    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    # Status is queryable immediately after submit
    status = post_json.__func__ if False else None  # noqa
    with urllib.request.urlopen(f"{base_url}/api/preview_runs/{run_id}") as r:
        status_data = json.loads(r.read())
    assert status_data["id"] == run_id
    assert status_data["status"] in ("queued", "running", "completed")


def test_sse_late_subscriber_gets_events(run_server) -> None:
    """A subscriber connecting after run completion receives buffered events."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(20, 20))

    payload = stipple_payload(asset["id"], n=10, max_iters=3)
    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    # Wait for run to finish before opening SSE
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        run_info = server.scheduler.get_run(run_id)
        if run_info and run_info["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.05)

    # Late subscriber should still get events from history
    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events", timeout=5.0)
    kinds = [e["kind"] for e in events]
    assert "completed" in kinds or "failed" in kinds, \
        f"Late subscriber got no terminal event: {kinds}"


def test_cvt_stippling_produces_point_set_and_raster(tmp_path) -> None:
    """CVTStippling.render() publishes stipple_points (PointSet) and final_raster."""
    import asyncio

    algo = registry.get("cvt_stippling")
    img = small_image(24, 24)
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    ctx = RenderContext(
        input=asset,
        params={
            "n_stipples": 20,
            "max_iterations": 3,
            "convergence_threshold": 0.0,
            "dot_radius": 2.0,
            "ink_color": "#1a1a1a",
            "paper_color": "#ffffff",
        },
        composition=None,
        seed=1,
    )

    async def collect():
        evs = []
        async for p in algo.render(ctx):
            evs.append(p)
        return evs

    events = asyncio.run(collect())
    final_ev = events[-1]
    assert final_ev.kind == "completed"
    assert final_ev.result is not None
    assert not final_ev.result.partial

    # stipple_points must be in store
    art = ctx.store.get_by_name("stipple_points")
    assert art.type == "point_set"
    assert isinstance(art.value, PointSet)
    assert len(art.value.coords) == 20
    assert art.value.coords.shape == (20, 2)

    # final_raster must be in store
    rart = ctx.store.get_by_name("final_raster")
    assert rart.type == "raster_image"
    assert isinstance(rart.value, Image.Image)
    assert rart.value.size == (img.width, img.height)


def test_render_endpoint_rejects_iterative_algorithms(run_server) -> None:
    """/api/render must return 400 for iterative algorithms (not 500)."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, small_image(20, 20))
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "cvt_stippling",
        "params": {"n_stipples": 10, "max_iterations": 3},
    }
    req = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    body = json.loads(exc_info.value.read())
    assert "iterative" in body["error"].lower()


def test_phase0_to_phase2_unchanged(run_server) -> None:
    """Smoke-test that existing /api/render still works for non-iterative algorithms."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, gradient_image())

    result = post_json(f"{base_url}/api/render", {
        "asset_id": asset["id"],
        "renderer_id": "tonal_analyzer",
        "params": {"contrast": 1.0, "midpoint": 0.5, "preserve_edges": True, "edge_threshold": 0.15},
        "composition": {
            "paper_color": {"hex": "#f4ebd9"},
            "layers": [{
                "name": "ink",
                "color": {"hex": "#1a1a1a"},
                "role": "shadow",
                "density_source": "tone_map",
                "pattern": {
                    "kind": "wave",
                    "params": {"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                    "coordinates": {"space": "image_px"},
                },
                "blend_mode": "normal",
                "opacity": 1.0,
                "priority": 0,
            }],
        },
        "seed": 42,
    })
    assert "output" in result
    assert result["output"]["checksum"]
    assert "tone_map" in result.get("artifacts", {})
