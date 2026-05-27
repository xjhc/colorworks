"""
Phase 3.5 acceptance tests: Pang structure-aware halftoning.

Evidence required:
  ✓ pang_halftoning registered; is_iterative=True, is_cancellable=True
  ✓ Same seed + params → identical final_raster checksum (deterministic)
  ✓ Output structurally differs from CVT on a structured fixture image
  ✓ Warm-start imports exact point positions; energy ≤ cold at same step
  ✓ Invalid orientation_source raises ValueError with clear message
  ✓ render() yields started, iteration(s) with energy, completed
  ✓ SSE integration: preview run emits started, iteration(s), completed
  ✓ Phase 0–3 test suite unaffected
"""
from __future__ import annotations

import asyncio
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

# Populate registry
import colorworks.algorithms.cvt_stippling
import colorworks.algorithms.pang_halftoning
import colorworks.algorithms.pattern_catalog
import colorworks.algorithms.tonal_analyzer

from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.algorithms.cvt_stippling import CVTStippling
from colorworks.algorithms.pang_halftoning import PangHalftoning
from colorworks.domain import (
    ArtifactStore,
    CancelToken,
    PointSet,
    RasterGrid,
    RenderResult,
    WarmStartState,
)


# ── Fixture images ────────────────────────────────────────────────────────────

def small_image(width: int = 32, height: int = 32) -> Image.Image:
    arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def structured_image(width: int = 48, height: int = 48) -> Image.Image:
    """Diagonal-stripe image with a clear, strong orientation field."""
    yy, xx = np.mgrid[0:height, 0:width]
    val = 128 + 100 * np.sin((xx + yy) * 2 * np.pi / 8.0)
    arr = np.clip(val, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def make_asset(img: Image.Image) -> tuple[MediaAsset, RasterGrid]:
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="a", checksum="c", image=img, substrate=substrate)
    return asset, substrate


async def collect_events(algo, ctx):
    evs = []
    async for p in algo.render(ctx):
        evs.append(p)
    return evs


# ── Server fixture & helpers (self-contained for this file) ──────────────────

@pytest.fixture
def run_server(tmp_path):
    from colorworks.web.server import ColorworksServer
    from colorworks.storage import LocalStore
    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", store, server
    server.shutdown()
    server.server_close()
    t.join(timeout=3)


def upload_asset(base_url: str, image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=buf.getvalue(),
        headers={"X-Filename": "test.png", "Content-Type": "image/png"},
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


def read_sse(url: str, timeout: float = 30.0) -> list[dict]:
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
                block, buf = buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"data:"):
                        try:
                            events.append(json.loads(line[5:].strip()))
                        except json.JSONDecodeError:
                            pass
            if events and events[-1].get("kind") in ("completed", "cancelled", "failed"):
                break
    return events


# ── Unit tests: registration & definition ────────────────────────────────────

def test_pang_registered_and_iterative() -> None:
    algo = registry.get("pang_halftoning")
    assert algo.definition.id == "pang_halftoning"
    assert algo.definition.family.value == "halftoning"
    assert algo.definition.role.value == "renderer"
    assert algo.definition.execution_profile.is_iterative
    assert algo.definition.execution_profile.is_cancellable
    assert algo.definition.capabilities.deterministic


# ── Unit tests: determinism ───────────────────────────────────────────────────

def test_pang_deterministic_same_seed() -> None:
    """Same seed + params → identical final_raster checksum."""
    img = structured_image(32, 32)
    asset, _ = make_asset(img)
    params = {
        "n_dots": 30,
        "max_iterations": 5,
        "convergence_threshold": 0.0,
        "w_orient": 0.5,
        "ssim_window": 5,
        "dot_radius": 2.0,
    }

    def run(seed: int) -> str:
        algo = PangHalftoning()
        ctx = RenderContext(input=asset, params=params, composition=None, seed=seed)
        asyncio.run(collect_events(algo, ctx))
        return ctx.store.get_by_name("final_raster").checksum

    cs1 = run(42)
    cs2 = run(42)
    assert cs1 == cs2, f"Non-deterministic: {cs1} != {cs2}"

    # Different seed → different output
    cs3 = run(99)
    assert cs1 != cs3, "Different seeds should produce different outputs"


# ── Unit tests: structural difference from CVT ───────────────────────────────

def test_pang_differs_from_cvt_on_structured_image() -> None:
    """Pang and CVT must produce meaningfully different outputs on a structured fixture."""
    img = structured_image(48, 48)
    asset, _ = make_asset(img)

    pang_algo = PangHalftoning()
    pang_ctx = RenderContext(
        input=asset,
        params={
            "n_dots": 50,
            "max_iterations": 8,
            "convergence_threshold": 0.0,
            "w_orient": 1.0,
            "ssim_window": 7,
            "dot_radius": 2.0,
        },
        composition=None,
        seed=42,
    )
    asyncio.run(collect_events(pang_algo, pang_ctx))
    pang_raster = pang_ctx.store.get_by_name("final_raster").value

    cvt_algo = CVTStippling()
    cvt_ctx = RenderContext(
        input=asset,
        params={
            "n_stipples": 50,
            "max_iterations": 8,
            "convergence_threshold": 0.0,
            "dot_radius": 2.0,
        },
        composition=None,
        seed=42,
    )
    asyncio.run(collect_events(cvt_algo, cvt_ctx))
    cvt_raster = cvt_ctx.store.get_by_name("final_raster").value

    # Checksums must differ (different algorithms, different dot arrangements)
    assert (
        pang_ctx.store.get_by_name("final_raster").checksum
        != cvt_ctx.store.get_by_name("final_raster").checksum
    )

    pang_arr = np.asarray(pang_raster, dtype=np.float32)
    cvt_arr = np.asarray(cvt_raster, dtype=np.float32)
    mse = float(np.mean((pang_arr - cvt_arr) ** 2))
    assert mse > 50.0, (
        f"Pang and CVT outputs are suspiciously similar (MSE={mse:.2f}). "
        "Expected structural difference due to orientation-aware annealing."
    )

    # Also verify Pang produces a PointSet artifact (not just a raster)
    pt_art = pang_ctx.store.get_by_name("halftone_points")
    assert pt_art.type == "point_set"
    assert isinstance(pt_art.value, PointSet)
    assert len(pt_art.value.coords) == 50


# ── Unit tests: warm-start contract ──────────────────────────────────────────

def test_pang_warm_start_energy_no_worse_than_cold() -> None:
    """Importing warm state restores exact energy from that cold-run checkpoint."""
    img = structured_image(32, 32)
    asset, _ = make_asset(img)
    params = {
        "n_dots": 25,
        "max_iterations": 10,
        "convergence_threshold": 0.0,
        "w_orient": 0.5,
        "ssim_window": 5,
    }

    # Cold run: 5 manual steps, capture state
    cold = PangHalftoning()
    cold_ctx = RenderContext(input=asset, params=params, composition=None, seed=42)
    cold.initialize(cold_ctx)
    for it in range(5):
        cold.step(cold_ctx, it)
    saved_state = cold.export_warm_state(cold_ctx)

    assert saved_state.algorithm_id == "pang_halftoning"
    assert saved_state.payload.get("points") is not None

    # Compare both using a fresh density rebuild (eliminates incremental FP drift).
    # Incremental density updates accumulate tiny float errors; import_warm_state
    # rebuilds from scratch.  We compare both fresh to get a like-for-like baseline.
    H, W = cold._tone_work.shape
    cold._dot_density = cold._build_density_map(H, W)
    energy_cold_fresh = cold._compute_total_energy()

    warm = PangHalftoning()
    warm_ctx = RenderContext(
        input=asset, params=params, composition=None, seed=42, warm_start=saved_state
    )
    warm.import_warm_state(warm_ctx, saved_state)
    energy_warm_start = warm.current_energy()

    # Both rebuild from the same point positions → energies must be essentially equal.
    assert abs(energy_warm_start - energy_cold_fresh) < 1e-3, (
        f"Warm-start energy ({energy_warm_start:.6f}) differs from cold fresh energy "
        f"({energy_cold_fresh:.6f}) by more than tolerance. "
        "import_warm_state did not restore state correctly."
    )
    # And warm must be no worse than the (possibly slightly higher) incremental cold.
    # Allow 0.01 for accumulated float drift between incremental and fresh builds.
    energy_cold_incremental = saved_state.energy
    assert energy_warm_start <= energy_cold_incremental + 0.01, (
        f"Warm energy ({energy_warm_start:.4f}) > cold incremental energy "
        f"({energy_cold_incremental:.4f}) + tolerance."
    )


def test_pang_can_warm_start_logic() -> None:
    algo = PangHalftoning()
    state = WarmStartState(
        algorithm_id="pang_halftoning", algorithm_version="1.0.0",
        iteration=5, energy=1.23,
        params={"n_dots": 30, "ssim_window": 7},
        payload={"points": []},
    )
    # Matching n_dots and ssim_window → True
    assert algo.can_warm_start(state, {"n_dots": 30, "ssim_window": 7})
    # Different n_dots → False
    assert not algo.can_warm_start(state, {"n_dots": 50, "ssim_window": 7})
    # Different ssim_window → False
    assert not algo.can_warm_start(state, {"n_dots": 30, "ssim_window": 5})
    # Wrong algorithm_id → False
    bad = WarmStartState(
        algorithm_id="cvt_stippling", algorithm_version="1.0.0",
        iteration=0, energy=None, params={"n_dots": 30, "ssim_window": 7},
    )
    assert not algo.can_warm_start(bad, {"n_dots": 30, "ssim_window": 7})


# ── Unit tests: orientation_source validation ─────────────────────────────────

def test_pang_invalid_orientation_source_raises_value_error() -> None:
    """Non-internal orientation_source must raise ValueError with clear message."""
    img = small_image(16, 16)
    asset, _ = make_asset(img)

    for bad_src in ["some_other_run_field", "external_artifact_123", "my_run_orient"]:
        algo = PangHalftoning()
        ctx = RenderContext(
            input=asset,
            params={"n_dots": 5, "max_iterations": 2, "orientation_source": bad_src},
            composition=None,
            seed=0,
        )
        with pytest.raises(ValueError, match="orientation_source"):
            asyncio.run(collect_events(algo, ctx))


def test_pang_valid_orientation_source_aliases() -> None:
    """'internal', 'orientation_field', and '' are all accepted."""
    img = small_image(16, 16)
    asset, _ = make_asset(img)

    for src in ("internal", "orientation_field", ""):
        algo = PangHalftoning()
        ctx = RenderContext(
            input=asset,
            params={"n_dots": 5, "max_iterations": 2, "orientation_source": src,
                    "convergence_threshold": 0.0},
            composition=None,
            seed=0,
        )
        events = asyncio.run(collect_events(algo, ctx))
        assert events[-1].kind == "completed", (
            f"orientation_source='{src}' should be accepted; got {events[-1].kind}"
        )


# ── Unit tests: SSE event sequence ───────────────────────────────────────────

def test_pang_render_emits_started_iterations_completed() -> None:
    """render() yields started, ≥1 iteration with energy, then completed."""
    img = structured_image(24, 24)
    asset, _ = make_asset(img)
    ctx = RenderContext(
        input=asset,
        params={
            "n_dots": 15,
            "max_iterations": 6,
            "convergence_threshold": 0.0,
            "w_orient": 0.5,
            "ssim_window": 5,
        },
        composition=None,
        seed=7,
    )
    events = asyncio.run(collect_events(PangHalftoning(), ctx))
    kinds = [e.kind for e in events]

    assert kinds[0] == "started", f"First event must be 'started', got {kinds[0]}"
    assert "iteration" in kinds, f"No iteration events in {kinds}"
    assert kinds[-1] == "completed", f"Last event must be 'completed', got {kinds[-1]}"

    iter_events = [e for e in events if e.kind == "iteration"]
    assert len(iter_events) >= 1
    for ev in iter_events:
        assert ev.energy is not None, f"Iteration event missing energy field: {ev}"

    completed = events[-1]
    assert completed.result is not None
    assert not completed.result.partial
    assert completed.result.final_artifact_id is not None


def test_pang_cancel_during_iteration_yields_warm_state() -> None:
    """Cancelling a Pang run yields a 'cancelled' event with warm state."""
    img = structured_image(24, 24)
    asset, _ = make_asset(img)

    algo = PangHalftoning()
    ctx = RenderContext(
        input=asset,
        params={"n_dots": 20, "max_iterations": 100, "convergence_threshold": 0.0},
        composition=None,
        seed=5,
    )

    cancel_at_iter = [3]

    async def run_with_cancel():
        evs = []
        async for p in algo.render(ctx):
            evs.append(p)
            if p.kind == "iteration" and p.iteration == cancel_at_iter[0]:
                ctx.cancel.cancel()
        return evs

    events = asyncio.run(run_with_cancel())
    kinds = [e.kind for e in events]
    assert kinds[-1] == "cancelled", f"Expected cancelled, got {kinds}"
    last = events[-1]
    assert last.result is not None
    assert last.result.partial is True
    assert last.result.warm_state is not None
    assert last.result.warm_state.payload.get("points") is not None


# ── Integration test: SSE via HTTP server ─────────────────────────────────────

def test_pang_sse_via_server(run_server) -> None:
    """Full SSE flow: started → iteration(s) with energy → completed."""
    base_url, store, server = run_server
    asset = upload_asset(base_url, structured_image(24, 24))

    payload = {
        "asset_id": asset["id"],
        "renderer_id": "pang_halftoning",
        "params": {
            "n_dots": 15,
            "max_iterations": 6,
            "convergence_threshold": 0.0,
            "w_orient": 0.5,
            "ssim_window": 5,
            "dot_radius": 2.0,
        },
        "seed": 7,
        "session_id": "pang-sse-test",
    }

    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events", timeout=30.0)
    kinds = [e["kind"] for e in events]

    assert kinds[0] == "started", f"First event must be 'started': {kinds}"
    assert "iteration" in kinds, f"No iteration events: {kinds}"
    assert kinds[-1] == "completed", f"Last event must be 'completed': {kinds[-1]}"

    iter_events = [e for e in events if e["kind"] == "iteration"]
    for ev in iter_events:
        assert "energy" in ev, f"Iteration event missing energy: {ev}"

    completed = next(e for e in events if e["kind"] == "completed")
    assert "primary_artifact_id" in completed


# ── Regression: concurrent runs do not corrupt each other ────────────────────

def test_pang_concurrent_runs_do_not_corrupt_each_other() -> None:
    """Two concurrent Pang renders with different n_dots must each finish with the right count.

    This fails when the registry returns a singleton: both coroutines share the
    same instance and the second initialize() call overwrites _points for both.
    It passes when registry.get() returns a fresh instance per call.
    """
    img = small_image(16, 16)
    asset, _ = make_asset(img)

    async def run_pang(n_dots: int) -> int:
        algo = registry.get("pang_halftoning")
        ctx = RenderContext(
            input=asset,
            params={"n_dots": n_dots, "max_iterations": 10, "convergence_threshold": 0.0},
            composition=None,
            seed=0,
        )
        async for _ in algo.render(ctx):
            # Explicit yield to the event loop so asyncio.gather can interleave
            # both coroutines.  Without this, the generator never suspends and
            # the event loop completes the first task before starting the second,
            # so the singleton bug would not be detectable in unit tests.
            await asyncio.sleep(0)
        return len(ctx.store.get_by_name("halftone_points").value.coords)

    async def concurrent():
        return await asyncio.gather(run_pang(12), run_pang(31))

    counts = asyncio.run(concurrent())
    assert counts[0] == 12, (
        f"Expected 12 dots for first run, got {counts[0]}. "
        "Singleton state corruption: a concurrent run overwrote _points."
    )
    assert counts[1] == 31, (
        f"Expected 31 dots for second run, got {counts[1]}. "
        "Singleton state corruption: a concurrent run overwrote _points."
    )


# ── Regression: phase 0–3 algorithms still registered ────────────────────────

def test_existing_algorithms_still_registered() -> None:
    for algo_id in ("tonal_analyzer", "floyd_steinberg", "cvt_stippling", "pang_halftoning"):
        algo = registry.get(algo_id)
        assert algo.definition.id == algo_id
