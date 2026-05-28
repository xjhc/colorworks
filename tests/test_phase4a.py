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

# Populate registry with dbs and other required algorithms
import colorworks.algorithms.dbs
from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.algorithms.dbs import DBSRenderer, DEFAULT_HVS_CHECKSUM
from colorworks.domain import (
    ArtifactStore,
    CancelToken,
    RasterGrid,
    RenderResult,
    WarmStartState,
    RunStatus,
    RenderRun,
)
from colorworks.scheduler import RunScheduler


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def make_test_image(width: int = 16, height: int = 16) -> Image.Image:
    """Create a deterministic gradient image for testing."""
    yy, xx = np.mgrid[0:height, 0:width]
    val = (xx + yy) * 255.0 / (width + height - 2.0)
    arr = np.clip(val, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def make_asset(img: Image.Image) -> tuple[MediaAsset, RasterGrid]:
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="test_asset", checksum="test_checksum", image=img, substrate=substrate)
    return asset, substrate


async def collect_events(algo, ctx):
    evs = []
    async for p in algo.render(ctx):
        evs.append(p)
    return evs


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


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


# ── Acceptance Tests ──────────────────────────────────────────────────────────

def test_dbs_registration() -> None:
    """Verify that DBSRenderer is registered under family 'halftoning' and role 'renderer'."""
    algo = registry.get("dbs")
    assert algo.definition.id == "dbs"
    assert algo.definition.family.value == "halftoning"
    assert algo.definition.role.value == "renderer"
    assert algo.definition.execution_profile.is_iterative
    assert algo.definition.capabilities.deterministic
    assert len(algo.definition.calibration_assets) == 1
    assert algo.definition.calibration_assets[0].asset_id == "hvs_model"
    assert algo.definition.calibration_assets[0].checksum == DEFAULT_HVS_CHECKSUM


def test_dbs_determinism() -> None:
    """Verify identical outputs under identical seed and input image, and varying outputs under different seeds."""
    img = make_test_image(16, 16)
    asset, _ = make_asset(img)
    params = {
        "max_iterations": 2,
        "ink_color": "#1a1a1a",
        "paper_color": "#f4ebd9",
    }

    # Helper wrapper to run the DBS algorithm in-memory
    def run_dbs_algo(seed: int) -> str:
        # Mock calibration accessor referencing the default HVS model
        class MockStore:
            def get_calibration_metadata(self, checksum):
                return {"version": "1.0.0"}
            def get_calibration_data(self, checksum):
                from colorworks.algorithms.dbs import DEFAULT_HVS_DATA
                return DEFAULT_HVS_DATA

        from colorworks.algorithms import CalibrationAccessor
        mock_accessor = CalibrationAccessor(MockStore())

        algo = DBSRenderer()
        ctx = RenderContext(
            input=asset,
            params=params,
            composition=None,
            seed=seed,
            calibration=mock_accessor,
        )
        asyncio.run(collect_events(algo, ctx))
        return ctx.store.get_by_name("final_raster").checksum

    # Identical seed -> identical checksum
    cs1 = run_dbs_algo(42)
    cs2 = run_dbs_algo(42)
    assert cs1 == cs2, f"DBS with same seed is non-deterministic: {cs1} != {cs2}"

    # Different seed -> different checksum
    cs3 = run_dbs_algo(123)
    assert cs1 != cs3, "DBS with different seeds should yield different outputs"


def test_dbs_input_size_constraints(run_server) -> None:
    """Verify that submitting an input image size > 64x64 pixel limit raises a ValueError/HTTP 400."""
    base_url, store, _ = run_server

    # 1. Large image (128x128)
    large_img = make_test_image(128, 128)
    asset_record = upload_asset(base_url, large_img)

    payload = {
        "asset_id": asset_record["id"],
        "renderer_id": "dbs",
        "params": {
            "max_iterations": 2,
        },
        "seed": 42,
    }

    # Submit render run (which triggers size preflight check)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        post_json(f"{base_url}/api/render_runs", payload)

    assert exc_info.value.code == 400
    err_body = exc_info.value.read().decode("utf-8")
    assert "exceed the 64x64 pixel limit" in err_body


def test_calibration_asset_metadata_record_persistence(run_server) -> None:
    """Verify that calibration checksum and version are written to metadata, retrieved, and survive scheduler reload."""
    base_url, store, server = run_server

    # Make a small 16x16 image and upload it
    img = make_test_image(16, 16)
    asset_record = upload_asset(base_url, img)

    # Submit a preview run (to check preview run attributes too)
    payload = {
        "asset_id": asset_record["id"],
        "renderer_id": "dbs",
        "params": {
            "max_iterations": 2,
        },
        "seed": 42,
    }

    preview_res = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = preview_res["id"]
    assert "calibration_checksum" in preview_res
    assert "calibration_version" in preview_res
    assert preview_res["calibration_checksum"] == DEFAULT_HVS_CHECKSUM
    assert preview_res["calibration_version"] == "1.0.0"

    # Poll status until completed
    completed = False
    for _ in range(50):
        status_res = get_json(f"{base_url}/api/preview_runs/{run_id}")
        if status_res["status"] == "completed":
            completed = True
            break
        time.sleep(0.1)
    assert completed, "Preview run did not complete in time"

    # Promote the preview run to a render run
    promote_res = post_json(f"{base_url}/api/preview_runs/{run_id}/promote", {})
    render_run_id = promote_res["id"]
    assert promote_res["calibration_checksum"] == preview_res["calibration_checksum"]
    assert promote_res["calibration_version"] == "1.0.0"

    # Verify that it is written to the RenderRun JSON file in store
    run_file = store.runs_dir / f"{render_run_id}.json"
    assert run_file.exists()
    run_meta = json.loads(run_file.read_text(encoding="utf-8"))
    assert run_meta["calibration_checksum"] == preview_res["calibration_checksum"]
    assert run_meta["calibration_version"] == "1.0.0"

    # Reload scheduler from runs dir and assert restored fields
    fresh_scheduler = RunScheduler(store.runs_dir)
    try:
        restored_run = fresh_scheduler.get_run(render_run_id)
        assert restored_run is not None
        assert restored_run["calibration_checksum"] == preview_res["calibration_checksum"]
        assert restored_run["calibration_version"] == "1.0.0"
    finally:
        fresh_scheduler.shutdown()


def test_cache_invalidation(run_server) -> None:
    """Verify that different calibration checksums passed to get_artifact_cache_key yield distinct cache keys."""
    _, store, _ = run_server

    algo = registry.get("dbs")

    key1 = store.get_artifact_cache_key(
        algo_id=algo.definition.id,
        algo_version=algo.definition.version,
        artifact_name="final_raster",
        asset_checksum="asset_abc",
        params={"max_iterations": 2},
        parameters_def=algo.definition.parameters,
        calibration_assets_checksum="checksum_aaa",
    )

    key2 = store.get_artifact_cache_key(
        algo_id=algo.definition.id,
        algo_version=algo.definition.version,
        artifact_name="final_raster",
        asset_checksum="asset_abc",
        params={"max_iterations": 2},
        parameters_def=algo.definition.parameters,
        calibration_assets_checksum="checksum_bbb",
    )

    assert key1 != key2, "Different calibration checksums must produce different cache keys"


def test_dbs_disables_warm_start() -> None:
    """Verify that DBS can_warm_start returns False."""
    algo = DBSRenderer()
    state = WarmStartState(
        algorithm_id="dbs",
        algorithm_version="1.0.0",
        iteration=1,
        energy=10.0,
        params={},
    )
    assert algo.can_warm_start(state, {}) is False


def test_dbs_binary_output_validity() -> None:
    """Assert that the halftone matrix `b` consists strictly of {0.0, 1.0} values prior to final color mapping."""
    img = make_test_image(8, 8)
    asset, _ = make_asset(img)
    params = {
        "max_iterations": 1,
    }

    class MockStore:
        def get_calibration_metadata(self, checksum):
            return {"version": "1.0.0"}
        def get_calibration_data(self, checksum):
            from colorworks.algorithms.dbs import DEFAULT_HVS_DATA
            return DEFAULT_HVS_DATA

    from colorworks.algorithms import CalibrationAccessor
    mock_accessor = CalibrationAccessor(MockStore())

    algo = DBSRenderer()
    ctx = RenderContext(
        input=asset,
        params=params,
        composition=None,
        seed=42,
        calibration=mock_accessor,
    )

    # Run initialize and check intermediate state b
    algo.initialize(ctx)
    assert np.all(np.isin(algo._b, [0.0, 1.0]))

    # Run step and verify b values remain binary
    algo.step(ctx, 0)
    assert np.all(np.isin(algo._b, [0.0, 1.0]))
