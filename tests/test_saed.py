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

import colorworks.algorithms.saed
from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.algorithms.saed import SAEDRenderer, DEFAULT_SAED_CHECKSUM
from colorworks.domain import (
    ArtifactStore,
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


def make_flat_image(val: int, width: int = 16, height: int = 16) -> Image.Image:
    """Create a flat image of a given intensity."""
    arr = np.full((height, width, 3), val, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


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

def test_saed_registration() -> None:
    """Verify that SAEDRenderer is registered under family 'halftoning' and role 'renderer'."""
    algo = registry.get("saed")
    assert algo.definition.id == "saed"
    assert algo.definition.family.value == "halftoning"
    assert algo.definition.role.value == "renderer"
    assert not algo.definition.execution_profile.is_iterative
    assert algo.definition.capabilities.deterministic
    assert len(algo.definition.calibration_assets) == 1
    assert algo.definition.calibration_assets[0].asset_id == "saed_gabor_lut"
    assert algo.definition.calibration_assets[0].checksum == DEFAULT_SAED_CHECKSUM


def test_saed_schema_exposure(run_server) -> None:
    """Verify that /api/schemas exposes saed with parameters."""
    base_url, _, _ = run_server
    schemas = get_json(f"{base_url}/api/schemas")
    algos = schemas.get("algorithms", [])
    saed_schema = next((a for a in algos if a["id"] == "saed"), None)
    assert saed_schema is not None
    assert saed_schema["name"] == "Structure-Aware Error Diffusion"
    params = saed_schema["parameters"]
    assert any(p["key"] == "gabor_amplitude" for p in params)
    assert any(p["key"] == "anisotropy_alpha" for p in params)
    assert any(p["key"] == "edge_scaling" for p in params)


def test_saed_determinism_and_seed_insensitivity() -> None:
    """Verify identical outputs under varying seeds (seed-insensitivity) but fixed inputs/params."""
    img = make_test_image(32, 32)
    asset, _ = make_asset(img)
    params = {
        "contrast": 1.2,
        "midpoint": 0.45,
        "gabor_amplitude": 0.15,
        "anisotropy_alpha": 0.4,
        "ink_color": "#000000",
        "paper_color": "#ffffff",
    }

    def run_saed_algo(seed: int) -> tuple[str, np.ndarray]:
        class MockStore:
            def get_calibration_metadata(self, checksum):
                return {"version": "1.0.0"}
            def get_calibration_data(self, checksum):
                from colorworks.algorithms.saed import DEFAULT_SAED_DATA
                return DEFAULT_SAED_DATA

        from colorworks.algorithms import CalibrationAccessor
        mock_accessor = CalibrationAccessor(MockStore())

        algo = SAEDRenderer()
        ctx = RenderContext(
            input=asset,
            params=params,
            composition=None,
            seed=seed,
            calibration=mock_accessor,
        )
        asyncio.run(collect_events(algo, ctx))
        art = ctx.store.get_by_name("final_raster")
        # Load output image pixels
        img_out = art.value
        arr = np.array(img_out)
        return art.checksum, arr

    # Assert seed-insensitivity: checksums and image arrays must be identical for different seeds
    cs1, arr1 = run_saed_algo(42)
    cs2, arr2 = run_saed_algo(123)
    assert cs1 == cs2, f"SAED outputs differ by seed: {cs1} != {cs2}"
    assert np.array_equal(arr1, arr2), "SAED pixel outputs differ by seed"


def test_saed_orientation_sensitivity() -> None:
    """Verify that structure-aware parameters affect output, and Gabor LUT math is correct."""
    # 1. Assert Gabor LUT Gabor symmetry/alignment properties
    from colorworks.algorithms.saed import DEFAULT_SAED_DATA
    # DEFAULT_SAED_DATA shape is (180, 11, 11)

    # Angle 0 (horizontal flow): normal is vertical. x_rot depends only on y.
    # Therefore, the Gabor kernel should be symmetric along the horizontal axis: lut[0, y, x] == lut[0, y, -x]
    # Center index of kernel is 5. Compare index 3 (5-2) and 7 (5+2).
    assert np.allclose(DEFAULT_SAED_DATA[0, :, 3], DEFAULT_SAED_DATA[0, :, 7])

    # Angle 90 (vertical flow): normal is horizontal. x_rot depends only on x.
    # Therefore, the Gabor kernel should be symmetric along the vertical axis: lut[90, y, x] == lut[90, -y, x]
    assert np.allclose(DEFAULT_SAED_DATA[90, 3, :], DEFAULT_SAED_DATA[90, 7, :])

    # 2. Check horiz stripes image with structure-aware enabled vs disabled on same input
    size = 32
    yy, xx = np.mgrid[0:size, 0:size]
    h_stripes = (np.sin(yy * 0.5) * 127 + 128).astype(np.uint8)
    h_rgb = np.stack([h_stripes, h_stripes, h_stripes], axis=-1)
    img_h = Image.fromarray(h_rgb)
    asset_h, _ = make_asset(img_h)

    def run_saed(gabor_amp: float, aniso_alpha: float) -> str:
        algo = SAEDRenderer()
        ctx = RenderContext(
            input=asset_h,
            params={
                "gabor_amplitude": gabor_amp,
                "anisotropy_alpha": aniso_alpha,
                "sigma": 1.5,
                "etf_iterations": 2,
            },
            composition=None,
            seed=42,
        )
        asyncio.run(collect_events(algo, ctx))
        return ctx.store.get_by_name("final_raster").checksum

    cs_enabled = run_saed(0.3, 0.7)
    cs_disabled = run_saed(0.0, 0.0)
    assert cs_enabled != cs_disabled, "SAED output did not change when structure-aware parameters were disabled"


def test_saed_tone_convention_and_colors() -> None:
    """Verify tone convention (1.0=ink, 0.0=paper) and correct color mapping."""
    ink_color = "#112233"
    paper_color = "#ddeeff"
    from colorworks.algorithms.image_ops import parse_color
    ink_rgb = parse_color(ink_color)
    paper_rgb = parse_color(paper_color)

    # 1. Flat white image (val = 255) -> 0.0 density -> expect 100% paper_color
    img_white = make_flat_image(255, 16, 16)
    asset_w, _ = make_asset(img_white)
    algo = SAEDRenderer()
    ctx_w = RenderContext(
        input=asset_w,
        params={"ink_color": ink_color, "paper_color": paper_color},
        composition=None,
        seed=42,
    )
    asyncio.run(collect_events(algo, ctx_w))
    arr_w = np.array(ctx_w.store.get_by_name("final_raster").value)

    # Assert all pixels are paper_color
    assert np.all(arr_w == paper_rgb)

    # 2. Flat black image (val = 0) -> 1.0 density -> expect 100% ink_color
    img_black = make_flat_image(0, 16, 16)
    asset_b, _ = make_asset(img_black)
    ctx_b = RenderContext(
        input=asset_b,
        params={"ink_color": ink_color, "paper_color": paper_color},
        composition=None,
        seed=42,
    )
    asyncio.run(collect_events(algo, ctx_b))
    arr_b = np.array(ctx_b.store.get_by_name("final_raster").value)

    # Assert all pixels are ink_color
    assert np.all(arr_b == ink_rgb)


def test_saed_size_guardrails(run_server) -> None:
    """Verify that images exceeding 256x256 raise ValueError in all routes."""
    base_url, _, _ = run_server

    # Upload 257x256 image (exceeding 256 limit)
    large_img = make_test_image(257, 256)
    asset_record = upload_asset(base_url, large_img)

    payload = {
        "asset_id": asset_record["id"],
        "renderer_id": "saed",
        "params": {},
        "seed": 42,
    }

    # 1. Test /api/render synchronous route
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        post_json(f"{base_url}/api/render", payload)
    assert exc_info.value.code == 400
    assert "exceed the 256x256 pixel limit" in exc_info.value.read().decode()

    # 2. Test /api/preview_runs route
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        post_json(f"{base_url}/api/preview_runs", payload)
    assert exc_info.value.code == 400
    assert "exceed the 256x256 pixel limit" in exc_info.value.read().decode()

    # 3. Test /api/render_runs route
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        post_json(f"{base_url}/api/render_runs", payload)
    assert exc_info.value.code == 400
    assert "exceed the 256x256 pixel limit" in exc_info.value.read().decode()


def test_saed_calibration_metadata_record_persistence(run_server) -> None:
    """Verify calibration metadata persistence through preview, promotion, reload, and API."""
    base_url, store, server = run_server

    img = make_test_image(16, 16)
    asset_record = upload_asset(base_url, img)

    payload = {
        "asset_id": asset_record["id"],
        "renderer_id": "saed",
        "params": {},
        "seed": 42,
    }

    # 1. Submit preview run
    preview_res = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = preview_res["id"]
    assert "calibration_checksum" in preview_res
    assert "calibration_version" in preview_res
    assert preview_res["calibration_checksum"] == DEFAULT_SAED_CHECKSUM
    assert preview_res["calibration_version"] == "1.0.0"

    # Poll status until completed
    completed = False
    status_res = None
    for _ in range(50):
        status_res = get_json(f"{base_url}/api/preview_runs/{run_id}")
        if status_res["status"] == "completed":
            completed = True
            break
        time.sleep(0.1)
    assert completed, "Preview run did not complete"
    assert status_res["primary_artifact_id"] is not None
    assert status_res["final_artifact_id"] is not None

    # 2. Promote to render run
    promote_res = post_json(f"{base_url}/api/preview_runs/{run_id}/promote", {})
    render_run_id = promote_res["id"]
    assert promote_res["calibration_checksum"] == DEFAULT_SAED_CHECKSUM
    assert promote_res["calibration_version"] == "1.0.0"
    assert promote_res["primary_artifact_id"] == status_res["primary_artifact_id"]
    assert promote_res["final_artifact_id"] == status_res["final_artifact_id"]

    # Verify that it is written to the RenderRun JSON file in store
    run_file = store.runs_dir / f"{render_run_id}.json"
    assert run_file.exists()
    run_meta = json.loads(run_file.read_text(encoding="utf-8"))
    assert run_meta["calibration_checksum"] == DEFAULT_SAED_CHECKSUM
    assert run_meta["calibration_version"] == "1.0.0"

    # 3. Reload scheduler and assert restored fields
    fresh_scheduler = RunScheduler(store.runs_dir)
    try:
        restored_run = fresh_scheduler.get_run(render_run_id)
        assert restored_run is not None
        assert restored_run["calibration_checksum"] == DEFAULT_SAED_CHECKSUM
        assert restored_run["calibration_version"] == "1.0.0"
    finally:
        fresh_scheduler.shutdown()


def test_saed_runtime_reporting(run_server) -> None:
    """Measure and log the SAED execution time for 128x128 and 256x256 images."""
    base_url, _, _ = run_server

    for dim in (128, 256):
        img = make_test_image(dim, dim)
        asset_record = upload_asset(base_url, img)

        payload = {
            "asset_id": asset_record["id"],
            "renderer_id": "saed",
            "params": {},
            "seed": 42,
        }

        start = time.perf_counter()
        res = post_json(f"{base_url}/api/render", payload)
        duration = time.perf_counter() - start

        print(f"\n[TIMING EVIDENCE] SAED render of {dim}x{dim} took {duration:.4f} seconds ({res['output']['render_ms']} ms server-reported)")

        # Loose assertion to detect catastrophic degradation
        assert duration < 10.0, f"SAED render for {dim}x{dim} was excessively slow: {duration:.2f}s"


def test_saed_calibration_missing_data_propagates() -> None:
    """Verify that a KeyError is propagated when calibration data is missing from the store."""
    img = make_test_image(16, 16)
    asset, _ = make_asset(img)

    class MissingCalibrationAccessor:
        def get_metadata(self, checksum):
            return {"version": "1.0.0"}
        def get_data(self, checksum):
            raise KeyError("calibration data not found")

    algo = SAEDRenderer()
    ctx = RenderContext(
        input=asset,
        params={},
        composition=None,
        seed=42,
        calibration=MissingCalibrationAccessor(),
    )
    with pytest.raises(KeyError) as exc_info:
        asyncio.run(collect_events(algo, ctx))
    assert "calibration data not found" in str(exc_info.value)


def test_saed_calibration_invalid_shape_raises_value_error() -> None:
    """Verify that a ValueError is raised if the Gabor LUT has an invalid shape."""
    img = make_test_image(16, 16)
    asset, _ = make_asset(img)

    class InvalidCalibrationAccessor:
        def get_metadata(self, checksum):
            return {"version": "1.0.0"}
        def get_data(self, checksum):
            # Invalid 2D shape instead of 3D
            return np.zeros((11, 11), dtype=np.float32)

    algo = SAEDRenderer()
    ctx = RenderContext(
        input=asset,
        params={},
        composition=None,
        seed=42,
        calibration=InvalidCalibrationAccessor(),
    )
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(collect_events(algo, ctx))
    assert "SAED Gabor LUT must be a 3D array" in str(exc_info.value)
