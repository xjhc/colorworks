from __future__ import annotations

import io
import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.algorithms.palette_quantize import PaletteQuantizeRenderer
from colorworks.domain import ArtifactStore, RasterGrid
from colorworks.scheduler import RunScheduler


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


@pytest.mark.asyncio
async def test_palette_quantize_direct():
    img = make_test_image(32, 32)
    asset, substrate = make_asset(img)
    
    # 4 colors, adaptive, dithered
    ctx = RenderContext(
        input=asset,
        params={"colors": 4, "palette": "adaptive", "dither": True},
        composition=None,
        seed=42,
        store=ArtifactStore(),
    )
    
    algo = PaletteQuantizeRenderer()
    evs = []
    async for p in algo.render(ctx):
        evs.append(p)
        
    assert len(evs) >= 2
    assert evs[-1].kind == "completed"
    
    art = ctx.store.get_by_name("final_raster")
    assert art.type == "raster_image"
    assert isinstance(art.value, Image.Image)
    assert art.value.size == (32, 32)
    
    colors = art.value.getcolors()
    assert len(colors) <= 4


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


def test_candidates_endpoint(run_server):
    base_url, store, server = run_server
    img = make_test_image(32, 32)
    asset = upload_asset(base_url, img)
    
    payload = {
        "asset_id": asset["id"],
        "output": {"max_width": 16, "max_height": None, "fit": "fit"},
        "colors": {"count": 4, "palette": "adaptive"},
        "style_filter": ["dither"]
    }
    
    res = post_json(f"{base_url}/api/candidates", payload)
    assert "candidate_set_id" in res
    assert "candidates" in res
    candidates = res["candidates"]
    assert len(candidates) > 0
    
    for cand in candidates:
        assert cand["style_tag"] == "dither"
        assert "preview_run_id" in cand
        
        # Verify preview runs are registered in scheduler
        run_id = cand["preview_run_id"]
        # Poll status until completed
        for _ in range(50):
            run_status = server.scheduler.get_run(run_id)
            if run_status and run_status["status"] == "completed":
                break
            time.sleep(0.1)
            
        run_status = server.scheduler.get_run(run_id)
        assert run_status["status"] == "completed"
        art_id = run_status["final_artifact_id"] or run_status["primary_artifact_id"]
        assert art_id is not None
        
        with urllib.request.urlopen(f"{base_url}/api/artifacts/{art_id}") as response:
            data = response.read()
            assert len(data) > 0
            img_out = Image.open(io.BytesIO(data))
            assert img_out.size == (16, 16)


def test_select_candidates_ink_paper():
    from colorworks.algorithms.quick_mode import select_candidates
    # ink_paper: the flat palette_quantize card collapses to 2-color adaptive,
    # while tone_dither cards become a duotone (ink->paper) ramp at the requested count.
    selected = select_candidates(colors=6, palette="ink_paper")

    quant_cands = [c for c in selected if c["algorithm"] == "palette_quantize"]
    assert len(quant_cands) == 1
    assert quant_cands[0]["params"]["colors"] == 2
    assert quant_cands[0]["params"]["palette"] == "adaptive"

    tone_cands = [c for c in selected if c["algorithm"] == "tone_dither"]
    assert len(tone_cands) >= 4
    for cand in tone_cands:
        assert cand["params"]["colors"] == 6
        assert cand["params"]["palette"] == "duotone"
        assert "method" in cand["params"]


def test_select_candidates_grayscale_multitone():
    from colorworks.algorithms.quick_mode import select_candidates
    selected = select_candidates(colors=4, palette="grayscale")
    tone_cands = [c for c in selected if c["algorithm"] == "tone_dither"]
    assert tone_cands
    for cand in tone_cands:
        assert cand["params"]["colors"] == 4
        assert cand["params"]["palette"] == "grayscale"


def test_select_candidates_style_filter_flow():
    from colorworks.algorithms.quick_mode import select_candidates
    selected = select_candidates(colors=4, palette="grayscale", style_filter=["flow"])
    assert selected
    methods = {c["params"].get("method") for c in selected}
    assert methods == {"flow"}

