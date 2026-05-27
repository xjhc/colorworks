from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import threading
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

from colorworks.domain import (
    RasterGrid,
    ScalarField,
    BinaryMask,
    ArtifactStore,
    Composition,
    InkLayerSpec,
    PaletteColor,
    PatternSpec,
    PatternCoordinateSpec,
)
from colorworks.algorithms import registry, MediaAsset, RenderContext
from colorworks.compositor import Compositor
from colorworks.storage import LocalStore

import colorworks.algorithms.tonal_analyzer
import colorworks.algorithms.pattern_catalog
import colorworks.algorithms.floyd_steinberg

@pytest.fixture(autouse=True)
def clean_registry():
    state = registry.save_state()
    yield
    registry.restore_state(state)

def test_patterns_registration_and_generation() -> None:
    # Verify patterns are registered
    patterns = [p.kind for p in registry.list_patterns()]
    assert "maze" in patterns
    assert "blue_noise" in patterns
    assert "ordered_dither" in patterns
    assert "hatch" in patterns
    
    # Generate patterns
    store = ArtifactStore()
    compositor = Compositor(store)
    
    # ordered_dither
    pat = PatternSpec(kind="ordered_dither", params={"matrix_size": 8})
    res = compositor._generate_pattern(pat, 32, 32, 42)
    assert res.shape == (32, 32)
    assert np.min(res) >= 0.0 and np.max(res) <= 1.0
    
    # blue_noise
    pat = PatternSpec(kind="blue_noise", params={"size": 16})
    res = compositor._generate_pattern(pat, 32, 32, 42)
    assert res.shape == (32, 32)
    assert np.min(res) >= 0.0 and np.max(res) <= 1.0
    
    # maze
    pat = PatternSpec(kind="maze", params={"scale": 16.0, "line_width": 2.0})
    res = compositor._generate_pattern(pat, 32, 32, 42)
    assert res.shape == (32, 32)
    assert np.min(res) >= 0.0 and np.max(res) <= 1.0
    
    # hatch
    pat = PatternSpec(kind="hatch", params={"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0})
    res = compositor._generate_pattern(pat, 32, 32, 42)
    assert res.shape == (32, 32)
    assert np.min(res) >= 0.0 and np.max(res) <= 1.0

def test_floyd_steinberg_execution() -> None:
    # Get algorithm
    algo = registry.get("floyd_steinberg")
    assert algo.definition.role.value == "renderer"
    assert algo.definition.output_spec.produces_composition is False
    
    img = Image.new("RGB", (64, 64), color="gray")
    asset = MediaAsset(id="test-asset", checksum="dummy-hash", image=img, substrate=RasterGrid(64, 64))
    store = ArtifactStore()
    ctx = RenderContext(
        input=asset,
        params={"contrast": 1.2, "midpoint": 0.5, "ink_color": "#000000", "paper_color": "#ffffff"},
        composition=None,
        seed=42,
        store=store,
    )
    
    # Run analyze
    algo.analyze(ctx)
    
    # Check final_raster is published
    final_raster = store.get_by_name("final_raster")
    assert isinstance(final_raster.value, Image.Image)
    assert final_raster.value.size == (64, 64)
    
    res = algo.compose(ctx)
    assert res.algorithm_primary_artifact_id == final_raster.id
    assert res.default_composition is None
    
    # Test color validations
    ctx_invalid = RenderContext(
        input=asset,
        params={"contrast": 1.0, "midpoint": 0.5, "ink_color": "not-a-color", "paper_color": "#ffffff"},
        composition=None,
        seed=42,
        store=ArtifactStore(),
    )
    with pytest.raises(ValueError):
        algo.analyze(ctx_invalid)

def test_preset_crud(tmp_path) -> None:
    store = LocalStore(tmp_path)
    
    # List presets (must include built-ins)
    presets = store.list_presets()
    builtins = [p["id"] for p in presets if p.get("is_builtin")]
    assert "wave_halftone" in builtins
    assert "maze_halftone" in builtins
    assert "hatch" in builtins
    
    # Save new user preset
    user_preset = {
        "id": "my_preset",
        "name": "My Custom Preset",
        "description": "Custom test preset",
        "renderer_id": "tonal_analyzer",
        "params": {"contrast": 1.5},
        "composition": {
            "paper_color": {"hex": "#ffffff"},
            "layers": []
        }
    }
    preset_id = store.save_preset(user_preset)
    assert preset_id == "my_preset"
    
    # Load user preset
    loaded = store.get_preset("my_preset")
    assert loaded["name"] == "My Custom Preset"
    assert loaded["is_builtin"] is False
    
    # Verify preset list includes user preset
    all_presets = store.list_presets()
    ids = [p["id"] for p in all_presets]
    assert "my_preset" in ids
    
    # Attempting to delete a built-in must fail
    with pytest.raises(ValueError, match="Cannot delete a built-in preset"):
        store.delete_preset("wave_halftone")
        
    # Attempting to overwrite a built-in must fail
    overwrite_preset = {
        "id": "wave_halftone",
        "name": "Overwritten Wave",
        "renderer_id": "tonal_analyzer",
        "params": {}
    }
    with pytest.raises(ValueError, match="Cannot overwrite a built-in preset"):
        store.save_preset(overwrite_preset)
        
    # Delete user preset
    store.delete_preset("my_preset")
    with pytest.raises(KeyError):
        store.get_preset("my_preset")

    # Test invalid preset ID formats and path traversal attempts
    invalid_preset = {
        "id": "../traversal",
        "name": "Invalid",
        "renderer_id": "tonal_analyzer",
        "params": {}
    }
    with pytest.raises(ValueError):
        store.save_preset(invalid_preset)

    invalid_preset_2 = {
        "id": "my.preset.with.dots",
        "name": "Invalid Dots",
        "renderer_id": "tonal_analyzer",
        "params": {}
    }
    with pytest.raises(ValueError, match="Preset ID must only contain lowercase alphanumeric"):
        store.save_preset(invalid_preset_2)

    with pytest.raises(KeyError):
        store.get_preset("../traversal")

    with pytest.raises(KeyError):
        store.delete_preset("../traversal")

def test_caching_behavior(tmp_path) -> None:
    store = LocalStore(tmp_path)
    asset_checksum = "dummy-asset-checksum"
    
    algo = registry.get("tonal_analyzer")
    
    # Tone map parameters change invalidates key
    key1 = store.get_artifact_cache_key(
        algo.definition.id, algo.definition.version, "tone_map", asset_checksum,
        {"contrast": 1.0, "midpoint": 0.5}, algo.definition.parameters
    )
    key2 = store.get_artifact_cache_key(
        algo.definition.id, algo.definition.version, "tone_map", asset_checksum,
        {"contrast": 1.5, "midpoint": 0.5}, algo.definition.parameters
    )
    assert key1 != key2
    
    # Color or composition changes do NOT affect tone_map keys
    key3 = store.get_artifact_cache_key(
        algo.definition.id, algo.definition.version, "tone_map", asset_checksum,
        {"contrast": 1.0, "midpoint": 0.5, "ink_color": "#ff0000"}, algo.definition.parameters
    )
    assert key1 == key3

def test_multilayer_checksums() -> None:
    # Verify that changing layer rendering order changes final raster checksum
    store = ArtifactStore()
    store.publish("tone_map", ScalarField(RasterGrid(16, 16), np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32), "float32"))
    
    l1 = InkLayerSpec(
        name="layer1",
        color=PaletteColor("#FF0000"),
        role="shadow",
        density_source="tone_map",
        pattern=PatternSpec(kind="wave", params={"frequency": 4.0, "angle_deg": 0.0, "phase": 0.0}),
        priority=0
    )
    l2 = InkLayerSpec(
        name="layer2",
        color=PaletteColor("#0000FF"),
        role="shadow",
        density_source="tone_map",
        pattern=PatternSpec(kind="wave", params={"frequency": 4.0, "angle_deg": 90.0, "phase": 0.0}),
        priority=1
    )
    
    comp1 = Composition(paper_color=PaletteColor("#FFFFFF"), layers=[l1, l2])
    comp2 = Composition(paper_color=PaletteColor("#FFFFFF"), layers=[
        InkLayerSpec(
            name=l1.name, color=l1.color, role=l1.role, density_source=l1.density_source, pattern=l1.pattern, priority=1
        ),
        InkLayerSpec(
            name=l2.name, color=l2.color, role=l2.role, density_source=l2.density_source, pattern=l2.pattern, priority=0
        )
    ])
    
    compositor = Compositor(store)
    img1 = compositor.composite(comp1, 16, 16)
    img2 = compositor.composite(comp2, 16, 16)
    
    checksum1 = hashlib.sha256(img1.tobytes()).hexdigest()
    checksum2 = hashlib.sha256(img2.tobytes()).hexdigest()
    
    # Changing the priority order changes the render order and thus the output image
    assert checksum1 != checksum2

@pytest.fixture
def run_server(tmp_path):
    from colorworks.web.server import ColorworksServer
    
    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", store
    server.shutdown()
    server.server_close()
    t.join()

def test_server_presets_and_cache_reuse(run_server) -> None:
    base_url, store = run_server
    
    # 1. Upload asset
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    
    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=img_bytes,
        headers={"X-Filename": "test.png", "Content-Type": "image/png"}
    )
    with urllib.request.urlopen(req) as resp:
        asset_info = json.loads(resp.read().decode("utf-8"))["asset"]
    asset_id = asset_info["id"]
    
    # Get presets
    req_presets = urllib.request.Request(f"{base_url}/api/presets")
    with urllib.request.urlopen(req_presets) as resp:
        presets_payload = json.loads(resp.read().decode("utf-8"))["presets"]

    wave_preset = next(p for p in presets_payload if p["id"] == "wave_halftone")
    maze_preset = next(p for p in presets_payload if p["id"] == "maze_halftone")
    hatch_preset = next(p for p in presets_payload if p["id"] == "hatch")

    # Render Wave Halftone Preset
    render_payload_1 = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": wave_preset["params"],
        "composition": wave_preset["composition"],
        "seed": 42
    }
    req_render_1 = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(render_payload_1).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_render_1) as resp:
        res1 = json.loads(resp.read().decode("utf-8"))

    # Render Maze Halftone Preset
    render_payload_2 = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": maze_preset["params"],
        "composition": maze_preset["composition"],
        "seed": 42
    }
    req_render_2 = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(render_payload_2).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_render_2) as resp:
        res2 = json.loads(resp.read().decode("utf-8"))

    # Render Hatch Preset
    render_payload_3 = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": hatch_preset["params"],
        "composition": hatch_preset["composition"],
        "seed": 42
    }
    req_render_3 = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(render_payload_3).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_render_3) as resp:
        res3 = json.loads(resp.read().decode("utf-8"))

    # Three different compositions → three different final outputs
    assert res1["output"]["checksum"] != res2["output"]["checksum"]
    assert res2["output"]["checksum"] != res3["output"]["checksum"]
    assert res1["output"]["checksum"] != res3["output"]["checksum"]

    # All three presets share the same analysis params → tone_map artifact
    # should be reused from cache (same ID) across all three renders
    assert res1["artifacts"]["tone_map"]["id"] == res2["artifacts"]["tone_map"]["id"], \
        "tone_map artifact should be reused from cache across composition-only preset changes"
    assert res1["artifacts"]["tone_map"]["id"] == res3["artifacts"]["tone_map"]["id"], \
        "tone_map artifact should be reused from cache across composition-only preset changes"

def test_server_floyd_steinberg_bypasses_compositor(run_server) -> None:
    base_url, store = run_server
    
    # Upload asset
    img = Image.new("RGB", (16, 16), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    
    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=img_bytes,
        headers={"X-Filename": "test.png", "Content-Type": "image/png"}
    )
    with urllib.request.urlopen(req) as resp:
        asset_info = json.loads(resp.read().decode("utf-8"))["asset"]
    asset_id = asset_info["id"]
    
    # Spy on Compositor
    from colorworks.compositor import Compositor
    original_composite = Compositor.composite
    Compositor.composite = MagicMock(side_effect=original_composite)
    
    try:
        render_payload = {
            "asset_id": asset_id,
            "renderer_id": "floyd_steinberg",
            "params": {
                "contrast": 1.2,
                "midpoint": 0.5,
                "ink_color": "#1a1a1a",
                "paper_color": "#f4ebd9"
            },
            "seed": 42
        }
        req_render = urllib.request.Request(
            f"{base_url}/api/render",
            data=json.dumps(render_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_render) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            
        # Verify final_raster is in artifacts list
        assert "final_raster" in res["artifacts"]
        assert len(res["artifacts"]) == 1
        
        # Verify compositor was NOT called or instantiated to composite
        assert Compositor.composite.call_count == 0
    finally:
        Compositor.composite = original_composite

def test_server_different_seeds_invalidate_cache(run_server) -> None:
    base_url, store = run_server
    
    # Upload asset
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    
    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=img_bytes,
        headers={"X-Filename": "test_seed.png", "Content-Type": "image/png"}
    )
    with urllib.request.urlopen(req) as resp:
        asset_info = json.loads(resp.read().decode("utf-8"))["asset"]
    asset_id = asset_info["id"]
    
    # Define a maze composition which depends on the seed
    comp = {
        "paper_color": {"hex": "#f4ebd9", "name": "paper"},
        "layers": [
            {
                "name": "ink",
                "color": {"hex": "#1a1a1a", "name": "ink"},
                "role": "shadow",
                "density_source": "tone_map",
                "pattern": {
                    "kind": "maze",
                    "params": {"scale": 8.0, "line_width": 1.0},
                    "mask_source": None,
                    "coordinates": {
                        "space": "image_px",
                        "seed": None  # Falls back to run seed!
                    }
                }
            }
        ]
    }
    
    # Run 1: seed 42
    payload_42 = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": False
        },
        "composition": comp,
        "seed": 42
    }
    req_42 = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(payload_42).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_42) as resp:
        res_42 = json.loads(resp.read().decode("utf-8"))
        
    # Run 2: seed 43
    payload_43 = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": False
        },
        "composition": comp,
        "seed": 43
    }
    req_43 = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(payload_43).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_43) as resp:
        res_43 = json.loads(resp.read().decode("utf-8"))
        
    # Assert output checksums are different (because different seeds produce different maze structures)
    assert res_42["output"]["checksum"] != res_43["output"]["checksum"]

    # Run 3: seed 42 again (verify cache hit)
    req_42_again = urllib.request.Request(
        f"{base_url}/api/render",
        data=json.dumps(payload_42).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_42_again) as resp:
        res_42_again = json.loads(resp.read().decode("utf-8"))

    # Identical artifact ID (not just checksum) proves the cache was hit —
    # a fresh computation always produces a new artifact ID even if deterministic
    assert res_42["output"]["checksum"] == res_42_again["output"]["checksum"]
    assert res_42["artifacts"]["tone_map"]["id"] == res_42_again["artifacts"]["tone_map"]["id"], \
        "tone_map artifact ID must match on cache hit (same ID, not just same checksum)"

