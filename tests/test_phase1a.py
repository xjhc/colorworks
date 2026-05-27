from __future__ import annotations

import hashlib
import io
import json
import pytest
import numpy as np
from PIL import Image

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
    PatternKindDef,
    ParameterDef,
    ParameterType,
)
from colorworks.algorithms import registry, MediaAsset, RenderContext
from colorworks.algorithms.tonal_analyzer import TonalAnalyzer, to_gray, remap_tone
from colorworks.compositor import Compositor
from colorworks.recipe import Recipe, load_recipe, save_recipe
from colorworks.storage import LocalStore


def test_tonal_analyzer_basic_execution() -> None:
    # Create a 64x64 source image
    img = Image.new("RGB", (64, 64), color="gray")
    asset = MediaAsset(
        id="test-asset",
        checksum="dummy-hash",
        image=img,
        substrate=RasterGrid(64, 64),
    )

    store = ArtifactStore()
    ctx = RenderContext(
        input=asset,
        params={"contrast": 1.2, "midpoint": 0.5, "preserve_edges": True, "edge_threshold": 0.15},
        composition=None,
        seed=42,
        store=store,
    )

    analyzer = TonalAnalyzer()
    
    # Run analyze
    analyzer.analyze(ctx)

    # Check tone_map and edge_mask are published
    tone_map = store.get_by_name("tone_map")
    assert isinstance(tone_map.value, ScalarField)
    assert tone_map.value.data.shape == (64, 64)

    edge_mask = store.get_by_name("edge_mask")
    assert isinstance(edge_mask.value, BinaryMask)
    assert edge_mask.value.data.shape == (64, 64)

    # Verify default composition
    res = analyzer.compose(ctx)
    assert res.algorithm_primary_artifact_id == tone_map.id
    assert res.default_composition is not None
    assert len(res.default_composition.layers) == 1
    assert res.default_composition.layers[0].pattern.kind == "wave"


def test_compositor_generation_and_density_boost() -> None:
    store = ArtifactStore()
    # Let tone_map be 0.5 everywhere
    store.publish("tone_map", ScalarField(RasterGrid(16, 16), np.ones((16, 16), dtype=np.float32) * 0.5, "float32"))
    # Let edge_mask have edges at y=0,1 (top 2 rows)
    edges = np.zeros((16, 16), dtype=bool)
    edges[0:2, :] = True
    store.publish("edge_mask", BinaryMask(RasterGrid(16, 16), edges))

    # Compose: paper is white, ink is black
    comp = Composition(
        paper_color=PaletteColor("#FFFFFF"),
        layers=[
            InkLayerSpec(
                name="ink",
                color=PaletteColor("#000000"),
                role="shadow",
                density_source="tone_map",
                pattern=PatternSpec(
                    kind="wave",
                    params={"frequency": 10.0, "angle_deg": 0.0, "phase": 0.0},
                    mask_source="edge_mask",
                    coordinates=PatternCoordinateSpec(space="image_px")
                )
            )
        ]
    )

    compositor = Compositor(store)
    img = compositor.composite(comp, 16, 16)
    arr = np.asarray(img.convert("L"))

    # In top 2 rows, edge_mask is True -> density boosted by 1.0 -> final density is 1.0
    # Since density is 1.0, and pattern is wave (max 1.0), density_final >= P is always True,
    # so top 2 rows should be solid ink (black/0)
    assert np.all(arr[0:2, :] == 0)


def test_recipe_compat_and_tonal_round_trip(tmp_path) -> None:
    # Test new recipe round-trip
    params = {
        "contrast": 1.1,
        "midpoint": 0.45,
        "preserve_edges": True,
        "edge_threshold": 0.20,
    }
    comp = {
        "paper_color": {"hex": "#f4ebd9", "name": "paper"},
        "layers": [
            {
                "name": "ink",
                "color": {"hex": "#1a1a1a", "name": "ink"},
                "role": "shadow",
                "density_source": "tone_map",
                "pattern": {
                    "kind": "wave",
                    "params": {"frequency": 12.0, "angle_deg": 90.0, "phase": 0.5},
                    "mask_source": "edge_mask",
                    "coordinates": {"space": "image_px"}
                }
            }
        ]
    }
    
    recipe = Recipe.create(
        name="Tonal Wave Recipe",
        asset_id="some-asset",
        asset_checksum="asset-checksum",
        params=params,
        composition=comp,
        renderer_id="tonal_analyzer"
    )

    recipe_path = tmp_path / "tonal_recipe.json"
    save_recipe(recipe_path, recipe)

    # Load and assert
    loaded = load_recipe(recipe_path)
    assert loaded.renderer_id == "tonal_analyzer"
    assert loaded.params["contrast"] == 1.1
    assert loaded.composition["layers"][0]["pattern"]["params"]["frequency"] == 12.0


def test_second_pattern_extensibility() -> None:
    # 1. Define custom pattern kind Def
    custom_pattern_def = PatternKindDef(
        kind="test_stripe",
        name="Test Stripe",
        description="A dynamic stripe pattern.",
        parameters=[
            ParameterDef("stripe_width", "Stripe Width", ParameterType.FLOAT, default=10.0),
        ]
    )
    
    try:
        # 2. Register it in global registry
        registry.register_pattern(custom_pattern_def)
        
        # 3. Register a pattern generator callback
        def stripe_generator(pattern, width, height, run_seed):
            w = float(pattern.params.get("stripe_width", 10.0))
            # Generate simple horizontal stripes
            y = np.arange(height)[:, None]
            stripes = (y % (2 * w) < w).astype(np.float32)
            return np.tile(stripes, (1, width))

        registry.register_pattern_generator("test_stripe", stripe_generator)

        # 4. Run compositor with this new pattern kind
        store = ArtifactStore()
        store.publish("tone_map", ScalarField(RasterGrid(16, 16), np.ones((16, 16), dtype=np.float32) * 0.8, "float32"))
        
        comp = Composition(
            paper_color=PaletteColor("#FFFFFF"),
            layers=[
                InkLayerSpec(
                    name="ink",
                    color=PaletteColor("#000000"),
                    role="shadow",
                    density_source="tone_map",
                    pattern=PatternSpec(
                        kind="test_stripe",
                        params={"stripe_width": 4.0},
                        coordinates=PatternCoordinateSpec(space="image_px")
                    )
                )
            ]
        )
        compositor = Compositor(store)
        img = compositor.composite(comp, 16, 16)
        
        # Verify stripes produce multiple colors
        colors = {tuple(pixel) for pixel in np.asarray(img).reshape(-1, 3)}
        assert len(colors) > 1
    finally:
        registry.unregister_pattern("test_stripe")


def test_per_artifact_caching_scoping(tmp_path) -> None:
    store = LocalStore(tmp_path)
    asset_checksum = "dummy-asset-checksum"
    
    # Tone map parameters
    contrast_1 = 1.0
    midpoint_1 = 0.5
    
    # Edge parameters
    preserve_1 = True
    edge_thresh_1 = 0.15
    
    # Compute keys
    tone_key_1 = store.get_tone_map_cache_key(asset_checksum, contrast_1, midpoint_1)
    edge_key_1 = store.get_edge_mask_cache_key(asset_checksum, preserve_1, edge_thresh_1)
    
    # Mock save artifacts in store
    tone_data_1 = np.ones((8, 8), dtype=np.float32) * 0.5
    edge_data_1 = np.zeros((8, 8), dtype=bool)
    
    tone_checksum_1 = hashlib.sha256(tone_data_1.tobytes()).hexdigest()
    edge_checksum_1 = hashlib.sha256(edge_data_1.tobytes()).hexdigest()
    
    # Save cache
    store.save_cached_artifact(tone_key_1, tone_data_1, {"id": "tone1", "name": "tone_map", "type": "scalar_field", "checksum": tone_checksum_1})
    store.save_cached_artifact(edge_key_1, edge_data_1, {"id": "edge1", "name": "edge_mask", "type": "binary_mask", "checksum": edge_checksum_1})
    
    # 2. Change wave params (frequency, angle, phase) or ink color, but keep contrast, midpoint, edge params the same:
    # Check if tone_map and edge_mask keys hit cache!
    tone_key_2 = store.get_tone_map_cache_key(asset_checksum, contrast_1, midpoint_1)
    edge_key_2 = store.get_edge_mask_cache_key(asset_checksum, preserve_1, edge_thresh_1)
    
    assert tone_key_1 == tone_key_2
    assert edge_key_1 == edge_key_2
    
    # 3. Change tone parameters (contrast) and verify tone_map key changes:
    contrast_2 = 1.5
    tone_key_3 = store.get_tone_map_cache_key(asset_checksum, contrast_2, midpoint_1)
    assert tone_key_1 != tone_key_3
    
    # Changing contrast does NOT change edge_mask key!
    edge_key_3 = store.get_edge_mask_cache_key(asset_checksum, preserve_1, edge_thresh_1)
    assert edge_key_1 == edge_key_3
    
    # 4. Change edge parameters (edge_threshold) and verify edge_mask key changes:
    edge_thresh_2 = 0.25
    edge_key_4 = store.get_edge_mask_cache_key(asset_checksum, preserve_1, edge_thresh_2)
    assert edge_key_1 != edge_key_4
    
    # Changing edge threshold does NOT change tone_map key!
    tone_key_4 = store.get_tone_map_cache_key(asset_checksum, contrast_1, midpoint_1)
    assert tone_key_1 == tone_key_4


def test_cached_vs_uncached_tonal_renders_are_identical(tmp_path) -> None:
    # 1. Setup local store and a dummy asset
    store = LocalStore(tmp_path)
    # We write a deterministic grid
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    record = store.save_asset(filename="test.png", content=buf.getvalue())
    
    comp = Composition(
        paper_color=PaletteColor("#f4ebd9"),
        layers=[
            InkLayerSpec(
                name="ink",
                color=PaletteColor("#1a1a1a"),
                role="shadow",
                density_source="tone_map",
                pattern=PatternSpec(
                    kind="wave",
                    params={"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                    mask_source="edge_mask",
                    coordinates=PatternCoordinateSpec(space="image_px")
                )
            )
        ]
    )

    algo = registry.get("tonal_analyzer")
    params = {"contrast": 1.2, "midpoint": 0.5, "preserve_edges": True, "edge_threshold": 0.15}
    substrate = RasterGrid(record.width, record.height)
    asset = MediaAsset(id=record.id, checksum=record.checksum, image=img.copy(), substrate=substrate)

    # First run (Cache MISS)
    store_1 = ArtifactStore(output_dir=store.artifacts_dir)
    ctx_1 = RenderContext(input=asset, params=params, composition=None, seed=42, store=store_1)
    
    algo.analyze(ctx_1)
    
    for name in algo.produced_in_analyze:
        art = store_1.get_by_name(name)
        if name == "tone_map":
            key = store.get_tone_map_cache_key(record.checksum, params["contrast"], params["midpoint"])
        else:
            key = store.get_edge_mask_cache_key(record.checksum, params["preserve_edges"], params["edge_threshold"])
        store.save_cached_artifact(key, art.value.data, {"id": art.id, "name": name, "type": art.type, "checksum": art.checksum})

    res_1 = algo.compose(ctx_1)
    
    compositor_1 = Compositor(store_1)
    final_img_1 = compositor_1.composite(comp, record.width, record.height, ctx_1.seed)
    checksum_1 = hashlib.sha256(final_img_1.tobytes()).hexdigest()

    # Second run (Cache HIT)
    store_2 = ArtifactStore(output_dir=store.artifacts_dir)
    ctx_2 = RenderContext(input=asset, params=params, composition=None, seed=42, store=store_2)
    
    for name in algo.produced_in_analyze:
        if name == "tone_map":
            key = store.get_tone_map_cache_key(record.checksum, params["contrast"], params["midpoint"])
        else:
            key = store.get_edge_mask_cache_key(record.checksum, params["preserve_edges"], params["edge_threshold"])
        
        arr, meta = store.get_cached_artifact(key)
        if name == "tone_map":
            field = ScalarField(substrate, arr, "float32")
            pub_id = store_2.publish(name, field)
            ctx_2.working.put("tone_id", pub_id)
        else:
            mask = BinaryMask(substrate, arr)
            store_2.publish(name, mask)

    res_2 = algo.compose(ctx_2)
    
    compositor_2 = Compositor(store_2)
    final_img_2 = compositor_2.composite(comp, record.width, record.height, ctx_2.seed)
    checksum_2 = hashlib.sha256(final_img_2.tobytes()).hexdigest()

    # Verify they are identical
    assert checksum_1 == checksum_2
    assert np.array_equal(store_1.get_by_name("tone_map").value.data, store_2.get_by_name("tone_map").value.data)


@pytest.fixture(autouse=True)
def clean_registry():
    state = registry.save_state()
    yield
    registry.restore_state(state)


@pytest.fixture
def run_server(tmp_path):
    import threading
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


def test_server_cached_vs_uncached_renders_are_identical(run_server) -> None:
    base_url, store = run_server
    import urllib.request
    from unittest.mock import MagicMock

    # 1. Upload an asset
    img = Image.new("RGB", (32, 32), color="gray")
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    req = urllib.request.Request(
        f"{base_url}/api/assets",
        data=img_bytes,
        headers={"X-Filename": "test.png", "Content-Type": "image/png"}
    )
    with urllib.request.urlopen(req) as resp:
        asset_info = json.loads(resp.read().decode("utf-8"))["asset"]
    asset_id = asset_info["id"]

    # Render params and composition
    render_payload = {
        "asset_id": asset_id,
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": True,
            "edge_threshold": 0.15
        },
        "composition": {
            "paper_color": {"hex": "#f4ebd9", "name": "paper"},
            "layers": [
                {
                    "name": "ink",
                    "color": {"hex": "#1a1a1a", "name": "ink"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "wave",
                        "params": {"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                        "mask_source": "edge_mask",
                        "coordinates": {"space": "image_px"}
                    }
                }
            ]
        },
        "seed": 42
    }

    # Spy on TonalAnalyzer.analyze on the registry instance
    algo = registry.get("tonal_analyzer")
    original_analyze = algo.analyze
    algo.analyze = MagicMock(side_effect=original_analyze)

    try:
        # First POST request (Cache MISS)
        req2 = urllib.request.Request(
            f"{base_url}/api/render",
            data=json.dumps(render_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req2) as resp:
            res1 = json.loads(resp.read().decode("utf-8"))

        # Second POST request (Cache HIT)
        with urllib.request.urlopen(req2) as resp:
            res2 = json.loads(resp.read().decode("utf-8"))

        # Assert outputs are identical
        assert res1["output"]["checksum"] == res2["output"]["checksum"]
        assert res1["artifacts"]["tone_map"]["id"] == res2["artifacts"]["tone_map"]["id"]
        assert res1["artifacts"]["edge_mask"]["id"] == res2["artifacts"]["edge_mask"]["id"]

        # Assert TonalAnalyzer.analyze was called exactly once (proving cache was used on request 2!)
        assert algo.analyze.call_count == 1
    finally:
        algo.analyze = original_analyze

