from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from colorworks.algorithms import MediaAsset, RenderContext, registry
from colorworks.compositor import Compositor
from colorworks.domain import (
    ArtifactStore,
    BinaryMask,
    Composition,
    InkLayerSpec,
    PaletteColor,
    PatternSpec,
    RasterGrid,
    ScalarField,
    StrokeSet,
    VectorField2D,
)
from colorworks.storage import LocalStore

import colorworks.algorithms.pattern_catalog
import colorworks.algorithms.structure_analyzer
import colorworks.algorithms.tonal_analyzer


def gradient_image(width: int = 48, height: int = 48) -> Image.Image:
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    rgb = np.stack([ramp, ramp, ramp], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@pytest.fixture
def run_server(tmp_path):
    from colorworks.web.server import ColorworksServer

    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", store
    server.shutdown()
    server.server_close()
    thread.join()


def upload_asset(base_url: str, image: Image.Image, filename: str = "source.png") -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    request = urllib.request.Request(
        f"{base_url}/api/assets",
        data=buffer.getvalue(),
        headers={"X-Filename": filename, "Content-Type": "image/png"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))["asset"]


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def structure_params() -> dict:
    return {
        "contrast": 1.0,
        "midpoint": 0.5,
        "sigma": 2.0,
        "etf_iterations": 1,
        "etf_radius": 3,
    }


def hatch_composition(orientation_source: str | None = "orientation_field") -> dict:
    return {
        "paper_color": {"hex": "#f4ebd9"},
        "layers": [
            {
                "name": "flow hatch",
                "color": {"hex": "#1a1a1a"},
                "role": "shadow",
                "density_source": "tone_map",
                "pattern": {
                    "kind": "hatch",
                    "params": {"frequency": 10.0, "angle_deg": 0.0, "phase": 0.0},
                    "orientation_source": orientation_source,
                    "mask_source": None,
                    "coordinates": {"space": "image_px"},
                },
                "threshold": None,
                "blend_mode": "normal",
                "opacity": 1.0,
                "priority": 0,
            }
        ],
    }


def test_structure_analyzer_produces_tensor_orientation_and_default_hatch() -> None:
    algo = registry.get("structure_analyzer")
    image = gradient_image(40, 32)
    asset = MediaAsset(
        id="asset",
        checksum="checksum",
        image=image,
        substrate=RasterGrid(image.width, image.height),
    )
    ctx = RenderContext(
        input=asset,
        params=structure_params(),
        composition=None,
        seed=17,
        store=ArtifactStore(),
    )

    algo.analyze(ctx)
    tone = ctx.store.get_by_name("tone_map")
    tensor = ctx.store.get_by_name("structure_tensor")
    orientation = ctx.store.get_by_name("orientation_field")

    assert tone.type == "scalar_field"
    assert tensor.type == "structure_tensor_field"
    assert orientation.type == "vector_field_2d"
    assert orientation.value.data.shape == (32, 40, 2)
    assert np.isfinite(orientation.value.data).all()
    norms = np.linalg.norm(orientation.value.data, axis=-1)
    assert np.nanmax(norms) <= 1.00001

    result = algo.compose(ctx)
    assert result.default_composition is not None
    layer = result.default_composition.layers[0]
    assert layer.pattern.kind == "hatch"
    assert layer.pattern.orientation_source == "orientation_field"


def test_compositor_validates_orientation_sources_and_allows_fallback_angle() -> None:
    substrate = RasterGrid(32, 32)
    store = ArtifactStore()
    store.publish("tone_map", ScalarField(substrate, np.full((32, 32), 0.2, dtype=np.float32), "float32"))
    store.publish("edge_mask", BinaryMask(substrate, np.zeros((32, 32), dtype=bool)))
    vertical = np.zeros((32, 32, 2), dtype=np.float32)
    vertical[:, :, 1] = 1.0
    store.publish("orientation_field", VectorField2D(substrate, vertical, is_bidirectional=True))

    def make_layer(orientation_source: str | None) -> InkLayerSpec:
        return InkLayerSpec(
            name="hatch",
            color=PaletteColor("#1a1a1a"),
            role="shadow",
            density_source="tone_map",
            pattern=PatternSpec(
                kind="hatch",
                params={"frequency": 8.0, "angle_deg": 0.0, "phase": 0.0},
                orientation_source=orientation_source,
            ),
        )

    compositor = Compositor(store)
    valid_sets = compositor.build_stroke_set(
        Composition(PaletteColor("#f4ebd9"), [make_layer("orientation_field")]),
        32,
        32,
        run_seed=3,
    )
    assert len(valid_sets) == 1
    assert isinstance(valid_sets[0][1], StrokeSet)
    assert len(valid_sets[0][1].strokes) > 0

    fallback_sets = compositor.build_stroke_set(
        Composition(PaletteColor("#f4ebd9"), [make_layer(None)]),
        32,
        32,
        run_seed=3,
    )
    assert len(fallback_sets[0][1].strokes) > 0

    with pytest.raises(ValueError, match="not found"):
        compositor.build_stroke_set(
            Composition(PaletteColor("#f4ebd9"), [make_layer("missing_orientation")]),
            32,
            32,
        )

    with pytest.raises(ValueError, match="not a VectorField2D"):
        compositor.build_stroke_set(
            Composition(PaletteColor("#f4ebd9"), [make_layer("tone_map")]),
            32,
            32,
        )


def test_server_structure_views_svg_export_and_recipe_round_trip(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(), "phase2.png")

    payload = {
        "asset_id": asset["id"],
        "renderer_id": "structure_analyzer",
        "params": structure_params(),
        "composition": hatch_composition("orientation_field"),
        "seed": 23,
    }
    render_result = post_json(f"{base_url}/api/render", payload)

    assert set(render_result["artifacts"]) == {"tone_map", "structure_tensor", "orientation_field"}
    orientation_url = render_result["artifacts"]["orientation_field"]["url"]
    for view in ("orientation_hsv", "glyph_field"):
        with urllib.request.urlopen(f"{base_url}{orientation_url}?view={view}") as response:
            body = response.read()
            assert response.headers["Content-Type"] == "image/png"
            assert len(body) > 100

    svg_request = urllib.request.Request(
        f"{base_url}/api/export/svg",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(svg_request) as response:
        svg_bytes = response.read()
        assert "image/svg+xml" in response.headers["Content-Type"]

    root = ET.fromstring(svg_bytes)
    assert root.tag.endswith("svg")
    assert any(el.tag.endswith("line") or el.tag.endswith("path") for el in root.iter())

    recipe_payload = {
        "name": "Structure recipe",
        "asset_id": asset["id"],
        "renderer_id": "structure_analyzer",
        "params": structure_params(),
        "composition": hatch_composition("orientation_field"),
    }
    recipe = post_json(f"{base_url}/api/recipes", recipe_payload)
    loaded_recipe = get_json(f"{base_url}/api/recipes/{recipe['id']}")
    assert loaded_recipe["renderer_id"] == "structure_analyzer"
    assert loaded_recipe["composition"]["layers"][0]["pattern"]["orientation_source"] == "orientation_field"


def test_svg_export_rejects_compositions_without_stroke_layers(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(), "phase2-no-strokes.png")
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "structure_analyzer",
        "params": structure_params(),
        "composition": {
            "paper_color": {"hex": "#f4ebd9"},
            "layers": [
                {
                    "name": "wave",
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
                }
            ],
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/export/svg",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode("utf-8")
    assert "no hatch/crosshatch" in body


def test_direct_renderer_recipe_round_trip_does_not_store_composition(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(), "phase2-direct-renderer.png")

    recipe_payload = {
        "name": "Direct renderer recipe",
        "asset_id": asset["id"],
        "renderer_id": "floyd_steinberg",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "ink_color": "#1a1a1a",
            "paper_color": "#f4ebd9",
        },
        "composition": hatch_composition("orientation_field"),
    }

    recipe = post_json(f"{base_url}/api/recipes", recipe_payload)
    loaded_recipe = get_json(f"{base_url}/api/recipes/{recipe['id']}")

    assert loaded_recipe["renderer_id"] == "floyd_steinberg"
    assert "composition" not in loaded_recipe


def test_server_structure_cache_reuses_analyze_stage(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(), "phase2-cache.png")
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "structure_analyzer",
        "params": structure_params(),
        "composition": hatch_composition("orientation_field"),
        "seed": 51,
    }

    first = post_json(f"{base_url}/api/render", payload)
    second = post_json(f"{base_url}/api/render", payload)
    # Identical artifact IDs prove the second request was served from cache,
    # not freshly computed (a fresh render allocates new artifact IDs).
    assert first["output"]["checksum"] == second["output"]["checksum"]
    assert first["artifacts"]["orientation_field"]["id"] == second["artifacts"]["orientation_field"]["id"]
