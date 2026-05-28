from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colorworks.storage import LocalStore
from colorworks.web.server import ColorworksServer


def gradient_image(width: int = 48, height: int = 48) -> Image.Image:
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    rgb = np.stack([ramp, ramp, ramp], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@pytest.fixture
def run_server(tmp_path):
    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", store
    server.shutdown()
    server.server_close()
    t.join(timeout=3)


def upload_asset(base_url: str, image: Image.Image, filename: str = "test.png") -> dict:
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


def read_sse(url: str, timeout: float = 5.0) -> list[dict]:
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


def test_png_export_bounds_and_format_synchronous(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(width=64, height=32))
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "ordered_bayer",
        "params": {"matrix_size": 8, "threshold": 0.5, "contrast": 1.0},
    }
    res = post_json(f"{base_url}/api/render", payload)
    output_url = res["output"]["url"]

    # Fetch output PNG and verify dimensions/mode
    with urllib.request.urlopen(f"{base_url}{output_url}") as r:
        img_data = r.read()
    img = Image.open(io.BytesIO(img_data))
    assert img.format == "PNG"
    assert img.size == (64, 32)


def test_png_export_iterative_and_cancellation(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(width=32, height=32))
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "cvt_stippling",
        "params": {"n_dots": 20, "max_iterations": 10, "convergence_threshold": 0.0, "dot_radius": 1.5},
    }

    # Submit run
    run = post_json(f"{base_url}/api/preview_runs", payload)
    run_id = run["id"]

    # Read SSE events (we expect to see at least started/iteration/completed)
    events = read_sse(f"{base_url}/api/preview_runs/{run_id}/events")
    assert len(events) > 0
    assert events[-1]["kind"] == "completed"

    final_artifact_id = events[-1]["final_artifact_id"]
    # Verify we can download this final PNG artifact and it has correct dimensions
    with urllib.request.urlopen(f"{base_url}/api/artifacts/{final_artifact_id}") as r:
        img_data = r.read()
    img = Image.open(io.BytesIO(img_data))
    assert img.format == "PNG"
    assert img.size == (32, 32)

    # Now test client-side cancellation via DELETE request
    run2 = post_json(f"{base_url}/api/preview_runs", payload)
    run_id2 = run2["id"]

    # Send DELETE immediately to cancel
    req = urllib.request.Request(
        f"{base_url}/api/preview_runs/{run_id2}",
        method="DELETE"
    )
    with urllib.request.urlopen(req) as r:
        del_resp = json.loads(r.read())
    assert del_resp["status"] in ("cancelled", "not_found")


def test_svg_export_validation_rejects_without_composition_or_strokes(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image())

    # Case 1: renderer_id doesn't use composition (returns None composition)
    payload_no_comp = {
        "asset_id": asset["id"],
        "renderer_id": "ordered_bayer",
        "params": {"matrix_size": 8, "threshold": 0.5, "contrast": 1.0},
    }
    req = urllib.request.Request(
        f"{base_url}/api/export/svg",
        data=json.dumps(payload_no_comp).encode(),
        headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode()
    assert "requires a composited pipeline" in body or "unregistered renderer_id" in body

    # Case 2: composition has no stroke layers
    payload_no_strokes = {
        "asset_id": asset["id"],
        "renderer_id": "tonal_analyzer",
        "params": {"contrast": 1.0, "smooth_radius": 2.0},
        "composition": {
            "paper_color": {"hex": "#ffffff"},
            "layers": [
                {
                    "name": "solid_bg",
                    "color": {"hex": "#000000"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "solid",
                        "params": {},
                        "coordinates": {"space": "image_px"},
                    },
                    "blend_mode": "normal",
                    "opacity": 1.0,
                    "priority": 0,
                }
            ],
        },
    }
    req2 = urllib.request.Request(
        f"{base_url}/api/export/svg",
        data=json.dumps(payload_no_strokes).encode(),
        headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req2)
    assert exc_info.value.code == 400
    body2 = exc_info.value.read().decode()
    assert "no hatch/crosshatch stroke layers" in body2


def test_svg_export_independent_of_prior_render(run_server) -> None:
    base_url, store = run_server
    asset = upload_asset(base_url, gradient_image(width=100, height=80))

    # Valid composition with a hatch layer (which has strokes)
    payload = {
        "asset_id": asset["id"],
        "renderer_id": "tonal_analyzer",
        "params": {"contrast": 1.0, "smooth_radius": 1.0},
        "composition": {
            "paper_color": {"hex": "#ffebd8"},
            "layers": [
                {
                    "name": "hatch_layer",
                    "color": {"hex": "#000011"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "hatch",
                        "params": {"frequency": 10.0, "angle_deg": 30.0, "phase": 0.0},
                        "coordinates": {"space": "image_px"},
                    },
                    "blend_mode": "normal",
                    "opacity": 0.8,
                    "priority": 0,
                }
            ],
        },
    }

    # Fetch SVG export immediately without calling /api/render first
    req = urllib.request.Request(
        f"{base_url}/api/export/svg",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        assert r.headers["Content-Type"] == "image/svg+xml"
        svg_content = r.read().decode()

    # Parse SVG and verify root attributes (width, height, viewBox)
    root = ET.fromstring(svg_content)
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "100"
    assert root.attrib["height"] == "80"
    assert root.attrib["viewBox"] == "0 0 100 80"

    # Verify background paper rect exists and matches hex
    bg_rect = root.find("{http://www.w3.org/2000/svg}rect")
    assert bg_rect is not None
    assert bg_rect.attrib["fill"] == "#ffebd8"
