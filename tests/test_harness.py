from __future__ import annotations

import io
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from colorworks.algorithms.fixtures import FIXTURES
from colorworks.algorithms.comparison_harness import run_harness, compute_file_sha256
from colorworks.web.server import ColorworksServer
from colorworks.storage import LocalStore


def test_fixtures_deterministic() -> None:
    # Prove that all fixtures are generated deterministically
    for name, generator_fn in FIXTURES.items():
        img1 = generator_fn()
        img2 = generator_fn()

        # Check matching mode and size
        assert img1.mode == img2.mode
        assert img1.size == img2.size

        # Check pixel content identity
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), f"Fixture {name} is not deterministic"


def test_comparison_harness_isolation_and_manifest(tmp_path: Path) -> None:
    # Run harness with tmp_path to isolate output
    table_str = run_harness(output_dir=tmp_path)

    assert table_str is not None
    assert "| Algorithm |" in table_str

    # Check that manifest JSON exists
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert isinstance(manifest, list)
    from colorworks.presets import BUILTIN_PRESETS
    core_algos = ["floyd_steinberg", "pang_halftoning", "cvt_stippling", "dbs", "saed"]
    expected_count = len(FIXTURES) * (len(core_algos) + len(BUILTIN_PRESETS))
    assert len(manifest) == expected_count

    expected_fixture_names = set(FIXTURES.keys())
    expected_run_ids = set(core_algos) | {p["id"] for p in BUILTIN_PRESETS}

    # Verify all expected runs are in the manifest
    seen_combinations = set()

    for entry in manifest:
        assert "fixture_name" in entry
        assert "fixture_checksum" in entry
        assert "run_id" in entry
        assert "kind" in entry
        assert "algorithm_id" in entry
        assert "preset_id" in entry
        assert "params" in entry
        assert "source_path" in entry
        assert "source_url" in entry
        assert "output_path" in entry
        assert "output_url" in entry
        assert "checksum" in entry
        assert "sha256" in entry
        assert "runtime_ms" in entry
        assert "dimensions" in entry
        assert "width" in entry
        assert "height" in entry
        assert "metrics" in entry
        assert "mse" in entry
        assert "mean_intensity" in entry

        fixture_name = entry["fixture_name"]
        fixture_checksum = entry["fixture_checksum"]
        run_id = entry["run_id"]
        kind = entry["kind"]
        algo_id = entry["algorithm_id"]
        preset_id = entry["preset_id"]

        assert fixture_name in expected_fixture_names
        assert run_id in expected_run_ids
        assert kind in {"algorithm", "preset"}
        assert len(fixture_checksum) == 64  # SHA-256 hex string length

        if kind == "preset":
            preset_data = next(p for p in BUILTIN_PRESETS if p["id"] == run_id)
            assert algo_id == preset_data["renderer_id"]
            assert preset_id == run_id

            # Verify preset metadata fields exist and match
            assert "preset_metadata" in entry
            meta = entry["preset_metadata"]
            assert meta["description"] == preset_data["description"]
            assert meta["recommended_for"] == preset_data["recommended_for"]
            assert meta["style_tags"] == preset_data["style_tags"]
            assert meta["sort_order"] == preset_data["sort_order"]
        else:
            assert algo_id == run_id
            assert preset_id is None

        # Dimensions check
        dims = entry["dimensions"]
        assert len(dims) == 2
        if fixture_name == "icon":
            assert dims == [48, 48]
        else:
            assert dims == [64, 64]

        # Verify output path and source path are isolated under tmp_path
        out_path = Path(entry["output_path"])
        assert out_path.exists()
        assert out_path.parent.resolve() == tmp_path.resolve()

        source_path = Path(entry["source_path"])
        assert source_path.exists()
        assert source_path.parent.resolve() == tmp_path.resolve()

        # Checksum check
        checksum = entry["checksum"]
        computed_checksum = compute_file_sha256(out_path)
        assert checksum == computed_checksum, "Checksum mismatch in manifest"

        # Metrics check
        metrics = entry["metrics"]
        assert "mse" in metrics
        assert "mean_intensity" in metrics
        assert metrics["mse"] >= 0.0
        assert 0.0 <= metrics["mean_intensity"] <= 1.0

        # Runtime check
        assert entry["runtime_ms"] >= 0.0

        # Output image is non-empty and not a solid fallback blank image
        out_img = Image.open(out_path)
        assert out_img.size == tuple(dims)

        out_arr = np.array(out_img.convert("L"))
        # Verify standard deviation > 0.0 to prove it's not a solid blank/empty image
        assert np.std(out_arr) > 0.0, f"Output image for {fixture_name} using {run_id} is a solid color"

        seen_combinations.add((fixture_name, run_id))

    # Check that we covered all combinations
    assert len(seen_combinations) == expected_count


def test_comparison_server_endpoints(tmp_path: Path) -> None:
    # Set up server and directory
    store = LocalStore(tmp_path)
    server = ColorworksServer(("127.0.0.1", 0), store)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # Run comparison harness to generate files in store comparison latest directory
        comp_dir = tmp_path / "comparison" / "latest"
        run_harness(output_dir=comp_dir)

        # Test 1: Fetch manifest via endpoint
        manifest_url = f"{base_url}/api/comparison/manifest"
        with urllib.request.urlopen(manifest_url) as r:
            assert r.status == 200
            manifest_data = json.loads(r.read())

        assert isinstance(manifest_data, list)
        from colorworks.presets import BUILTIN_PRESETS
        core_algos = ["floyd_steinberg", "pang_halftoning", "cvt_stippling", "dbs", "saed"]
        expected_count = len(FIXTURES) * (len(core_algos) + len(BUILTIN_PRESETS))
        assert len(manifest_data) == expected_count

        # Check entries
        entry = manifest_data[0]
        assert "fixture_name" in entry
        assert "fixture_checksum" in entry
        assert "run_id" in entry
        assert "kind" in entry
        assert "algorithm_id" in entry
        assert "preset_id" in entry
        assert "params" in entry
        assert "source_path" in entry
        assert "source_url" in entry
        assert "output_path" in entry
        assert "output_url" in entry
        assert "checksum" in entry
        assert "sha256" in entry
        assert "runtime_ms" in entry
        assert "dimensions" in entry
        assert "width" in entry
        assert "height" in entry
        assert "metrics" in entry
        assert "mse" in entry
        assert "mean_intensity" in entry

        # Test 2: Prove output and source images are reachable via server
        for name in ["portrait", "landscape", "icon"]:
            source_img_url = f"{base_url}/api/comparison/images/source_{name}.png"
            with urllib.request.urlopen(source_img_url) as r:
                assert r.status == 200
                img_data = r.read()
                assert len(img_data) > 0
                img = Image.open(io.BytesIO(img_data))
                assert img.format == "PNG"

            output_img_url = f"{base_url}/api/comparison/images/{name}_floyd_steinberg.png"
            with urllib.request.urlopen(output_img_url) as r:
                assert r.status == 200
                img_data = r.read()
                assert len(img_data) > 0
                img = Image.open(io.BytesIO(img_data))
                assert img.format == "PNG"

        # Test 3: Path traversal rejection and filename validation
        bad_urls = [
            f"{base_url}/api/comparison/images/../../artifacts/index.json",
            f"{base_url}/api/comparison/images/..%2F..%2Fartifacts%2Findex.json",
            f"{base_url}/api/comparison/images/subdir/../../manifest.json",
            f"{base_url}/api/comparison/images/..%2F%2E%2E%2F..%2Fetc%2Fpasswd",
            f"{base_url}/api/comparison/images/manifest.json",
            f"{base_url}/api/comparison/images/nonexistent.png",
        ]
        for bad_url in bad_urls:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(bad_url)
            assert exc_info.value.code == 404

    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=3)


def test_builtin_presets_valid_and_renderable() -> None:
    from colorworks.presets import BUILTIN_PRESETS
    from colorworks.domain import RasterGrid, ArtifactStore
    from colorworks.algorithms import MediaAsset, RenderContext, registry
    from colorworks.algorithms.comparison_harness import RegistryCalibrationAccessor, parse_composition
    import asyncio

    # Dummy image
    img = Image.new("L", (16, 16), color=128)
    substrate = RasterGrid(16, 16)
    asset = MediaAsset(
        id="dummy",
        checksum="dummy_checksum",
        image=img,
        substrate=substrate,
    )

    for preset in BUILTIN_PRESETS:
        # Check metadata
        assert "description" in preset and isinstance(preset["description"], str)
        assert len(preset["description"]) > 0
        assert "recommended_for" in preset and isinstance(preset["recommended_for"], list)
        assert len(preset["recommended_for"]) > 0
        assert "style_tags" in preset and isinstance(preset["style_tags"], list)
        assert len(preset["style_tags"]) > 0
        assert "sort_order" in preset and isinstance(preset["sort_order"], int)

        # Render test
        store = ArtifactStore(output_dir=None)
        ctx = RenderContext(
            input=asset,
            params=preset["params"],
            composition=None,
            seed=42,
            store=store,
            calibration=RegistryCalibrationAccessor(),
        )

        algo = registry.get(preset["renderer_id"])

        async def run_algo():
            async for _ in algo.render(ctx):
                pass

        asyncio.run(run_algo())

        # If it uses compositor, test compositing
        if preset["renderer_id"] in ("tonal_analyzer", "structure_analyzer"):
            assert "composition" in preset
            comp_obj = parse_composition(preset["composition"], seed=42)
            from colorworks.compositor import Compositor
            compositor = Compositor(store)
            out_img = compositor.composite(comp_obj, 16, 16, 42)
            assert out_img is not None


def test_every_fixture_has_recommended_preset() -> None:
    from colorworks.presets import BUILTIN_PRESETS
    from colorworks.algorithms.fixtures import FIXTURES

    for fixture_name in FIXTURES.keys():
        recommended_presets = [
            p for p in BUILTIN_PRESETS
            if fixture_name in p.get("recommended_for", [])
        ]
        assert len(recommended_presets) > 0, f"Fixture '{fixture_name}' has no recommended presets!"
