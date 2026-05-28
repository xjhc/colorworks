from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from colorworks.algorithms.fixtures import FIXTURES
from colorworks.algorithms.comparison_harness import run_harness, compute_file_sha256


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
    # 7 fixtures * 8 runs (5 core algorithms + 3 presets) = 56 entries
    assert len(manifest) == 56

    expected_fixture_names = {
        "portrait",
        "landscape",
        "line_art",
        "noisy_scan",
        "high_contrast",
        "low_contrast",
        "icon",
    }

    expected_algorithm_ids = {
        "floyd_steinberg",
        "pang_halftoning",
        "cvt_stippling",
        "dbs",
        "saed",
        "wave_halftone",
        "maze_halftone",
        "hatch",
    }

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
        assert "output_path" in entry
        assert "checksum" in entry
        assert "runtime_ms" in entry
        assert "dimensions" in entry
        assert "metrics" in entry

        fixture_name = entry["fixture_name"]
        fixture_checksum = entry["fixture_checksum"]
        run_id = entry["run_id"]
        kind = entry["kind"]
        algo_id = entry["algorithm_id"]
        preset_id = entry["preset_id"]

        assert fixture_name in expected_fixture_names
        assert run_id in expected_algorithm_ids
        assert kind in {"algorithm", "preset"}
        assert len(fixture_checksum) == 64  # SHA-256 hex string length

        if kind == "preset":
            assert algo_id == "tonal_analyzer"
            assert preset_id == run_id
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

        # Verify output path is isolated under tmp_path
        out_path_str = entry["output_path"]
        out_path = Path(out_path_str)
        assert out_path.exists()
        assert out_path.parent.resolve() == tmp_path.resolve()

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

    # Check that we covered all 56 combinations
    assert len(seen_combinations) == 56
