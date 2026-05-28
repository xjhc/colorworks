from __future__ import annotations

import asyncio
import time
import io
import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

from colorworks.algorithms import (
    MediaAsset,
    RenderContext,
    registry,
    calibration_registry,
)
from colorworks.domain import (
    ArtifactStore,
    RasterGrid,
    Composition,
    PaletteColor,
    InkLayerSpec,
    PatternSpec,
    PatternCoordinateSpec,
)
from colorworks.algorithms.fixtures import FIXTURES


class RegistryCalibrationAccessor:
    def get_metadata(self, checksum: str) -> dict[str, Any]:
        return calibration_registry.list_assets()[checksum][1]

    def get_data(self, checksum: str) -> Any:
        return calibration_registry.list_assets()[checksum][0]


def parse_composition(comp_dict: dict, seed: int = 42) -> Composition:
    paper_hex = comp_dict.get("paper_color", {}).get("hex", "#f4ebd9")
    layers_list = []
    for l in comp_dict.get("layers", []):
        pat_dict = l.get("pattern", {})
        pat_spec = PatternSpec(
            kind=pat_dict.get("kind", "wave"),
            params=pat_dict.get("params", {}),
            field_source=pat_dict.get("field_source"),
            orientation_source=pat_dict.get("orientation_source"),
            warp_source=pat_dict.get("warp_source"),
            mask_source=pat_dict.get("mask_source"),
            coordinates=PatternCoordinateSpec(
                space=pat_dict.get("coordinates", {}).get("space", "image_px"),
                origin=tuple(pat_dict.get("coordinates", {}).get("origin", [0.0, 0.0])),
                scale=float(pat_dict.get("coordinates", {}).get("scale", 1.0)),
                rotation_deg=float(pat_dict.get("coordinates", {}).get("rotation_deg", 0.0)),
                seed=pat_dict.get("coordinates", {}).get("seed")
                if pat_dict.get("coordinates", {}).get("seed") is not None
                else seed,
            ),
        )
        layers_list.append(InkLayerSpec(
            name=l.get("name", "ink"),
            color=PaletteColor(l.get("color", {}).get("hex", "#1a1a1a")),
            role=l.get("role", "shadow"),
            density_source=l.get("density_source", "tone_map"),
            pattern=pat_spec,
            threshold=l.get("threshold"),
            blend_mode=l.get("blend_mode", "normal"),
            opacity=float(l.get("opacity", 1.0)),
            priority=int(l.get("priority", 0)),
        ))
    return Composition(paper_color=PaletteColor(paper_hex), layers=layers_list)


async def run_algorithm(algo: Any, ctx: RenderContext) -> None:
    async for _ in algo.render(ctx):
        pass


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_harness(output_dir: Path | str | None = None) -> str:
    if output_dir is None:
        output_dir = Path("colorworks_data/comparison/latest")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing PNG files and manifest.json to avoid stale outputs
    for p in output_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass
    manifest_file = output_dir / "manifest.json"
    if manifest_file.exists():
        try:
            manifest_file.unlink()
        except OSError:
            pass

    # Ensure all required algorithms are registered
    import colorworks.algorithms.floyd_steinberg
    import colorworks.algorithms.pang_halftoning
    import colorworks.algorithms.cvt_stippling
    import colorworks.algorithms.dbs
    import colorworks.algorithms.saed
    import colorworks.algorithms.tonal_analyzer
    from colorworks.presets import BUILTIN_PRESETS

    core_algos = ["floyd_steinberg", "pang_halftoning", "cvt_stippling", "dbs", "saed"]
    runs = []
    # Core algorithms
    for algo_id in core_algos:
        algo = registry.get(algo_id)
        runs.append({
            "id": algo_id,
            "name": algo.definition.name,
            "is_preset": False,
            "preset_data": None
        })
    # Presets
    for preset in BUILTIN_PRESETS:
        runs.append({
            "id": preset["id"],
            "name": preset["name"],
            "is_preset": True,
            "preset_data": preset
        })

    manifest_data = []
    markdown_sections = []

    # Run for all fixtures
    for fixture_name, generator_fn in FIXTURES.items():
        img = generator_fn()
        width, height = img.size

        # Compute deterministic checksum of the input fixture
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fixture_checksum = hashlib.sha256(buf.getvalue()).hexdigest()

        substrate = RasterGrid(width, height)
        asset = MediaAsset(
            id=f"fixture_{fixture_name}",
            checksum=fixture_checksum,
            image=img,
            substrate=substrate,
        )

        orig_gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

        fixture_results = []

        for spec in runs:
            run_id = spec["id"]
            run_name = spec["name"]
            is_preset = spec["is_preset"]

            if not is_preset:
                # Build standard ink/paper params for consistency in MSE/density metric
                params = {
                    "ink_color": "#000000",
                    "paper_color": "#ffffff",
                }

                # Set algorithm-specific iterations/dots to keep it fast and consistent
                if run_id == "pang_halftoning":
                    params["n_dots"] = 200
                    params["max_iterations"] = 30
                elif run_id == "cvt_stippling":
                    params["n_stipples"] = 200
                    params["max_iterations"] = 15
                elif run_id == "dbs":
                    params["max_iterations"] = 3
                elif run_id == "saed":
                    # SAED defaults
                    params["contrast"] = 1.0
                    params["midpoint"] = 0.5
                    params["gabor_amplitude"] = 0.15
                    params["anisotropy_alpha"] = 0.4

                store = ArtifactStore(output_dir=None)
                ctx = RenderContext(
                    input=asset,
                    params=params,
                    composition=None,
                    seed=42,
                    store=store,
                    calibration=RegistryCalibrationAccessor(),
                )

                start = time.perf_counter()
                algo = registry.get(run_id)
                asyncio.run(run_algorithm(algo, ctx))
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                # Retrieve the final raster image (fail hard if missing)
                art = store.get_by_name("final_raster")
                out_img = art.value
            else:
                # Run the preset renderer (tonal_analyzer)
                preset_data = spec["preset_data"]
                params = dict(preset_data["params"])

                store = ArtifactStore(output_dir=None)
                ctx = RenderContext(
                    input=asset,
                    params=params,
                    composition=None,
                    seed=42,
                    store=store,
                    calibration=RegistryCalibrationAccessor(),
                )

                start = time.perf_counter()
                algo = registry.get("tonal_analyzer")
                asyncio.run(run_algorithm(algo, ctx))

                # Now compose the layers
                comp_obj = parse_composition(preset_data["composition"], seed=42)
                from colorworks.compositor import Compositor
                compositor = Compositor(store)
                out_img = compositor.composite(comp_obj, width, height, 42)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Save image
            out_path = output_dir / f"{fixture_name}_{run_id}.png"
            out_img.save(out_path, format="PNG")

            # Checksum of the output PNG
            out_checksum = compute_file_sha256(out_path)

            # Metrics
            out_gray = np.array(out_img.convert("L"), dtype=np.float32) / 255.0
            mse = np.mean((orig_gray - out_gray) ** 2)
            mean_intensity = np.mean(out_gray)

            # Keep for markdown section
            fixture_results.append({
                "id": run_id,
                "name": run_name,
                "runtime_ms": elapsed_ms,
                "mse": mse,
                "mean_intensity": mean_intensity,
                "output_path": str(out_path),
            })

            # Keep for JSON manifest
            manifest_data.append({
                "fixture_name": fixture_name,
                "fixture_checksum": fixture_checksum,
                "run_id": run_id,
                "kind": "preset" if is_preset else "algorithm",
                "algorithm_id": "tonal_analyzer" if is_preset else run_id,
                "preset_id": run_id if is_preset else None,
                "params": params,
                "output_path": str(out_path),
                "checksum": out_checksum,
                "runtime_ms": elapsed_ms,
                "dimensions": [width, height],
                "metrics": {
                    "mse": float(mse),
                    "mean_intensity": float(mean_intensity),
                }
            })

        # Generate Markdown Table for this fixture
        lines = [
            f"### Fixture: {fixture_name} ({width}x{height})",
            "| Algorithm | Runtime (ms) | MSE | Mean Intensity | Output Path |",
            "| --- | --- | --- | --- | --- |"
        ]
        for r in fixture_results:
            lines.append(
                f"| {r['name']} ({r['id']}) | {r['runtime_ms']:.2f} | {r['mse']:.5f} | {r['mean_intensity']:.4f} | {r['output_path']} |"
            )
        markdown_sections.append("\n".join(lines))

    # Write Manifest JSON
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return "\n\n".join(markdown_sections)


if __name__ == "__main__":
    print("Running Quality Comparison Harness...")
    table = run_harness()
    print("\nQuality Comparison Results:")
    print(table)
