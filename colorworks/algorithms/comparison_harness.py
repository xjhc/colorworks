from __future__ import annotations

import asyncio
import time
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
from colorworks.domain import ArtifactStore, RasterGrid


class RegistryCalibrationAccessor:
    def get_metadata(self, checksum: str) -> dict[str, Any]:
        return calibration_registry.list_assets()[checksum][1]

    def get_data(self, checksum: str) -> Any:
        return calibration_registry.list_assets()[checksum][0]


def make_gradient_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a deterministic gradient image for testing."""
    yy, xx = np.mgrid[0:height, 0:width]
    val = (xx + yy) * 255.0 / (width + height - 2.0)
    arr = np.clip(val, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


async def run_algorithm(algo: Any, ctx: RenderContext) -> None:
    async for _ in algo.render(ctx):
        pass


def run_harness() -> str:
    # Ensure colorworks_data/comparison directory exists
    output_dir = Path("colorworks_data/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    img = make_gradient_image(64, 64)
    substrate = RasterGrid(img.width, img.height)
    asset = MediaAsset(id="comparison_fixture", checksum="fixture_cs", image=img, substrate=substrate)

    orig_gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    # Algorithms to run
    algo_ids = ["floyd_steinberg", "pang_halftoning", "cvt_stippling", "dbs", "saed"]
    
    results = []

    # Make sure they are registered
    import colorworks.algorithms.floyd_steinberg
    import colorworks.algorithms.pang_halftoning
    import colorworks.algorithms.cvt_stippling
    import colorworks.algorithms.dbs
    import colorworks.algorithms.saed

    for algo_id in algo_ids:
        algo = registry.get(algo_id)
        
        # Build standard ink/paper params for consistency in MSE/density metric
        params = {
            "ink_color": "#000000",
            "paper_color": "#ffffff",
        }
        
        # Set algorithm-specific iterations/dots to keep it fast and consistent
        if algo_id == "pang_halftoning":
            params["n_dots"] = 200
            params["max_iterations"] = 30
        elif algo_id == "cvt_stippling":
            params["n_stipples"] = 200
            params["max_iterations"] = 15
        elif algo_id == "dbs":
            params["max_iterations"] = 3
        elif algo_id == "saed":
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
        asyncio.run(run_algorithm(algo, ctx))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Retrieve the final raster image
        try:
            art = store.get_by_name("final_raster")
            out_img = art.value
        except KeyError:
            # Fallback if final_raster wasn't generated
            out_img = Image.new("RGB", (64, 64), (255, 255, 255))

        # Save image
        out_path = output_dir / f"{algo_id}.png"
        out_img.save(out_path, format="PNG")

        # Metrics
        out_gray = np.array(out_img.convert("L"), dtype=np.float32) / 255.0
        mse = np.mean((orig_gray - out_gray) ** 2)
        mean_intensity = np.mean(out_gray)

        results.append({
            "id": algo_id,
            "name": algo.definition.name,
            "runtime_ms": elapsed_ms,
            "mse": mse,
            "mean_intensity": mean_intensity,
            "output_path": str(out_path),
        })

    # Generate Markdown Table
    lines = [
        "| Algorithm | Runtime (ms) | MSE | Mean Intensity | Output Path |",
        "| --- | --- | --- | --- | --- |"
    ]
    for r in results:
        lines.append(
            f"| {r['name']} ({r['id']}) | {r['runtime_ms']:.2f} | {r['mse']:.5f} | {r['mean_intensity']:.4f} | {r['output_path']} |"
        )
    table_str = "\n".join(lines)
    return table_str


if __name__ == "__main__":
    print("Running Quality Comparison Harness...")
    table = run_harness()
    print("\nQuality Comparison Results:")
    print(table)
