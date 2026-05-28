from __future__ import annotations

from pathlib import Path
from colorworks.algorithms.comparison_harness import run_harness


def test_comparison_harness() -> None:
    # Run the quality comparison harness
    table_str = run_harness()
    
    assert table_str is not None
    assert "| Algorithm |" in table_str
    assert "saed" in table_str
    assert "dbs" in table_str
    assert "floyd_steinberg" in table_str
    
    # Check that output files exist and are not empty
    output_dir = Path("colorworks_data/comparison")
    assert output_dir.exists()
    
    algo_ids = ["floyd_steinberg", "pang_halftoning", "cvt_stippling", "dbs", "saed"]
    for algo_id in algo_ids:
        img_path = output_dir / f"{algo_id}.png"
        assert img_path.exists()
        assert img_path.stat().st_size > 0
