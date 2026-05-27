from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image

from colorworks.renderers.bayer import BayerParams, INK_RGB, PAPER_RGB, bayer_matrix, ordered_dither


def test_bayer_matrix_is_normalized_and_ordered() -> None:
    matrix = bayer_matrix(4)

    assert matrix.shape == (4, 4)
    assert matrix.min() > 0.0
    assert matrix.max() < 1.0
    assert len(np.unique(matrix)) == 16
    assert matrix[0, 0] < matrix[0, 1]


def test_ordered_dither_outputs_only_ink_and_paper() -> None:
    gradient = np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 16
    image = Image.fromarray(gradient)

    output = ordered_dither(image, BayerParams(matrix_size=4, threshold=0.0, contrast=1.0))
    colors = {tuple(pixel) for pixel in np.asarray(output).reshape(-1, 3)}

    assert colors == {INK_RGB, PAPER_RGB}


def test_ordered_dither_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    source = Image.fromarray(rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8))
    params = BayerParams(matrix_size=8, threshold=-0.08, contrast=1.35)

    first = ordered_dither(source, params)
    second = ordered_dither(source, params)

    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(second.tobytes()).hexdigest()


def test_one_megapixel_render_is_reasonably_fast() -> None:
    gradient = np.tile(np.arange(1000, dtype=np.uint16), (1000, 1))
    source = Image.fromarray(((gradient / 999) * 255).astype(np.uint8))

    started = time.perf_counter()
    output = ordered_dither(source, BayerParams(matrix_size=8, threshold=0.0, contrast=1.2))
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert output.size == (1000, 1000)
    assert elapsed_ms < 300
