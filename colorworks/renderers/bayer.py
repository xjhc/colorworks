from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_MATRIX_SIZES: Final[tuple[int, ...]] = (2, 4, 8, 16)
INK_RGB: Final[tuple[int, int, int]] = (18, 18, 18)
PAPER_RGB: Final[tuple[int, int, int]] = (244, 235, 217)


@dataclass(frozen=True)
class BayerParams:
    matrix_size: int = 8
    threshold: float = 0.0
    contrast: float = 1.0

    def normalized(self) -> "BayerParams":
        if self.matrix_size not in SUPPORTED_MATRIX_SIZES:
            raise ValueError(
                f"matrix_size must be one of {SUPPORTED_MATRIX_SIZES}; got {self.matrix_size}"
            )
        if not -0.5 <= self.threshold <= 0.5:
            raise ValueError("threshold must be between -0.5 and 0.5")
        if not 0.1 <= self.contrast <= 3.0:
            raise ValueError("contrast must be between 0.1 and 3.0")
        return self

    def to_json(self) -> dict[str, int | float]:
        return {
            "matrix_size": self.matrix_size,
            "threshold": self.threshold,
            "contrast": self.contrast,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "BayerParams":
        return cls(
            matrix_size=int(data.get("matrix_size", cls.matrix_size)),
            threshold=float(data.get("threshold", cls.threshold)),
            contrast=float(data.get("contrast", cls.contrast)),
        ).normalized()


def bayer_matrix(size: int) -> np.ndarray:
    """Return a normalized Bayer threshold matrix in the half-open range [0, 1)."""

    if size not in SUPPORTED_MATRIX_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_MATRIX_SIZES}; got {size}")

    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    current_size = 2
    while current_size < size:
        matrix = np.block(
            [
                [4 * matrix + 0, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
        current_size *= 2

    return (matrix + 0.5) / float(size * size)


def ordered_dither(image: Image.Image, params: BayerParams) -> Image.Image:
    """Render an image as a black-on-paper ordered dither."""

    params = params.normalized()
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    adjusted = np.clip((gray - 0.5) * params.contrast + 0.5, 0.0, 1.0)
    density = np.clip(1.0 - adjusted + params.threshold, 0.0, 1.0)

    threshold_matrix = bayer_matrix(params.matrix_size)
    height, width = density.shape
    repeats_y = (height + params.matrix_size - 1) // params.matrix_size
    repeats_x = (width + params.matrix_size - 1) // params.matrix_size
    tiled_thresholds = np.tile(threshold_matrix, (repeats_y, repeats_x))[:height, :width]

    ink_mask = density >= tiled_thresholds
    output = np.empty((height, width, 3), dtype=np.uint8)
    output[ink_mask] = INK_RGB
    output[~ink_mask] = PAPER_RGB
    return Image.fromarray(output)
