from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from PIL import Image, ImageOps

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{3}$|^#[0-9a-fA-F]{6}$")


FitMode = Literal["fit", "cover", "stretch"]


@dataclass(frozen=True)
class ResizeSpec:
    max_width: int | None = None
    max_height: int | None = None
    fit: FitMode = "fit"

    def is_noop(self) -> bool:
        return self.max_width is None and self.max_height is None

    def cache_token(self) -> str:
        if self.is_noop():
            return ""
        return f"w={self.max_width or '-'},h={self.max_height or '-'},fit={self.fit}"

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> "ResizeSpec":
        if not data:
            return cls()
        mw = data.get("max_width")
        mh = data.get("max_height")
        fit = data.get("fit", "fit")
        if fit not in ("fit", "cover", "stretch"):
            raise ValueError(f"invalid fit mode: {fit}")
        return cls(
            max_width=int(mw) if mw not in (None, "", 0) else None,
            max_height=int(mh) if mh not in (None, "", 0) else None,
            fit=fit,
        )


def resize_for_output(image: Image.Image, spec: ResizeSpec) -> Image.Image:
    """Resize a PIL image to fit the given output spec. No-op if spec is empty.

    fit: scale down to fit inside the box, preserve aspect, never upscale.
    cover: scale so the image covers the box, preserve aspect, then center-crop.
    stretch: resize to exactly the target dimensions, ignoring aspect.
    """
    if spec.is_noop():
        return image

    src_w, src_h = image.size
    mw = spec.max_width
    mh = spec.max_height

    if spec.fit == "stretch":
        target_w = mw if mw is not None else src_w
        target_h = mh if mh is not None else src_h
        if (target_w, target_h) == (src_w, src_h):
            return image
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # fit and cover share aspect-preserving scaling math
    if mw is not None and mh is not None:
        scale_w = mw / src_w
        scale_h = mh / src_h
        scale = min(scale_w, scale_h) if spec.fit == "fit" else max(scale_w, scale_h)
    elif mw is not None:
        scale = mw / src_w
    else:
        scale = mh / src_h

    # never upscale in fit mode
    if spec.fit == "fit" and scale >= 1.0:
        return image

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if spec.fit == "cover" and mw is not None and mh is not None:
        # center-crop to exact box
        left = max(0, (new_w - mw) // 2)
        top = max(0, (new_h - mh) // 2)
        resized = resized.crop((left, top, left + mw, top + mh))

    return resized


def derive_asset_checksum(base_checksum: str, spec: ResizeSpec) -> str:
    """Return an effective asset checksum that captures the output spec.

    Used so the render cache keys treat the same source image at different
    output sizes as distinct inputs. When spec is a no-op, returns the base
    checksum unchanged for backwards compatibility with existing caches.
    """
    if spec.is_noop():
        return base_checksum
    payload = f"{base_checksum}:{spec.cache_token()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_color(hex_str: str) -> None:
    if not HEX_COLOR_RE.match(hex_str):
        raise ValueError(f"Invalid hex color format: {hex_str}. Must be #RGB or #RRGGBB.")


def parse_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def colorize_binary_ink_mask(mask: np.ndarray, ink_color: str, paper_color: str) -> Image.Image:
    """Colorize a boolean or binary mask where True/1.0 is ink_color and False/0.0 is paper_color.

    Explicit tone convention: 1.0/True = ink, 0.0/False = paper.
    """
    validate_color(ink_color)
    validate_color(paper_color)
    ink_rgb = parse_color(ink_color)
    paper_rgb = parse_color(paper_color)

    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    bool_mask = (mask > 0.5)
    out[bool_mask] = ink_rgb
    out[~bool_mask] = paper_rgb

    return Image.fromarray(out, mode="RGB")


def to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0


def remap_tone(gray: np.ndarray, contrast: float, midpoint: float) -> np.ndarray:
    return np.clip((gray - midpoint) * contrast + 0.5, 0.0, 1.0)


def convolve2d_nearest(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    output = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            output += padded[i : i + H, j : j + W] * kernel[i, j]
    return output


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(max(1.0, 3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    kernel = np.exp(-(offsets.astype(np.float32) ** 2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    # Horizontal blur
    padded_h = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    h_blurred = np.zeros_like(img)
    for offset, weight in zip(offsets, kernel):
        h_blurred += padded_h[:, radius + offset : radius + offset + img.shape[1]] * weight

    # Vertical blur
    padded_v = np.pad(h_blurred, ((radius, radius), (0, 0)), mode="edge")
    v_blurred = np.zeros_like(img)
    for offset, weight in zip(offsets, kernel):
        v_blurred += padded_v[radius + offset : radius + offset + img.shape[0], :] * weight
    return v_blurred


def etf_smooth(t: np.ndarray, Jxx: np.ndarray, Jyy: np.ndarray, iterations: int, radius: int) -> np.ndarray:
    H, W, _ = t.shape
    mag = np.sqrt(np.maximum(Jxx + Jyy, 0.0))

    for _ in range(iterations):
        t_new = np.zeros_like(t)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist2 = dx**2 + dy**2
                if dist2 > radius**2:
                    continue

                w_s = np.exp(-dist2 / (2.0 * (radius / 2.0)**2))

                y_min, y_max = max(0, dy), min(H, H + dy)
                x_min, x_max = max(0, dx), min(W, W + dx)

                y_shift_min, y_shift_max = max(0, -dy), min(H, H - dy)
                x_shift_min, x_shift_max = max(0, -dx), min(W, W - dx)

                t_neighbor = t[y_shift_min:y_shift_max, x_shift_min:x_shift_max]
                mag_neighbor = mag[y_shift_min:y_shift_max, x_shift_min:x_shift_max]

                dot = (t_neighbor[:, :, 0] * t[y_min:y_max, x_min:x_max, 0] +
                       t_neighbor[:, :, 1] * t[y_min:y_max, x_min:x_max, 1])
                sign = np.where(dot >= 0, 1.0, -1.0)

                weight = w_s * mag_neighbor

                t_new[y_min:y_max, x_min:x_max, 0] += weight * sign * t_neighbor[:, :, 0]
                t_new[y_min:y_max, x_min:x_max, 1] += weight * sign * t_neighbor[:, :, 1]

        # Avoid division-by-zero runtime warning by using np.divide with where clause
        norm = np.linalg.norm(t_new, axis=-1, keepdims=True)
        t = np.divide(t_new, norm, out=t, where=norm > 1e-6)

    return t
