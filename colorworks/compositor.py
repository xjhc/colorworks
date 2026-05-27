from __future__ import annotations

import numpy as np
from PIL import Image

from colorworks.domain import (
    Composition,
    ArtifactStore,
    InkLayerSpec,
    PatternSpec,
    PatternCoordinateSpec,
)
from colorworks.algorithms import registry

def parse_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)

class Compositor:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def composite(self, composition: Composition, width: int, height: int, run_seed: int = 0) -> Image.Image:
        # Create the initial canvas with paper color
        paper_rgb = parse_color(composition.paper_color.hex)
        canvas = np.full((height, width, 3), paper_rgb, dtype=np.uint8)

        # Sort layers by priority
        sorted_layers = sorted(composition.layers, key=lambda l: l.priority)

        for layer in sorted_layers:
            # 1. Load density source
            density_art = self.store.get_by_name(layer.density_source)
            density = density_art.value.data
            
            # If the source is named tone_map, it represents tone (0=black, 1=white).
            # We invert it to get ink density (1=ink, 0=no ink).
            if density_art.name == "tone_map":
                density = 1.0 - density

            # Ensure density matches canvas size
            if density.shape != (height, width):
                # Resize density
                density_img = Image.fromarray((density * 255.0).astype(np.uint8), mode="L")
                density_img = density_img.resize((width, height), Image.Resampling.BILINEAR)
                density = np.asarray(density_img, dtype=np.float32) / 255.0

            # 2. Load mask source if specified
            mask = None
            if layer.pattern.mask_source:
                try:
                    mask_art = self.store.get_by_name(layer.pattern.mask_source)
                    mask = mask_art.value.data
                    if mask.shape != (height, width):
                        # Resize mask
                        mask_img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
                        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
                        mask = np.asarray(mask_img) > 127
                except KeyError:
                    pass

            # 3. Generate pattern field P
            P = self._generate_pattern(layer.pattern, width, height, run_seed)

            # 4. Apply density boost using the mask
            if mask is not None:
                # Add the mask as a density boost (value 1.0)
                density_final = np.clip(density + mask.astype(np.float32) * 1.0, 0.0, 1.0)
            else:
                density_final = density

            # 5. Threshold pattern to determine ink locations
            # Apply threshold if specified on layer
            if layer.threshold is not None:
                # Only apply pattern where density is above the threshold
                ink_mask = (density_final >= layer.threshold) & (density_final >= P)
            else:
                ink_mask = density_final >= P

            # 6. Apply blending
            ink_rgb = parse_color(layer.color.hex)
            opacity = layer.opacity

            if layer.blend_mode == "multiply":
                # Multiply blend mode: Canvas * Ink
                # We blend with opacity: blended = canvas * (1 - opacity) + (canvas * ink) * opacity
                ink_factor = np.array(ink_rgb, dtype=np.float32) / 255.0
                canvas_float = canvas.astype(np.float32)
                multiply_val = canvas_float * ink_factor
                blended = (1.0 - opacity) * canvas_float + opacity * multiply_val
                
                canvas[ink_mask] = np.clip(blended[ink_mask], 0.0, 255.0).astype(np.uint8)
            else:
                # Normal blend mode
                # blended = canvas * (1 - opacity) + ink * opacity
                canvas_float = canvas.astype(np.float32)
                ink_val = np.array(ink_rgb, dtype=np.float32)
                blended = (1.0 - opacity) * canvas_float + opacity * ink_val
                
                canvas[ink_mask] = np.clip(blended[ink_mask], 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(canvas)

    def _generate_pattern(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        # Check if there is a custom registered generator for this pattern kind
        custom_gen = registry.get_pattern_generator(pattern.kind)
        if custom_gen is not None:
            return custom_gen(pattern, width, height, run_seed)

        if pattern.kind == "wave":
            return self._generate_wave(pattern, width, height, run_seed)
        elif pattern.kind == "solid":
            # Solid ink pattern is just 0 threshold everywhere (always ink)
            return np.zeros((height, width), dtype=np.float32)
        else:
            # Fallback to solid
            return np.zeros((height, width), dtype=np.float32)

    def _generate_wave(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        # Read parameters
        freq = float(pattern.params.get("frequency", 8.0))
        angle_deg = float(pattern.params.get("angle_deg", 45.0))
        phase = float(pattern.params.get("phase", 0.0))

        coords = pattern.coordinates
        space = coords.space
        origin = coords.origin
        scale = coords.scale
        rotation_deg = coords.rotation_deg

        # Total rotation
        theta = np.radians(angle_deg + rotation_deg)

        # Create grid
        x = np.arange(width, dtype=np.float32)
        y = np.arange(height, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        # Transform coordinates based on space
        if space == "normalized":
            # Normalize to [0, 1]
            X = X / float(width)
            Y = Y / float(height)
            # Adjust frequency to be canvas-relative
            # Frequency is cycles / 100 pixels, so in normalized space:
            f_scaled = freq * scale
        else:
            # "image_px" and "output_px"
            X = X - origin[0]
            Y = Y - origin[1]
            # Frequency is cycles / 100 pixels
            f_scaled = (freq / 100.0) * scale

        # Calculate distance along wave direction
        dist = X * np.cos(theta) + Y * np.sin(theta)

        # Sinusoidal wave
        # P is in range [0, 1]
        phase_val = dist * f_scaled + phase
        P = 0.5 + 0.5 * np.cos(2.0 * np.pi * phase_val)

        return P
