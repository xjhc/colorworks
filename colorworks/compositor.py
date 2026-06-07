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
from colorworks.algorithms.dither import (
    bayer_threshold_map,
    blue_noise_threshold_map,
    maze_threshold_map,
)

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
            if layer.pattern.kind in ("hatch", "crosshatch"):
                from PIL import ImageDraw
                ink_mask_img = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(ink_mask_img)

                stroke_sets = self.build_stroke_set(
                    Composition(paper_color=composition.paper_color, layers=[layer]),
                    width,
                    height,
                    run_seed,
                )
                if stroke_sets:
                    _, stroke_set = stroke_sets[0]
                    for stroke in stroke_set.strokes:
                        pts = stroke.path.points
                        if len(pts) < 2:
                            continue
                        if stroke.width_profile is not None:
                            for i in range(len(pts) - 1):
                                p1 = pts[i]
                                p2 = pts[i + 1]
                                seg_w = float((stroke.width_profile[i] + stroke.width_profile[i + 1]) / 2.0)
                                draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=255, width=max(1, int(seg_w)))
                        else:
                            draw.line([(p[0], p[1]) for p in pts], fill=255, width=1)

                ink_mask = np.asarray(ink_mask_img) > 127
            else:
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

    def build_stroke_set(
        self,
        composition: Composition,
        width: int,
        height: int,
        run_seed: int = 0,
    ) -> list[tuple[InkLayerSpec, StrokeSet]]:
        from colorworks.domain import StrokeSet, Stroke, Polyline, RasterGrid, VectorField2D
        results = []

        for layer in composition.layers:
            if layer.pattern.kind not in ("hatch", "crosshatch"):
                continue

            # 1. Load density source
            density_art = self.store.get_by_name(layer.density_source)
            density = density_art.value.data
            if density_art.name == "tone_map":
                density = 1.0 - density

            # Ensure density matches size
            if density.shape != (height, width):
                density_img = Image.fromarray((density * 255.0).astype(np.uint8), mode="L")
                density_img = density_img.resize((width, height), Image.Resampling.BILINEAR)
                density = np.asarray(density_img, dtype=np.float32) / 255.0

            if layer.pattern.mask_source:
                try:
                    mask_art = self.store.get_by_name(layer.pattern.mask_source)
                    mask = mask_art.value.data
                    if mask.shape != (height, width):
                        mask_img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
                        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
                        mask = np.asarray(mask_img) > 127
                    density = np.clip(density + mask.astype(np.float32), 0.0, 1.0)
                except KeyError:
                    pass

            # 2. Get orientation field vector
            orientation_src = layer.pattern.orientation_source
            if orientation_src:
                try:
                    orientation_art = self.store.get_by_name(orientation_src)
                except KeyError:
                    raise ValueError(f"Orientation source '{orientation_src}' not found in store")

                if not isinstance(orientation_art.value, VectorField2D):
                    raise ValueError(f"Orientation source '{orientation_src}' is not a VectorField2D")
                vector_field = orientation_art.value
            else:
                vector_field = None

            # Parameters
            freq = max(0.1, float(layer.pattern.params.get("frequency", 8.0)))
            angle_deg = float(layer.pattern.params.get("angle_deg", 45.0))

            coords = layer.pattern.coordinates
            rotation_deg = coords.rotation_deg

            # Calculate line spacing in pixels
            spacing = max(1.0, 100.0 / freq)

            # Generate strokes
            strokes = []
            substrate = RasterGrid(width, height)

            directions = [angle_deg + rotation_deg]
            if layer.pattern.kind == "crosshatch":
                directions.append(angle_deg + rotation_deg + 90.0)

            for flow_angle in directions:
                if vector_field is not None:
                    raw_vf = vector_field.data
                    use_rotation_90 = (flow_angle != (angle_deg + rotation_deg))
                else:
                    theta = np.radians(flow_angle)
                    const_v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                    raw_vf = None
                    use_rotation_90 = False

                grid_res = max(4.0, spacing / 2.0)
                grid_h = int(np.ceil(height / grid_res))
                grid_w = int(np.ceil(width / grid_res))
                occupied = np.zeros((grid_h, grid_w), dtype=bool)

                def cell_for(px, py):
                    return int(px / grid_res), int(py / grid_res)

                def is_cell_occupied(px, py):
                    gx, gy = cell_for(px, py)
                    if 0 <= gx < grid_w and 0 <= gy < grid_h:
                        return occupied[gy, gx]
                    return True

                def mark_cell_occupied(px, py):
                    gx, gy = cell_for(px, py)
                    if 0 <= gx < grid_w and 0 <= gy < grid_h:
                        occupied[gy, gx] = True

                seeds = []
                step = max(1, int(round(spacing)))
                for sy in range(0, height, step):
                    for sx in range(0, width, step):
                        seeds.append((float(sx), float(sy)))

                seed_val = coords.seed if (coords and coords.seed is not None) else run_seed
                rng = np.random.default_rng(seed_val)
                rng.shuffle(seeds)

                ds = 2.0
                for seed_x, seed_y in seeds:
                    if is_cell_occupied(seed_x, seed_y):
                        continue

                    pts = [(seed_x, seed_y)]
                    mark_cell_occupied(seed_x, seed_y)

                    def trace_dir(start_x, start_y, direction_sign):
                        cur_x, cur_y = start_x, start_y
                        prev_dir = None
                        path_pts = []

                        for _ in range(500):
                            if raw_vf is not None:
                                nx = int(np.clip(cur_x, 0, width - 1))
                                ny = int(np.clip(cur_y, 0, height - 1))
                                vx = raw_vf[ny, nx, 0]
                                vy = raw_vf[ny, nx, 1]
                                if use_rotation_90:
                                    vx, vy = -vy, vx
                            else:
                                vx, vy = const_v[0], const_v[1]

                            mag = np.hypot(vx, vy)
                            if mag < 1e-6:
                                break
                            dx = vx / mag
                            dy = vy / mag

                            v_dir = np.array([dx, dy])
                            if prev_dir is not None:
                                if np.dot(v_dir, prev_dir) < 0:
                                    v_dir = -v_dir
                            else:
                                v_dir = v_dir * direction_sign

                            prev_dir = v_dir
                            next_x = cur_x + v_dir[0] * ds
                            next_y = cur_y + v_dir[1] * ds

                            if not (0 <= next_x < width and 0 <= next_y < height):
                                break

                            cur_cell = cell_for(cur_x, cur_y)
                            next_cell = cell_for(next_x, next_y)
                            if next_cell != cur_cell and is_cell_occupied(next_x, next_y):
                                break

                            if next_cell != cur_cell:
                                mark_cell_occupied(next_x, next_y)
                            path_pts.append((next_x, next_y))
                            cur_x, cur_y = next_x, next_y
                        return path_pts

                    forward_pts = trace_dir(seed_x, seed_y, 1.0)
                    backward_pts = trace_dir(seed_x, seed_y, -1.0)

                    streamline_pts = list(reversed(backward_pts)) + pts + forward_pts
                    if len(streamline_pts) >= 3:
                        pts_arr = np.array(streamline_pts, dtype=np.float32)
                        ny_indices = np.clip(pts_arr[:, 1], 0, height - 1).astype(int)
                        nx_indices = np.clip(pts_arr[:, 0], 0, width - 1).astype(int)
                        d_vals = density[ny_indices, nx_indices]

                        thresh = layer.threshold if layer.threshold is not None else 0.05

                        cur_stroke_pts = []
                        cur_stroke_widths = []

                        for i in range(len(pts_arr)):
                            d = d_vals[i]
                            if d >= thresh:
                                cur_stroke_pts.append(pts_arr[i])
                                max_w = spacing * 0.8
                                cur_stroke_widths.append(d * max_w)
                            else:
                                if len(cur_stroke_pts) >= 2:
                                    strokes.append(Stroke(
                                        path=Polyline(np.array(cur_stroke_pts, dtype=np.float32)),
                                        width_profile=np.array(cur_stroke_widths, dtype=np.float32),
                                        color_index=0
                                    ))
                                cur_stroke_pts = []
                                cur_stroke_widths = []

                        if len(cur_stroke_pts) >= 2:
                            strokes.append(Stroke(
                                path=Polyline(np.array(cur_stroke_pts, dtype=np.float32)),
                                width_profile=np.array(cur_stroke_widths, dtype=np.float32),
                                color_index=0
                            ))

            results.append((layer, StrokeSet(substrate, strokes)))

        return results

    def _generate_pattern(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        # Check if there is a custom registered generator for this pattern kind
        custom_gen = registry.get_pattern_generator(pattern.kind)
        if custom_gen is not None:
            return custom_gen(pattern, width, height, run_seed)

        if pattern.kind == "wave":
            return self._generate_wave(pattern, width, height, run_seed)
        elif pattern.kind == "ordered_dither":
            return self._generate_ordered_dither(pattern, width, height, run_seed)
        elif pattern.kind == "blue_noise":
            return self._generate_blue_noise(pattern, width, height, run_seed)
        elif pattern.kind == "maze":
            return self._generate_maze(pattern, width, height, run_seed)
        elif pattern.kind == "hatch":
            return self._generate_hatch(pattern, width, height, run_seed)
        elif pattern.kind == "solid":
            # Solid ink pattern is just 0 threshold everywhere (always ink)
            return np.zeros((height, width), dtype=np.float32)
        else:
            # Fallback to solid
            return np.zeros((height, width), dtype=np.float32)

    def _generate_ordered_dither(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        # Delegates to the shared mask generator in colorworks.algorithms.dither.
        return bayer_threshold_map(width, height, int(pattern.params.get("matrix_size", 8)))

    def _generate_blue_noise(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        seed = pattern.coordinates.seed if (pattern.coordinates and pattern.coordinates.seed is not None) else run_seed
        return blue_noise_threshold_map(width, height, int(pattern.params.get("size", 64)), seed)

    def _generate_maze(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
        seed = pattern.coordinates.seed if (pattern.coordinates and pattern.coordinates.seed is not None) else run_seed
        return maze_threshold_map(
            width,
            height,
            float(pattern.params.get("scale", 16.0)),
            float(pattern.params.get("line_width", 2.0)),
            seed,
        )

    def _generate_hatch(self, pattern: PatternSpec, width: int, height: int, run_seed: int) -> np.ndarray:
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
            X = X / float(width)
            Y = Y / float(height)
            f_scaled = freq * scale
        else:
            X = X - origin[0]
            Y = Y - origin[1]
            f_scaled = (freq / 100.0) * scale

        # Calculate distance along wave direction
        dist = X * np.cos(theta) + Y * np.sin(theta)

        # Periodic triangle wave in range [0, 1]
        phase_val = dist * f_scaled + phase
        fractional_part = phase_val - np.floor(phase_val)
        P = 2.0 * np.abs(fractional_part - 0.5)

        return P

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
