from __future__ import annotations

BUILTIN_PRESETS = [
    {
        "id": "wave_halftone",
        "name": "Wave Halftone",
        "description": "Sinusoidal halftone with edge preservation",
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": True,
            "edge_threshold": 0.15
        },
        "composition": {
            "paper_color": {"hex": "#f4ebd9", "name": "paper"},
            "layers": [
                {
                    "name": "ink",
                    "color": {"hex": "#1a1a1a", "name": "ink"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "wave",
                        "params": {"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                        "mask_source": "edge_mask",
                        "coordinates": {
                            "space": "image_px"
                        }
                    },
                    "threshold": None,
                    "blend_mode": "normal",
                    "opacity": 1.0,
                    "priority": 0
                }
            ]
        },
        "is_builtin": True
    },
    {
        "id": "maze_halftone",
        "name": "Maze Halftone",
        "description": "Truchet maze pattern modulated by image tone",
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": True,
            "edge_threshold": 0.15
        },
        "composition": {
            "paper_color": {"hex": "#f4ebd9", "name": "paper"},
            "layers": [
                {
                    "name": "ink",
                    "color": {"hex": "#1a1a1a", "name": "ink"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "maze",
                        "params": {"scale": 16.0, "line_width": 2.0},
                        "mask_source": "edge_mask",
                        "coordinates": {
                            "space": "image_px"
                        }
                    },
                    "threshold": None,
                    "blend_mode": "normal",
                    "opacity": 1.0,
                    "priority": 0
                }
            ]
        },
        "is_builtin": True
    },
    {
        "id": "hatch",
        "name": "Hatch",
        "description": "Dual-layer parallel lines hatch pattern",
        "renderer_id": "tonal_analyzer",
        "params": {
            "contrast": 1.2,
            "midpoint": 0.5,
            "preserve_edges": True,
            "edge_threshold": 0.15
        },
        "composition": {
            "paper_color": {"hex": "#f4ebd9", "name": "paper"},
            "layers": [
                {
                    "name": "hatch_1",
                    "color": {"hex": "#1a1a1a", "name": "dark ink"},
                    "role": "shadow",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "hatch",
                        "params": {"frequency": 8.0, "angle_deg": 45.0, "phase": 0.0},
                        "mask_source": "edge_mask",
                        "coordinates": {
                            "space": "image_px"
                        }
                    },
                    "threshold": None,
                    "blend_mode": "normal",
                    "opacity": 1.0,
                    "priority": 0
                },
                {
                    "name": "hatch_2",
                    "color": {"hex": "#404040", "name": "light ink"},
                    "role": "highlight",
                    "density_source": "tone_map",
                    "pattern": {
                        "kind": "hatch",
                        "params": {"frequency": 8.0, "angle_deg": 135.0, "phase": 0.0},
                        "mask_source": "edge_mask",
                        "coordinates": {
                            "space": "image_px"
                        }
                    },
                    "threshold": None,
                    "blend_mode": "multiply",
                    "opacity": 0.8,
                    "priority": 1
                }
            ]
        },
        "is_builtin": True
    }
]
