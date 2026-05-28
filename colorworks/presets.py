from __future__ import annotations

BUILTIN_PRESETS = [
    {
        "id": "wave_halftone",
        "name": "Wave Halftone",
        "description": "Sinusoidal halftone pattern with edge preservation. Ideal for landscapes and gradients.",
        "recommended_for": ["landscape", "low_contrast"],
        "style_tags": ["halftone", "wave", "artistic"],
        "sort_order": 1,
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
        "description": "Truchet maze pattern modulated by local tone. Ideal for graphic art and bold shapes.",
        "recommended_for": ["icon", "high_contrast"],
        "style_tags": ["maze", "truchet", "graphic"],
        "sort_order": 2,
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
                        "params": {"scale": 12.0, "line_width": 1.5},
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
        "description": "Artistic dual-layer crosshatch pattern for shading. Ideal for portraits and landscape structures.",
        "recommended_for": ["portrait", "landscape"],
        "style_tags": ["hatch", "shading", "artistic"],
        "sort_order": 3,
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
                        "params": {"frequency": 10.0, "angle_deg": 45.0, "phase": 0.0},
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
                        "params": {"frequency": 10.0, "angle_deg": 135.0, "phase": 0.0},
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
    },
    {
        "id": "stipple_portrait",
        "name": "Stipple Portrait",
        "description": "Voronoi stippling for natural, organic stipple point placement. Ideal for portraits.",
        "recommended_for": ["portrait", "low_contrast"],
        "style_tags": ["stipple", "voronoi", "organic"],
        "sort_order": 4,
        "renderer_id": "cvt_stippling",
        "params": {
            "n_stipples": 200,
            "max_iterations": 15,
            "ink_color": "#1a1a1a",
            "paper_color": "#f4ebd9"
        },
        "is_builtin": True
    },
    {
        "id": "classic_dither",
        "name": "Classic Dither",
        "description": "Floyd-Steinberg error diffusion for crisp, pixel-accurate dithering. Ideal for line art.",
        "recommended_for": ["line_art", "high_contrast"],
        "style_tags": ["dither", "diffusion", "classic"],
        "sort_order": 5,
        "renderer_id": "floyd_steinberg",
        "params": {
            "contrast": 1.0,
            "midpoint": 0.5,
            "ink_color": "#1a1a1a",
            "paper_color": "#f4ebd9"
        },
        "is_builtin": True
    },
    {
        "id": "structure_ink",
        "name": "Structure Ink",
        "description": "Structure-Aware Error Diffusion (SAED) aligning dots along contours. Ideal for portraits.",
        "recommended_for": ["portrait", "noisy_scan"],
        "style_tags": ["diffusion", "structure-aware", "contour"],
        "sort_order": 6,
        "renderer_id": "saed",
        "params": {
            "contrast": 1.0,
            "midpoint": 0.5,
            "sigma": 3.0,
            "etf_iterations": 3,
            "etf_radius": 5,
            "gabor_amplitude": 0.2,
            "anisotropy_alpha": 0.5,
            "edge_scaling": 5.0,
            "ink_color": "#1a1a1a",
            "paper_color": "#f4ebd9"
        },
        "is_builtin": True
    },
    {
        "id": "clean_scan",
        "name": "Clean Scan",
        "description": "High-contrast error diffusion to threshold and clear scanned noise. Ideal for documents.",
        "recommended_for": ["noisy_scan"],
        "style_tags": ["dither", "threshold", "clean"],
        "sort_order": 7,
        "renderer_id": "floyd_steinberg",
        "params": {
            "contrast": 1.8,
            "midpoint": 0.45,
            "ink_color": "#1a1a1a",
            "paper_color": "#ffffff"
        },
        "is_builtin": True
    }
]
