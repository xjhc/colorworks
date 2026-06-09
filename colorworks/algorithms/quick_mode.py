from __future__ import annotations

# Registry table of candidate algorithms for Quick Mode — the curated style set.
#
# Most cards are powered by the `tone_dither` renderer, which honours the
# requested colour count and palette — so "Colors: 4" produces genuine 4-colour
# dithered output (not 2-colour). The `style_tag` drives the style-filter chips:
#   dither   — ordered (Bayer) / blue-noise / Floyd-Steinberg (tone-accurate)
#   flow     — structure-aware flowing "waves" that follow image contours
#   flat     — flat N-colour poster (no dither)
CANDIDATES = [
    {
        "algorithm": "tone_dither",
        "label": "Flow (waves)",
        "description": "Waves that flow around the subject",
        "style_tag": "flow",
        "params": {"method": "flow", "frequency": 5.0, "warp": 7.0, "angle_deg": 45.0, "detail": 2.5},
    },
    {
        "algorithm": "tone_dither",
        "label": "Ordered (Bayer)",
        "description": "Crisp grid dither across N tones",
        "style_tag": "dither",
        "params": {"method": "bayer", "matrix_size": 8},
    },
    {
        "algorithm": "tone_dither",
        "label": "Blue Noise",
        "description": "Organic, grain-like dither",
        "style_tag": "dither",
        "params": {"method": "blue_noise", "noise_size": 64},
    },
    {
        "algorithm": "tone_dither",
        "label": "Floyd–Steinberg",
        "description": "Error-diffused, fine texture",
        "style_tag": "dither",
        "params": {"method": "floyd_steinberg"},
    },
    {
        "algorithm": "palette_quantize",
        "label": "Flat poster",
        "description": "Flat N-color, no dither",
        "style_tag": "flat",
        "params": {"dither": False},
    },
]

# Quick palette value -> tone_dither palette mode.
_PALETTE_MAP = {
    "adaptive": "adaptive",
    "grayscale": "grayscale",
    "ink_paper": "duotone",
}


def select_candidates(colors: int, palette: str, style_filter: list[str] | None = None) -> list[dict]:
    """Filter and parameterize candidate algorithms.

    tone_dither and palette_quantize cards are parameterized with the requested
    colour count and palette. The 1-bit halftone/stipple algorithms ignore the
    colour count by construction (the UI labels them "2-color").
    """
    filter_tags = None
    if style_filter:
        filter_tags = {t.lower() for t in style_filter if t.lower() != "all"}
        if not filter_tags:
            filter_tags = None

    tone_palette = _PALETTE_MAP.get(palette, "adaptive")

    selected = []
    for cand in CANDIDATES:
        style_tag = cand["style_tag"].lower()
        if filter_tags and style_tag not in filter_tags:
            continue

        params = dict(cand["params"])
        algo = cand["algorithm"]

        if algo == "tone_dither":
            params["colors"] = colors
            params["palette"] = tone_palette
        elif algo == "palette_quantize":
            if palette == "ink_paper":
                params["colors"] = 2
                params["palette"] = "adaptive"
            else:
                params["colors"] = colors
                params["palette"] = "grayscale" if palette == "grayscale" else "adaptive"

        selected.append({
            "algorithm": algo,
            "label": cand["label"],
            "description": cand["description"],
            "style_tag": cand["style_tag"],
            "params": params,
        })

    return selected
