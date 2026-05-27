# Pang-Style Structure-Aware Halftoning — Follow-Up Plan

Phase 3 delivered the iterative framework and CVT stippling. Pang halftoning
was deferred because it depends on SSE-quality orientation fields (Phase 2
ETF) and involves a multi-stage annealing loop with non-trivial energy
functions; adding it cleanly inside Phase 3 would have bloated scope.

## What Pang is

Pang et al. "Structure-Aware Halftoning" (2010): given a tone map and
orientation field, place halftone dots whose local density matches tone and
whose arrangement follows the orientation field. The energy function has two
terms:

- **Tone fidelity**: average dot density in a Gaussian-weighted neighbourhood
  matches tone_map at that pixel.
- **Orientation alignment**: dot displacement vectors are preferentially
  aligned with (or orthogonal to) the orientation field.

Annealing moves: propose a random dot position update; accept if it lowers
total energy (Metropolis criterion with temperature schedule).

## File to create

`colorworks/algorithms/pang_halftoning.py`

```
class PangHalftoning(IterativeAlgorithm):
    definition = DEFINITION   # family=HALFTONING, role=RENDERER
    ...
    def initialize(self, ctx):
        # Start from CVT stippling layout for warm init, or uniform random
        ...
    def step(self, ctx, it):
        # N Metropolis swap proposals, accept/reject
        # Returns ΔE (energy change, toward 0)
        ...
    def finalize(self, ctx, *, partial, warm_state):
        # Rasterize dots (Gaussian splat or filled circle)
        # Publish final_raster
        ...
    def can_warm_start(self, state, new_params):
        # OK if n_dots, ssim_window, w_t all close enough
        ...
```

## Parameters

| key | type | default | notes |
|---|---|---|---|
| n_dots | int | 500 | number of halftone dots |
| dot_radius | float | 2.0 | rendered dot radius (px) |
| max_iterations | int | 50 | annealing sweeps (each = N proposals) |
| temperature_start | float | 1.0 | initial annealing temperature |
| temperature_end | float | 0.01 | final temperature |
| w_tone | float | 1.0 | tone-fidelity weight |
| w_orient | float | 0.5 | orientation-alignment weight |
| ssim_window | int | 7 | Gaussian neighbourhood radius |
| orientation_source | str | "internal" | **internal only for now** — computes orientation field from the input image via structure tensor; other values reserved for future cross-run artifact borrowing (Phase 4+) |

## Acceptance checks (for whoever implements it)

1. Pang output visually differs from CVT stippling (dots follow orientation
   field directionality, not just density).
2. Deterministic under fixed seed: same params → same checksum.
3. Warm-start from a cancelled run reaches lower energy in ≤ half the
   iterations of a cold start.
4. Orientation source missing → raise `ValueError` with a user-safe message.
5. `IterationPreview(mode="direct_raster")` works; SSE emits energy each sweep.

## Dependencies

- **Phase 3.5 implementation is self-contained.** The algorithm computes its
  own tone map and structure-tensor orientation field from the input image,
  reusing Phase 2 helpers (`to_gray`, `gaussian_blur`, `convolve2d_nearest`).
  No prior analyzer run is required.
- Cross-run artifact borrowing (e.g. `orientation_run_id`) is reserved for a
  future phase (Phase 4+). The `orientation_source` parameter currently only
  accepts `"internal"` (or its aliases); other values raise `ValueError`.
- No new package dependencies; uses only numpy + PIL.

## Effort estimate

2–3 days: energy function + annealing loop + rasterizer + tests.
The IterativeAlgorithm framework is ready; only the algorithm-specific logic
needs writing.
