/**
 * Static parameter schema for the studio knobs.
 *
 * Replaces the Python server's `/api/schemas` endpoint: the browser app has no
 * backend, so the `tone_dither` parameter definitions are mirrored here by hand
 * (kept in sync with `colorworks/algorithms/tone_dither.py`). Everything the
 * studio offers routes through one renderer — `renderToneDither` — with the
 * "Flat poster" style selecting `method:"flat"` (nearest-colour assignment).
 */
import type { DitherMethod, PaletteMode } from "./colorworks";

export type ParamType = "int" | "float" | "str" | "bool";

export interface OptionDef {
  value: string | number;
  label: string;
}

export interface VisibleWhen {
  param: string;
  equals: Array<string | number>;
}

export interface ParamDef {
  key: string;
  label: string;
  type: ParamType;
  default: string | number | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: OptionDef[];
  group: "palette" | "pattern" | "tone";
  uiHint?: "color";
  /** Shown only when this holds — or, given several, when all of them hold. */
  visibleWhen?: VisibleWhen | VisibleWhen[];
}

/** Mirror of tone_dither.py DEFINITION.parameters (plus `flat` as a method). */
export const TONE_DITHER_PARAMS: ParamDef[] = [
  {
    key: "colors",
    label: "Colors",
    type: "int",
    default: 4,
    min: 2,
    max: 16,
    step: 1,
    group: "palette",
  },
  {
    key: "palette",
    label: "Palette",
    type: "str",
    default: "adaptive",
    group: "palette",
    options: [
      { value: "adaptive", label: "Adaptive (from image)" },
      { value: "grayscale", label: "Grayscale" },
      { value: "duotone", label: "Duotone (ink → paper)" },
      { value: "custom", label: "Exact custom inks" },
    ],
  },
  {
    key: "spread",
    label: "Palette spread",
    type: "float",
    // Pushes adaptive inks from cluster averages toward the image's real lights,
    // darks, and saturated colours, so dithering can mix all the way to them.
    // Measured on 5 photos (blurred-ΔE, i.e. at viewing distance) every dither
    // method improves at 0.25–0.5; past ~0.75 the wider ink spacing reads as
    // grain. Flat poster ignores it (no mixing), so it's hidden there.
    default: 0.35,
    min: 0,
    max: 1,
    step: 0.05,
    group: "palette",
    visibleWhen: [
      { param: "palette", equals: ["adaptive"] },
      {
        param: "method",
        equals: [
          "bayer", "blue_noise", "ign", "r2", "floyd_steinberg", "atkinson", "jarvis", "stucki",
          "burkes", "sierra", "ostromoukhov", "yliluoma", "knoll", "ema", "flow",
        ],
      },
    ],
  },
  {
    key: "method",
    label: "Dither Method",
    type: "str",
    default: "floyd_steinberg",
    group: "pattern",
    options: [
      { value: "bayer", label: "Ordered (Bayer)" },
      { value: "blue_noise", label: "Blue Noise" },
      { value: "ign", label: "Interleaved Gradient Noise" },
      { value: "r2", label: "R2 (golden ratio)" },
      { value: "floyd_steinberg", label: "Floyd–Steinberg" },
      { value: "atkinson", label: "Atkinson" },
      { value: "jarvis", label: "Jarvis–Judice–Ninke" },
      { value: "stucki", label: "Stucki" },
      { value: "burkes", label: "Burkes" },
      { value: "sierra", label: "Sierra" },
      { value: "ostromoukhov", label: "Ostromoukhov (variable coeff.)" },
      { value: "yliluoma", label: "Yliluoma (palette mix)" },
      { value: "knoll", label: "Knoll (pattern mix)" },
      { value: "ema", label: "Yliluoma EMA (exact)" },
      { value: "flow", label: "Flow (waves)" },
      { value: "flat", label: "Flat poster (no dither)" },
    ],
  },
  {
    key: "matrix_size",
    label: "Bayer Matrix Size",
    type: "int",
    default: 8,
    group: "pattern",
    options: [
      { value: 2, label: "2×2" },
      { value: 4, label: "4×4" },
      { value: 8, label: "8×8" },
      { value: 16, label: "16×16" },
    ],
    visibleWhen: { param: "method", equals: ["bayer", "knoll", "ema"] },
  },
  {
    key: "serpentine",
    label: "Serpentine scan",
    type: "bool",
    // On by default: a fixed left-to-right raster biases every diffusion kernel
    // and shows up as diagonal "worms". Turn off to reproduce pre-2026 renders.
    default: true,
    group: "pattern",
    visibleWhen: {
      param: "method",
      equals: [
        "floyd_steinberg", "atkinson", "jarvis", "stucki",
        "burkes", "sierra", "ostromoukhov",
      ],
    },
  },
  {
    key: "candidates",
    label: "Candidate Searches (N)",
    type: "int",
    // 0 = auto: the renderer uses the article's N ≈ 2×palette rule (min 16).
    // N drives noisiness more than any other knob — raise it if the mix looks grainy.
    default: 0,
    min: 0,
    max: 64,
    step: 1,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["knoll", "ema"] },
  },
  {
    key: "dither_strength",
    label: "Dither Strength",
    type: "float",
    // How far Knoll's goal point roams from the true colour. Low = candidates
    // hug the original (subtle); high = wider mixing, which small palettes need.
    default: 0.8,
    min: 0,
    max: 1,
    step: 0.05,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["knoll"] },
  },
  {
    key: "noise_size",
    label: "Blue Noise Size",
    type: "int",
    default: 64,
    group: "pattern",
    options: [
      { value: 16, label: "16×16" },
      { value: 32, label: "32×32" },
      { value: 64, label: "64×64" },
      { value: 128, label: "128×128" },
    ],
    visibleWhen: { param: "method", equals: ["blue_noise"] },
  },
  {
    key: "frequency",
    label: "Wave Density",
    type: "float",
    default: 5,
    min: 1,
    max: 24,
    step: 0.5,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["flow"] },
  },
  {
    key: "warp",
    label: "Flow Strength",
    type: "float",
    default: 7,
    min: 0,
    max: 20,
    step: 0.5,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["flow"] },
  },
  {
    key: "angle_deg",
    label: "Flow Angle (deg)",
    type: "float",
    default: 45,
    min: 0,
    max: 180,
    step: 1,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["flow"] },
  },
  {
    key: "detail",
    label: "Flow Detail",
    type: "float",
    default: 2.5,
    min: 0.5,
    max: 8,
    step: 0.5,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["flow"] },
  },
  {
    key: "contrast",
    label: "Contrast",
    type: "float",
    default: 1,
    min: 0.1,
    max: 3,
    step: 0.05,
    group: "tone",
  },
  {
    key: "midpoint",
    label: "Midpoint",
    type: "float",
    default: 0.5,
    min: 0,
    max: 1,
    step: 0.01,
    group: "tone",
  },
  {
    key: "ink_color",
    label: "Ink Color (duotone)",
    type: "str",
    default: "#161616",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "paper_color",
    label: "Paper Color (duotone)",
    type: "str",
    default: "#f4ebd9",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
];

export const HALFTONE_PARAMS: ParamDef[] = [
  {
    key: "separation", label: "Plates", type: "str", default: "spot", group: "palette",
    options: [
      { value: "spot", label: "Single spot ink" },
      { value: "cmyk", label: "CMYK process" },
    ],
  },
  {
    key: "screen", label: "Screen", type: "str", default: "am", group: "pattern",
    options: [
      { value: "am", label: "AM · dot size" },
      { value: "fm", label: "FM · dot frequency" },
      { value: "hybrid", label: "Hybrid · broken rosette" },
    ],
  },
  { key: "cell", label: "Screen cell (px)", type: "int", default: 8, min: 2, max: 24, step: 1, group: "pattern" },
  {
    key: "dot_shape", label: "Dot shape", type: "str", default: "round", group: "pattern",
    options: [
      { value: "round", label: "Round" },
      { value: "square", label: "Square" },
      { value: "line", label: "Line" },
    ],
    visibleWhen: { param: "screen", equals: ["am", "hybrid"] },
  },
  { key: "angle", label: "Spot angle (deg)", type: "float", default: 45, min: 0, max: 180, step: 1, group: "pattern", visibleWhen: { param: "separation", equals: ["spot"] } },
  { key: "dot_gain", label: "Dot gain", type: "float", default: 0, min: -0.2, max: 0.2, step: 0.01, group: "tone" },
  { key: "ink_color", label: "Spot ink", type: "str", default: "#1c1712", group: "palette", uiHint: "color", visibleWhen: { param: "separation", equals: ["spot"] } },
  { key: "paper_color", label: "Paper", type: "str", default: "#f4ebd9", group: "palette", uiHint: "color" },
];

/** Depixelate params (mirror of depixelate.py DEFINITION.parameters). */
export const DEPIXELATE_PARAMS: ParamDef[] = [
  {
    key: "block",
    label: "Tile size",
    type: "int",
    default: 2,
    min: 2,
    max: 6,
    step: 1,
    group: "pattern",
  },
  {
    key: "palette",
    label: "Palette",
    type: "str",
    default: "adaptive",
    group: "palette",
    options: [
      { value: "original", label: "Original colors" },
      { value: "adaptive", label: "Adaptive (from image)" },
      { value: "grayscale", label: "Grayscale" },
      { value: "duotone", label: "Duotone (ink → paper)" },
    ],
  },
  {
    key: "colors",
    label: "Colors",
    type: "int",
    default: 4,
    min: 2,
    max: 8,
    step: 1,
    group: "palette",
    visibleWhen: { param: "palette", equals: ["adaptive", "grayscale", "duotone"] },
  },
  {
    key: "ink_color",
    label: "Ink Color (duotone)",
    type: "str",
    default: "#161616",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "paper_color",
    label: "Paper Color (duotone)",
    type: "str",
    default: "#f4ebd9",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "fill_mult",
    label: "Fill multiplier",
    type: "float",
    default: 1,
    min: 0.25,
    max: 4,
    step: 0.25,
    group: "pattern",
  },
  {
    key: "keep_marks",
    label: "Keep sparse marks",
    type: "bool",
    default: false,
    group: "pattern",
  },
  {
    key: "tau",
    label: "Mark threshold",
    type: "int",
    default: 45,
    min: 0,
    max: 255,
    step: 1,
    group: "pattern",
    visibleWhen: { param: "palette", equals: ["original"] },
  },
  {
    key: "pitch",
    label: "Grid pitch (0 = auto)",
    type: "int",
    default: 0,
    min: 0,
    max: 64,
    step: 1,
    group: "pattern",
  },
];

/** Repixel params (mirror of repixel.ts RepixelOptions). */
export const REPIXEL_PARAMS: ParamDef[] = [
  {
    key: "target",
    label: "Pixel target",
    type: "str",
    // Which scale to lock onto. The image can carry two: a fine glyph/braille
    // lattice (background) and a coarser colour sprite (foreground). There is no
    // single "real" pixel size — pick the one this render is for. Composite is the
    // default: it recovers both at once (braille field + colour sprite) and is the
    // "good decode out of the box" for the braille-art screenshots this mode targets.
    default: "composite",
    group: "pattern",
    options: [
      { value: "fine", label: "Fine glyph field (background)" },
      { value: "subject", label: "Color subject pitch (sprite)" },
      { value: "composite", label: "Composite (background + sprite)" },
      { value: "manual", label: "Manual pitch" },
    ],
  },
  {
    key: "sprite_sat",
    label: "Sprite saturation",
    type: "float",
    // RELATIVE saturation (max-min)/max above which a pixel belongs to the colour
    // sprite (vs the grey/white braille background). Lower = grabs more of the
    // sprite, including its shadowed edges. Relative (not absolute) so dark
    // sprite pixels still enclose the silhouette around the eyes.
    default: 0.3,
    min: 0.1,
    max: 0.7,
    step: 0.05,
    group: "pattern",
    visibleWhen: { param: "target", equals: ["composite"] },
  },
  {
    key: "eye_luma",
    label: "Eye darkness",
    type: "int",
    // Luma below which a pixel INSIDE the sprite body counts as a dark eye/detail.
    // Near-black eyes read as high saturation, so they're found by luma, not colour.
    default: 45,
    min: 0,
    max: 150,
    step: 5,
    group: "pattern",
    visibleWhen: { param: "target", equals: ["composite"] },
  },
  {
    key: "pitch",
    label: "Grid pitch (px)",
    type: "int",
    default: 0,
    min: 0,
    max: 40,
    step: 1,
    group: "pattern",
    visibleWhen: { param: "target", equals: ["manual"] },
  },
  {
    key: "tau",
    label: "Foreground threshold",
    type: "int",
    default: 45,
    min: 0,
    max: 255,
    step: 1,
    group: "pattern",
  },
  {
    key: "min_lit",
    label: "Min lit pixels",
    type: "int",
    default: 2,
    min: 1,
    max: 8,
    step: 1,
    group: "pattern",
  },
  {
    key: "shade",
    label: "Shade dots (halftone)",
    type: "bool",
    // ON: each cell is the area-average of its pixels, so size-modulated halftone
    // dots (e.g. a dotted sphere) recover as a smooth tonal gradient and solid
    // sprites stay full-colour. OFF: flat two-tone (crisp ink vs background).
    default: true,
    group: "pattern",
    // Composite forces a crisp background internally (its background is a dither
    // field), so the toggle would be a no-op there — hide it for that target.
    visibleWhen: { param: "target", equals: ["fine", "subject", "manual"] },
  },
  {
    key: "palette",
    label: "Palette",
    type: "str",
    // "Original colors" is the faithful default — with shading on, the recovered
    // tones are meaningful gradient, not noise. "Adaptive" snaps to a small ink
    // palette for a cleaner / poster look.
    default: "original",
    group: "palette",
    options: [
      { value: "original", label: "Original colors" },
      { value: "adaptive", label: "Adaptive (from image)" },
      { value: "grayscale", label: "Grayscale" },
      { value: "duotone", label: "Duotone (ink → paper)" },
    ],
  },
  {
    key: "colors",
    label: "Colors",
    type: "int",
    default: 6,
    min: 2,
    max: 12,
    step: 1,
    group: "palette",
    visibleWhen: { param: "palette", equals: ["adaptive", "grayscale", "duotone"] },
  },
  {
    key: "ink_color",
    label: "Ink Color (duotone)",
    type: "str",
    default: "#161616",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "paper_color",
    label: "Paper Color (duotone)",
    type: "str",
    default: "#f4ebd9",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "bg_mode",
    label: "Background",
    type: "str",
    default: "auto",
    group: "pattern",
    options: [
      { value: "auto", label: "Auto-detect" },
      { value: "custom", label: "Custom color" },
    ],
  },
  {
    key: "bg_color",
    label: "Background color",
    type: "str",
    default: "#181818",
    group: "pattern",
    uiHint: "color",
    visibleWhen: { param: "bg_mode", equals: ["custom"] },
  },
];

/** Block mosaic params (mirror of blockmosaic.ts BlockMosaicOptions). */
export const BLOCK_MOSAIC_PARAMS: ParamDef[] = [
  {
    key: "block",
    label: "Block size",
    type: "int",
    default: 2,
    group: "pattern",
    options: [
      { value: 2, label: "2×2" },
      { value: 3, label: "3×3" },
      { value: 4, label: "4×4" },
    ],
  },
  {
    key: "cell",
    label: "Cell size (px)",
    type: "int",
    default: 6,
    min: 2,
    max: 16,
    step: 1,
    group: "pattern",
  },
  {
    key: "method",
    label: "Fill mode",
    type: "str",
    default: "match",
    group: "pattern",
    options: [
      { value: "match", label: "Match — clean structure" },
      { value: "diffuse", label: "Diffuse — tonal accuracy" },
    ],
  },
  {
    key: "library",
    label: "Block library",
    type: "str",
    // Which preset tiles seed the candidate set. "none" leans entirely on the
    // learned blocks (a pure vector-quantisation mosaic).
    default: "auto",
    group: "pattern",
    options: [
      { value: "auto", label: "Auto (solids + checkers + diagonals)" },
      { value: "solids", label: "Solids only" },
      { value: "checker", label: "Solids + checkers" },
      { value: "none", label: "Learned only" },
    ],
  },
  {
    key: "learn",
    label: "Learned blocks",
    type: "int",
    // K data-driven blocks discovered from the image by k-means. 0 = presets only.
    default: 6,
    min: 0,
    max: 16,
    step: 1,
    group: "pattern",
  },
  {
    key: "library_bias",
    label: "Prefer library",
    type: "float",
    // 0 = neutral (best block wins); 1 = strongly prefer preset blocks, falling to
    // a learned block only when it's materially better.
    default: 0.5,
    min: 0,
    max: 1,
    step: 0.05,
    group: "pattern",
  },
  {
    key: "palette",
    label: "Palette",
    type: "str",
    default: "adaptive",
    group: "palette",
    options: [
      { value: "adaptive", label: "Adaptive (from image)" },
      { value: "grayscale", label: "Grayscale" },
      { value: "duotone", label: "Duotone (ink → paper)" },
    ],
  },
  {
    key: "colors",
    label: "Colors",
    type: "int",
    default: 4,
    min: 2,
    max: 8,
    step: 1,
    group: "palette",
    // Feeds the preset library; irrelevant when running on learned blocks alone.
    visibleWhen: { param: "library", equals: ["auto", "solids", "checker"] },
  },
  {
    key: "ink_color",
    label: "Ink Color (duotone)",
    type: "str",
    default: "#161616",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "paper_color",
    label: "Paper Color (duotone)",
    type: "str",
    default: "#f4ebd9",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "contrast",
    label: "Contrast",
    type: "float",
    default: 1,
    min: 0.1,
    max: 3,
    step: 0.05,
    group: "tone",
  },
  {
    key: "midpoint",
    label: "Midpoint",
    type: "float",
    default: 0.5,
    min: 0,
    max: 1,
    step: 0.01,
    group: "tone",
  },
];

/** Knit chart params (chart dimensions live in the shared Output Size panel). */
export const KNIT_PARAMS: ParamDef[] = [
  {
    key: "row_gauge",
    label: "Row gauge (rows ÷ sts)",
    type: "float",
    // A stockinette stitch is wider than tall (more rows/inch than sts/inch), so
    // rows are inflated by this ratio and the knitted fabric matches the source.
    // 1.0 = square chart (no correction). 1.4 ≈ typical stockinette.
    default: 1.4,
    min: 0.8,
    max: 2,
    step: 0.05,
    group: "pattern",
  },
  {
    key: "cell",
    label: "Cell size (px)",
    type: "int",
    default: 8,
    min: 4,
    max: 16,
    step: 1,
    group: "pattern",
  },
  {
    key: "method",
    label: "Dither method",
    type: "str",
    // Error diffusion gives the salt-and-pepper colorwork in flat-tone regions and
    // solid runs in saturated ones; "flat" is solid blocks only (no tone).
    default: "floyd_steinberg",
    group: "pattern",
    options: [
      { value: "floyd_steinberg", label: "Floyd–Steinberg (fine grain)" },
      { value: "atkinson", label: "Atkinson (sparse, crisp)" },
      { value: "stucki", label: "Stucki (smooth)" },
      { value: "yliluoma", label: "Yliluoma (palette mix)" },
      { value: "flat", label: "Flat (solid blocks)" },
    ],
  },
  {
    key: "grid",
    label: "Chart grid",
    type: "str",
    default: "decades",
    group: "pattern",
    options: [
      { value: "decades", label: "Lines + bold every 10" },
      { value: "lines", label: "Thin lines" },
      { value: "none", label: "None (cells only)" },
    ],
  },
  {
    key: "symbols",
    label: "Chart symbols",
    type: "str",
    default: "cells",
    group: "pattern",
    options: [
      { value: "cells", label: "Cells + legend" },
      { value: "legend", label: "Legend only" },
      { value: "none", label: "None" },
    ],
  },
  {
    key: "tidy",
    label: "Merge lone stitches",
    type: "bool",
    // Off by default: a fully-isolated stitch is merged into its surroundings, but
    // a deliberate sparse contrast dot is locally identical to a stray, so this
    // also erases the faintest 1-stitch dots. On = cleaner, more solid chart.
    default: false,
    group: "pattern",
  },
  {
    key: "palette",
    label: "Yarn palette",
    type: "str",
    default: "adaptive",
    group: "palette",
    options: [
      { value: "adaptive", label: "Adaptive (from image)" },
      { value: "grayscale", label: "Grayscale" },
      { value: "duotone", label: "Duotone (ink → paper)" },
    ],
  },
  {
    key: "colors",
    label: "Yarn count",
    type: "int",
    // 5 (not 4) by default: the extra yarn lets adaptive allocate a mid-grey for
    // a dark textured field, so a halftone background dithers into GREY dots
    // instead of borrowing a saturated foreground yarn. Still very knittable.
    default: 5,
    min: 2,
    max: 8,
    step: 1,
    group: "palette",
  },
  {
    key: "ink_color",
    label: "Ink Color (duotone)",
    type: "str",
    default: "#161616",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "paper_color",
    label: "Paper Color (duotone)",
    type: "str",
    default: "#f4ebd9",
    group: "palette",
    uiHint: "color",
    visibleWhen: { param: "palette", equals: ["duotone"] },
  },
  {
    key: "contrast",
    label: "Contrast",
    type: "float",
    default: 1,
    min: 0.1,
    max: 3,
    step: 0.05,
    group: "tone",
  },
  {
    key: "midpoint",
    label: "Midpoint",
    type: "float",
    default: 0.5,
    min: 0,
    max: 1,
    step: 0.01,
    group: "tone",
  },
];

export type RendererId = "tone_dither" | "halftone" | "depixelate" | "repixel" | "block_mosaic" | "knit";
export type StyleIntent = "faithful" | "organic" | "ordered" | "print" | "specialty";

export interface StyleDef {
  id: string;
  label: string;
  description: string;
  intent: StyleIntent;
  /** Which renderer this style drives (defaults to tone_dither). */
  renderer?: RendererId;
  /** Which param set this style exposes (defaults to TONE_DITHER_PARAMS). */
  params?: ParamDef[];
  /** Parameters fixed by this style (hidden from the knob list). */
  fixed: Record<string, string | number | boolean>;
}

/** Curated style set (mirrors quick_mode.CANDIDATES + the studio style picker). */
export const STYLES: StyleDef[] = [
  { id: "knoll", label: "Knoll — faithful palette mix", description: "Stable ordered mixing for a small fixed palette", intent: "faithful", fixed: { method: "knoll" } },
  { id: "yliluoma", label: "Yliluoma — gamma-aware mix", description: "Ordered palette mixing in linear light", intent: "faithful", fixed: { method: "yliluoma" } },
  { id: "ema", label: "Yliluoma EMA — exact blend", description: "Experimental exact-segment palette mixing", intent: "faithful", fixed: { method: "ema" } },
  { id: "flat", label: "Flat poster — no dither", description: "Nearest-color assignment without texture", intent: "faithful", fixed: { method: "flat" } },
  { id: "floyd_steinberg", label: "Floyd–Steinberg — fine grain", description: "Balanced error-diffused texture", intent: "organic", fixed: { method: "floyd_steinberg" } },
  { id: "blue_noise", label: "Blue noise — even grain", description: "Organic-looking grain without diffusion trails", intent: "organic", fixed: { method: "blue_noise" } },
  { id: "atkinson", label: "Atkinson — sparse contrast", description: "Punchy, sparse Macintosh-style grain", intent: "organic", fixed: { method: "atkinson" } },
  { id: "jarvis", label: "Jarvis–Judice–Ninke — wide diffusion", description: "Soft gradients from a wide error kernel", intent: "organic", fixed: { method: "jarvis" } },
  { id: "stucki", label: "Stucki — clean diffusion", description: "Wide, restrained error diffusion", intent: "organic", fixed: { method: "stucki" } },
  { id: "burkes", label: "Burkes — balanced diffusion", description: "Two-row balance of edge detail and grain", intent: "organic", fixed: { method: "burkes" } },
  { id: "sierra", label: "Sierra — crisp diffusion", description: "Three-row diffusion with retained edges", intent: "organic", fixed: { method: "sierra" } },
  { id: "ostromoukhov", label: "Ostromoukhov — tone-aware diffusion", description: "Variable coefficients reduce recurring worms", intent: "organic", fixed: { method: "ostromoukhov" } },
  { id: "bayer", label: "Ordered — Bayer grid", description: "Crisp, repeatable threshold pattern", intent: "ordered", fixed: { method: "bayer" } },
  { id: "flow", label: "Flow — image-warped waves", description: "Graphic bands that bend around the subject", intent: "ordered", fixed: { method: "flow" } },
  { id: "ign", label: "Interleaved gradient noise", description: "Fine diagonal structure without a stored tile", intent: "ordered", fixed: { method: "ign" } },
  { id: "r2", label: "R2 — quasirandom grain", description: "Non-repeating golden-ratio threshold sequence", intent: "ordered", fixed: { method: "r2" } },
  { id: "halftone", label: "Print screen — spot or CMYK", description: "AM, FM, or hybrid print screens with visible plates", intent: "print", renderer: "halftone", params: HALFTONE_PARAMS, fixed: {} },
  {
    id: "depixelate",
    label: "Depixelate — recover grid",
    description: "Recover the native pixel grid of an upscaled image; re-render each cell as a 2-colour dither tile",
    intent: "specialty",
    renderer: "depixelate",
    params: DEPIXELATE_PARAMS,
    fixed: {},
  },
  {
    id: "block_mosaic",
    label: "Block mosaic — tile from blocks",
    description:
      "Fill the image with multi-colour block tiles — preset blocks built from the palette plus learned blocks discovered from the image; best block wins per region",
    intent: "specialty",
    renderer: "block_mosaic",
    params: BLOCK_MOSAIC_PARAMS,
    // `method` is left as a visible knob (the match↔diffuse toggle).
    fixed: {},
  },
  {
    id: "knit",
    label: "Knit chart — stranded colorwork",
    description:
      "Turn the image into a knitting chart: 1 pixel = 1 stitch, quantised to a few yarns and dithered so flat tones become salt-and-pepper colorwork and solid shapes become solid runs; gauge-corrected rows, baked grid, and a yarn legend",
    intent: "specialty",
    renderer: "knit",
    params: KNIT_PARAMS,
    fixed: {},
  },
  // NOTE: "Glyph art" (repixel) is intentionally unlisted — it doesn't work well
  // yet. The renderer (repixel.ts), REPIXEL_PARAMS, and the studio wiring are kept
  // so it can be re-enabled by restoring a StyleDef here with renderer:"repixel".
];

export const DEFAULT_STYLE_ID = "floyd_steinberg";

/** The param set a style exposes (its own, or the tone-dither default). */
export function styleParams(style: StyleDef): ParamDef[] {
  return style.params ?? TONE_DITHER_PARAMS;
}

/** Convenience: the param defs keyed for lookup. */
export const PARAM_BY_KEY: Record<string, ParamDef> = Object.fromEntries(
  [
    ...TONE_DITHER_PARAMS,
    ...HALFTONE_PARAMS,
    ...DEPIXELATE_PARAMS,
    ...REPIXEL_PARAMS,
    ...BLOCK_MOSAIC_PARAMS,
    ...KNIT_PARAMS,
  ].map((p) => [p.key, p]),
);

export type { DitherMethod, PaletteMode };
