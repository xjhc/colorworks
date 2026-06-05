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
  visibleWhen?: VisibleWhen;
}

/** Mirror of tone_dither.py DEFINITION.parameters (plus `flat` as a method). */
export const TONE_DITHER_PARAMS: ParamDef[] = [
  {
    key: "colors",
    label: "Colors",
    type: "int",
    default: 4,
    min: 2,
    max: 8,
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
    ],
  },
  {
    key: "method",
    label: "Dither Method",
    type: "str",
    default: "bayer",
    group: "pattern",
    options: [
      { value: "bayer", label: "Ordered (Bayer)" },
      { value: "blue_noise", label: "Blue Noise" },
      { value: "floyd_steinberg", label: "Floyd–Steinberg" },
      { value: "flow", label: "Flow (waves)" },
      { value: "maze", label: "Maze (labyrinth)" },
      { value: "flat", label: "Flat poster (no dither)" },
    ],
  },
  {
    key: "mask_scale",
    label: "Maze Cell Size (px)",
    type: "float",
    default: 5,
    min: 2,
    max: 24,
    step: 0.5,
    group: "pattern",
    visibleWhen: { param: "method", equals: ["maze"] },
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
    visibleWhen: { param: "method", equals: ["bayer"] },
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

export type RendererId = "tone_dither" | "depixelate";

export interface StyleDef {
  id: string;
  label: string;
  description: string;
  /** Which renderer this style drives (defaults to tone_dither). */
  renderer?: RendererId;
  /** Which param set this style exposes (defaults to TONE_DITHER_PARAMS). */
  params?: ParamDef[];
  /** Parameters fixed by this style (hidden from the knob list). */
  fixed: Record<string, string | number | boolean>;
}

/** Curated style set (mirrors quick_mode.CANDIDATES + the studio style picker). */
export const STYLES: StyleDef[] = [
  { id: "flow", label: "Flow — flowing waves", description: "Waves that flow around the subject", fixed: { method: "flow" } },
  { id: "bayer", label: "Ordered — Bayer grid", description: "Crisp grid dither across N tones", fixed: { method: "bayer" } },
  { id: "blue_noise", label: "Blue noise — grain", description: "Organic, grain-like dither", fixed: { method: "blue_noise" } },
  { id: "floyd_steinberg", label: "Floyd–Steinberg", description: "Error-diffused, fine texture", fixed: { method: "floyd_steinberg" } },
  { id: "maze", label: "Maze — labyrinth", description: "Connected diagonal labyrinth", fixed: { method: "maze" } },
  { id: "flat", label: "Flat poster — no dither", description: "Flat N-colour poster", fixed: { method: "flat" } },
  {
    id: "depixelate",
    label: "Depixelate — recover grid",
    description: "Recover the native pixel grid of an upscaled image; re-render each cell as a 2-colour dither tile",
    renderer: "depixelate",
    params: DEPIXELATE_PARAMS,
    fixed: {},
  },
];

export const DEFAULT_STYLE_ID = "flow";

/** The param set a style exposes (its own, or the tone-dither default). */
export function styleParams(style: StyleDef): ParamDef[] {
  return style.params ?? TONE_DITHER_PARAMS;
}

/** Convenience: the param defs keyed for lookup. */
export const PARAM_BY_KEY: Record<string, ParamDef> = Object.fromEntries(
  [...TONE_DITHER_PARAMS, ...DEPIXELATE_PARAMS].map((p) => [p.key, p]),
);

export type { DitherMethod, PaletteMode };
