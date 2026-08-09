import { parseColor, rasterToGray, type RGB, type Raster, type RenderResult } from "./colorworks";

export type ScreenMode = "am" | "fm" | "hybrid";
export type SeparationMode = "spot" | "cmyk";
export type DotShape = "round" | "square" | "line";

export interface HalftoneOptions {
  separation?: SeparationMode;
  screen?: ScreenMode;
  cell?: number;
  angle?: number;
  dotShape?: DotShape;
  dotGain?: number;
  inkColor?: string;
  paperColor?: string;
  seed?: number;
}

export interface PlateInfo {
  name: string;
  color: RGB;
  angle: number;
}

export interface HalftoneResult extends RenderResult {
  plates: PlateInfo[];
  screen: ScreenMode;
  cell: number;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const fract = (v: number) => v - Math.floor(v);

function hash01(x: number, y: number, seed: number): number {
  let n = Math.imul(x | 0, 0x1f123bb5) ^ Math.imul(y | 0, 0x5f356495) ^ Math.imul(seed | 0, 0x6c8e9cf5);
  n ^= n >>> 15;
  n = Math.imul(n, 0x2c1b3c6d);
  n ^= n >>> 12;
  n = Math.imul(n, 0x297a2d39);
  n ^= n >>> 15;
  return (n >>> 0) / 4294967296;
}

/** Threshold of a centered, rotated AM screen cell. Lower values fill first. */
function amThreshold(x: number, y: number, cell: number, angle: number, shape: DotShape): number {
  const a = angle * Math.PI / 180;
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  const u = fract((x * ca + y * sa) / cell) - 0.5;
  const v = fract((-x * sa + y * ca) / cell) - 0.5;
  if (shape === "line") return clamp01(2 * Math.abs(u));
  if (shape === "square") return clamp01(4 * Math.max(u * u, v * v));
  return clamp01(Math.PI * (u * u + v * v));
}

function screenOn(
  density: number,
  x: number,
  y: number,
  plate: number,
  opts: Required<Pick<HalftoneOptions, "screen" | "cell" | "angle" | "dotShape" | "dotGain" | "seed">>,
  plateAngle: number,
): boolean {
  const d = clamp01(density + opts.dotGain);
  if (d <= 0) return false;
  if (d >= 0.999) return true;
  const am = amThreshold(x + 0.5, y + 0.5, opts.cell, plateAngle, opts.dotShape);
  const fm = hash01(x, y, opts.seed + plate * 977);
  const threshold = opts.screen === "am" ? am : opts.screen === "fm" ? fm : 0.62 * am + 0.38 * fm;
  return threshold < d;
}

function compositeInk(paper: RGB, plates: PlateInfo[], mask: number): RGB {
  let r = paper[0] / 255;
  let g = paper[1] / 255;
  let b = paper[2] / 255;
  for (let i = 0; i < plates.length; i++) {
    if (!(mask & (1 << i))) continue;
    r *= plates[i].color[0] / 255;
    g *= plates[i].color[1] / 255;
    b *= plates[i].color[2] / 255;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Print-screen renderer. AM varies centered dot area; FM varies dot frequency;
 * Hybrid keeps an AM skeleton but breaks up the most obvious rosettes.
 */
export function renderHalftone(raster: Raster, options: HalftoneOptions = {}): HalftoneResult {
  const separation = options.separation ?? "spot";
  const screen = options.screen ?? "am";
  const cell = Math.max(2, Math.min(32, Math.round(options.cell ?? 8)));
  const angle = options.angle ?? 45;
  const dotShape = options.dotShape ?? "round";
  const dotGain = Math.max(-0.25, Math.min(0.25, options.dotGain ?? 0));
  const seed = options.seed ?? 42;
  const paper = parseColor(options.paperColor ?? "#f4ebd9");

  const plates: PlateInfo[] = separation === "cmyk"
    ? [
        { name: "C", color: [0, 174, 239], angle: 15 },
        { name: "M", color: [236, 0, 140], angle: 75 },
        { name: "Y", color: [255, 241, 0], angle: 0 },
        { name: "K", color: [28, 23, 18], angle: 45 },
      ]
    : [{ name: "Spot", color: parseColor(options.inkColor ?? "#1c1712"), angle }];

  const { width: w, height: h, data } = raster;
  const gray = separation === "spot" ? rasterToGray(raster) : null;
  const indices = new Uint16Array(w * h);
  const maskToIndex = new Map<number, number>();
  const palette: RGB[] = [];
  const densities = new Float32Array(plates.length);
  const paletteIndex = (mask: number): number => {
    const hit = maskToIndex.get(mask);
    if (hit !== undefined) return hit;
    const idx = palette.length;
    maskToIndex.set(mask, idx);
    palette.push(compositeInk(paper, plates, mask));
    return idx;
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      densities.fill(0);
      if (separation === "spot") {
        densities[0] = 1 - gray![i];
      } else {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;
        const k = Math.min(1 - r, 1 - g, 1 - b);
        const denom = Math.max(1e-6, 1 - k);
        densities[0] = clamp01((1 - r - k) / denom);
        densities[1] = clamp01((1 - g - k) / denom);
        densities[2] = clamp01((1 - b - k) / denom);
        densities[3] = k;
      }
      let mask = 0;
      for (let p = 0; p < plates.length; p++) {
        if (screenOn(densities[p], x, y, p, { screen, cell, angle, dotShape, dotGain, seed }, plates[p].angle)) {
          mask |= 1 << p;
        }
      }
      indices[i] = paletteIndex(mask);
    }
  }

  return { indices, palette, width: w, height: h, plates, screen, cell };
}
