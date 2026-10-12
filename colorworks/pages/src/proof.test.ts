import { describe, expect, it } from "vitest";
import { computeProofMetrics, formatBytes } from "./proof";
import { renderToneDither, type Raster, type RenderResult } from "./colorworks";

function solid(w: number, h: number, v: number): Raster {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set([v, v, v, 255], i * 4);
  return { width: w, height: h, data };
}

describe("proof receipt", () => {
  it("reports zero error for an exact reconstruction", () => {
    const source: Raster = {
      width: 2, height: 1,
      data: new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]),
    };
    const output: RenderResult = {
      width: 2, height: 1,
      indices: new Uint16Array([0, 1]),
      palette: [[0, 0, 0], [255, 255, 255]],
    };
    expect(computeProofMetrics(source, output)).toEqual({ usedColors: 2, viewingError: 0, lowFrequencyError: 0 });
  });

  it("scores a checkerboard as the grey it mixes to, and a flat fill as far off", () => {
    // sRGB 188 is ~50% linear light: what an even black/white checkerboard
    // looks like from a distance.
    const w = 24, h = 24;
    const source = solid(w, h, 188);
    const palette: RenderResult["palette"] = [[0, 0, 0], [255, 255, 255]];
    const checker = new Uint16Array(w * h).map((_, i) => ((i % w) + ((i / w) | 0)) % 2);
    const dithered = computeProofMetrics(source, { width: w, height: h, indices: checker, palette })!;
    const flat = computeProofMetrics(source, { width: w, height: h, indices: new Uint16Array(w * h).fill(1), palette })!;
    expect(dithered.viewingError).toBeLessThan(0.5);
    expect(flat.viewingError).toBeGreaterThan(15);
  });

  it("ranks a 2-colour dither of a gradient above a flat 2-colour poster", () => {
    const w = 96, h = 32;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = Math.round((x / (w - 1)) * 255);
      data.set([v, v, v, 255], (y * w + x) * 4);
    }
    const source: Raster = { width: w, height: h, data };
    const opts = { colors: 2, palette: "grayscale" as const, seed: 1 };
    const fs = computeProofMetrics(source, renderToneDither(source, { ...opts, method: "floyd_steinberg" }))!;
    const flat = computeProofMetrics(source, renderToneDither(source, { ...opts, method: "flat" }))!;
    // Not by as much as it should: this diffusion runs in sRGB, so its midtones
    // mix lighter than the source in linear light — the metric shows that too.
    expect(fs.viewingError).toBeLessThan(flat.viewingError * 0.8);
  });

  it("formats browser export sizes compactly", () => {
    expect(formatBytes(900)).toBe("900 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });
});
