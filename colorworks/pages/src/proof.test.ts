import { describe, expect, it } from "vitest";
import { computeProofMetrics, formatBytes } from "./proof";
import type { Raster, RenderResult } from "./colorworks";

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
    expect(computeProofMetrics(source, output)).toEqual({ usedColors: 2, perceptualError: 0, lowFrequencyError: 0 });
  });

  it("formats browser export sizes compactly", () => {
    expect(formatBytes(900)).toBe("900 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });
});
