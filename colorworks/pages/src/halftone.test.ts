import { describe, expect, it } from "vitest";
import { renderHalftone } from "./halftone";
import type { Raster } from "./colorworks";

function solid(w: number, h: number, rgb: [number, number, number]): Raster {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data.set([...rgb, 255], i * 4);
  }
  return { width: w, height: h, data };
}

describe("print halftone", () => {
  it("maps paper white and full black to solid spot endpoints", () => {
    const white = renderHalftone(solid(16, 16, [255, 255, 255]), { separation: "spot", inkColor: "#112233", paperColor: "#fefefe" });
    const black = renderHalftone(solid(16, 16, [0, 0, 0]), { separation: "spot", inkColor: "#112233", paperColor: "#fefefe" });
    expect(new Set(white.indices).size).toBe(1);
    expect(white.palette[white.indices[0]]).toEqual([254, 254, 254]);
    expect(new Set(black.indices).size).toBe(1);
    expect(black.palette[black.indices[0]]).toEqual([17, 34, 51]);
  });

  it("is deterministic and produces a mixed AM screen for midtones", () => {
    const raster = solid(32, 32, [128, 128, 128]);
    const a = renderHalftone(raster, { screen: "am", cell: 8, angle: 45 });
    const b = renderHalftone(raster, { screen: "am", cell: 8, angle: 45 });
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
    expect(new Set(a.indices).size).toBe(2);
  });

  it("builds four standard process plates and at most sixteen overprints", () => {
    const raster = solid(24, 24, [80, 130, 210]);
    const out = renderHalftone(raster, { separation: "cmyk", screen: "hybrid", cell: 7 });
    expect(out.plates.map((p) => p.name)).toEqual(["C", "M", "Y", "K"]);
    expect(out.palette.length).toBeLessThanOrEqual(16);
    for (const i of out.indices) expect(i).toBeLessThan(out.palette.length);
  });
});
