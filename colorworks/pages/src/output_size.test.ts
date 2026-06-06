import { describe, it, expect } from "vitest";
import { type RGB } from "./colorworks";
import { boxFit, conformIndexed } from "./output_size";
import { renderDepixelate } from "./depixelate";

/** A small indexed render (the shape depixelate / renderToneDither return). */
function indexed(w: number, h: number, palette: RGB[]): { indices: Uint16Array; palette: RGB[]; width: number; height: number } {
  const indices = new Uint16Array(w * h);
  for (let i = 0; i < indices.length; i++) indices[i] = i % palette.length;
  return { indices, palette, width: w, height: h };
}

describe("boxFit", () => {
  it("downscales to the max width, preserving aspect (fit)", () => {
    const b = boxFit(1000, 750, 200, null, "fit", false);
    expect([b.canvasW, b.canvasH]).toEqual([200, 150]);
  });

  it("never upscales in fit mode unless allowed", () => {
    expect(boxFit(150, 150, 200, null, "fit", false).canvasW).toBe(150); // capped
    expect(boxFit(150, 150, 200, null, "fit", true).canvasW).toBe(200); // lifted
  });

  it("stretch hits the exact box; cover fills then centre-crops", () => {
    expect(boxFit(1000, 750, 200, 200, "stretch", false)).toMatchObject({ canvasW: 200, canvasH: 200 });
    const cover = boxFit(1000, 750, 200, 200, "cover", false);
    expect([cover.canvasW, cover.canvasH]).toEqual([200, 200]);
    expect(cover.dx).toBeLessThanOrEqual(0); // overflow is cropped, not letterboxed
  });

  it("is a no-op when no size is set", () => {
    expect(boxFit(640, 480, null, null, "fit", true)).toMatchObject({ canvasW: 640, canvasH: 480 });
  });
});

describe("conformIndexed", () => {
  it("returns the input untouched when no size is set", () => {
    const res = indexed(12, 12, [[0, 0, 0], [255, 255, 255]]);
    expect(conformIndexed(res, null, null, "fit")).toBe(res);
  });

  it("scales a small native grid up to the requested width (depixelate case)", () => {
    const res = indexed(12, 12, [[0, 0, 0], [255, 255, 255]]);
    const out = conformIndexed(res, 200, null, "fit");
    expect([out.width, out.height]).toEqual([200, 200]);
    expect(out.indices.length).toBe(200 * 200);
    expect(out.palette).toBe(res.palette); // palette preserved exactly
    for (const ix of out.indices) expect(ix).toBeLessThan(out.palette.length);
  });

  it("preserves aspect for a non-square native grid", () => {
    const res = indexed(8, 12, [[0, 0, 0], [255, 255, 255]]); // 2:3
    const out = conformIndexed(res, 200, null, "fit");
    expect([out.width, out.height]).toEqual([200, 300]);
  });

  it("nearest-neighbour keeps every index inside the source palette (no blending)", () => {
    const palette: RGB[] = [[10, 20, 30], [40, 50, 60], [70, 80, 90]];
    const res = indexed(5, 5, palette);
    const out = conformIndexed(res, 137, null, "stretch");
    const used = new Set(out.indices);
    for (const ix of used) expect(ix).toBeLessThan(palette.length);
  });
});

describe("depixelate honours the output size end-to-end", () => {
  /** Build an upscaled 6×6 native sprite, like the depixelate suite uses. */
  function upscaledSprite(factor: number) {
    const palette: RGB[] = [[20, 20, 20], [230, 230, 230], [210, 90, 60], [60, 120, 200]];
    const native = 6;
    const w = native * factor;
    const data = new Uint8ClampedArray(w * w * 4);
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const c = palette[(((x / factor) | 0) * 2 + ((y / factor) | 0) * 3) % palette.length];
        const i = (y * w + x) * 4;
        data[i] = c[0];
        data[i + 1] = c[1];
        data[i + 2] = c[2];
        data[i + 3] = 255;
      }
    }
    return { width: w, height: w, data };
  }

  it("raw output is the native grid; conformed output matches the requested size", () => {
    const r = upscaledSprite(10); // 60×60 → native 6×6 → block 2 → 12×12
    const res = renderDepixelate(r, { block: 2, pitch: 10 });
    expect([res.width, res.height]).toEqual([12, 12]); // decoupled from output size

    const out = conformIndexed(res, 200, null, "fit");
    expect([out.width, out.height]).toEqual([200, 200]); // honours "200px"
  });
});
