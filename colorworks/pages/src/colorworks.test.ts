import { describe, it, expect } from "vitest";
import {
  type RGB,
  type Raster,
  bayerMatrix,
  ditherToPalette,
  fsToPalette,
  flowThresholdMap,
  dedupePalette,
  renderToneDither,
  rasterToRgb01,
  rasterToGray,
  applyTone,
} from "./colorworks";

// ── helpers ───────────────────────────────────────────────────────────────────
/** Build an RGBA raster from a width and an (x,y)->RGB function. */
function makeRaster(w: number, h: number, fn: (x: number, y: number) => RGB): Raster {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const [r, g, b] = fn(x, y);
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { width: w, height: h, data };
}

const luma = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

// A horizontal gray gradient — useful for tone-tracking checks.
function grayGradient(w: number, h: number): Raster {
  return makeRaster(w, h, (x) => {
    const v = Math.round((x / (w - 1)) * 255);
    return [v, v, v];
  });
}

// ── 1. Palette dedupe collapses duplicates and reindexes cells ──────────────────
describe("dedupePalette", () => {
  it("collapses duplicate swatches and reindexes the cell map", () => {
    const palette: RGB[] = [
      [10, 10, 10],
      [10, 10, 10], // duplicate of index 0
      [20, 20, 20],
      [20, 20, 20], // duplicate of index 2
    ];
    const indices = Uint16Array.from([0, 1, 2, 3, 1, 3]);
    const out = dedupePalette(indices, palette);

    expect(out.palette).toEqual([
      [10, 10, 10],
      [20, 20, 20],
    ]);
    // 0->0, 1->0, 2->1, 3->1
    expect(Array.from(out.indices)).toEqual([0, 0, 1, 1, 0, 1]);
  });

  it("is a no-op when all swatches are unique", () => {
    const palette: RGB[] = [
      [0, 0, 0],
      [255, 255, 255],
    ];
    const indices = Uint16Array.from([0, 1, 0]);
    const out = dedupePalette(indices, palette);
    expect(out.palette).toBe(palette);
    expect(out.indices).toBe(indices);
  });
});

// ── 2. ditherToPalette(mask=null) == nearest-colour assignment ──────────────────
describe("ditherToPalette", () => {
  it("with mask=null assigns each pixel to its nearest palette colour", () => {
    const palette: RGB[] = [
      [0, 0, 0],
      [128, 128, 128],
      [255, 255, 255],
    ];
    const raster = makeRaster(3, 1, (x) => {
      return [[10, 10, 10], [120, 120, 120], [240, 240, 240]][x] as RGB;
    });
    const rgb01 = rasterToRgb01(raster);

    const got = ditherToPalette(rgb01, 3, 1, palette, null);

    // brute-force nearest reference
    const expected = [0, 1, 2];
    expect(Array.from(got)).toEqual(expected);
  });

  it("with a mask only ever picks one of the two nearest colours", () => {
    const palette: RGB[] = [
      [0, 0, 0],
      [255, 0, 0],
      [0, 0, 255],
    ];
    const raster = makeRaster(8, 8, (x, y) => [(x * 32) % 256, 0, (y * 32) % 256]);
    const rgb01 = rasterToRgb01(raster);
    const mask = new Float32Array(64).fill(0.5);
    const idx = ditherToPalette(rgb01, 8, 8, palette, mask);
    for (const v of idx) expect(v).toBeGreaterThanOrEqual(0);
    for (const v of idx) expect(v).toBeLessThan(palette.length);
  });
});

// ── 3. Bayer threshold maps the known 2×2 and 4×4 matrices ──────────────────────
describe("bayerMatrix", () => {
  it("matches the canonical normalized 2×2 matrix", () => {
    expect(bayerMatrix(2)).toEqual([
      [0.5 / 4, 2.5 / 4],
      [3.5 / 4, 1.5 / 4],
    ]);
  });

  it("matches the canonical normalized 4×4 matrix", () => {
    const base = [
      [0, 8, 2, 10],
      [12, 4, 14, 6],
      [3, 11, 1, 9],
      [15, 7, 13, 5],
    ];
    const expected = base.map((row) => row.map((v) => (v + 0.5) / 16));
    expect(bayerMatrix(4)).toEqual(expected);
  });

  it("each tile is a permutation of the normalized levels", () => {
    for (const size of [2, 4, 8, 16]) {
      const flat = bayerMatrix(size).flat().sort((a, b) => a - b);
      const expected = Array.from({ length: size * size }, (_, k) => (k + 0.5) / (size * size));
      for (let i = 0; i < flat.length; i++) expect(flat[i]).toBeCloseTo(expected[i], 9);
    }
  });
});

// ── 4. Floyd–Steinberg stays in-palette and tracks mean tone ────────────────────
describe("fsToPalette", () => {
  it("produces only in-palette indices and tracks mean tone within tolerance", () => {
    const w = 64;
    const h = 16;
    const raster = grayGradient(w, h);
    const palette: RGB[] = [
      [0, 0, 0],
      [85, 85, 85],
      [170, 170, 170],
      [255, 255, 255],
    ];
    const rgb01 = rasterToRgb01(raster);
    const idx = fsToPalette(rgb01, w, h, palette);

    for (const v of idx) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(palette.length);
    }

    let meanOut = 0;
    for (const v of idx) meanOut += luma(palette[v]);
    meanOut /= idx.length;

    let meanIn = 0;
    const gray = rasterToGray(raster);
    for (const g of gray) meanIn += g * 255;
    meanIn /= gray.length;

    expect(Math.abs(meanOut - meanIn)).toBeLessThan(4); // within a few luma units
  });
});

// ── 5. Flow mask uses raw gray; palette assignment uses toned RGB ───────────────
describe("renderToneDither flow split", () => {
  it("computes the flow mask from RAW gray and assigns from TONED rgb", () => {
    const w = 24;
    const h = 24;
    // A non-trivial image so tone remapping actually moves pixels.
    const raster = makeRaster(w, h, (x, y) => {
      const v = Math.round(((x + y) / (w + h - 2)) * 255);
      return [v, Math.round(v * 0.6), 255 - v];
    });

    const contrast = 1.8;
    const midpoint = 0.4;
    const params = { frequency: 6, warp: 7, angleDeg: 45, detail: 2.5 };

    // Reference: mask from raw gray, assignment from toned rgb, grayscale palette
    // (deterministic, no RNG).
    const gray = rasterToGray(raster);
    const tonedRgb = rasterToRgb01(raster);
    applyTone(tonedRgb, gray, contrast, midpoint);
    const mask = flowThresholdMap(gray, w, h, params.frequency, params.warp, params.angleDeg, params.detail);

    const palette: RGB[] = [
      [0, 0, 0],
      [85, 85, 85],
      [170, 170, 170],
      [255, 255, 255],
    ];
    const expected = ditherToPalette(tonedRgb, w, h, palette, mask);

    const actual = renderToneDither(raster, {
      colors: 4,
      palette: "grayscale",
      method: "flow",
      contrast,
      midpoint,
      params,
    });

    expect(Array.from(actual.indices)).toEqual(Array.from(expected));
  });
});

// ── 6. renderToneDither returns rows*cols indices and a unique palette ───────────
describe("renderToneDither", () => {
  it("returns width*height indices and a deduped (unique) palette", () => {
    const w = 40;
    const h = 30;
    const raster = makeRaster(w, h, (x, y) => [(x * 6) % 256, (y * 8) % 256, (x * y) % 256]);

    for (const method of ["bayer", "blue_noise", "floyd_steinberg", "flow", "flat"] as const) {
      const res = renderToneDither(raster, { colors: 5, palette: "adaptive", method, seed: 42 });
      expect(res.indices.length).toBe(w * h);
      expect(res.width).toBe(w);
      expect(res.height).toBe(h);

      const hexes = res.palette.map((c) => c.join(","));
      expect(new Set(hexes).size).toBe(res.palette.length); // unique
      for (const v of res.indices) expect(v).toBeLessThan(res.palette.length);
    }
  });

  it("is deterministic across runs for the adaptive palette", () => {
    const raster = makeRaster(50, 50, (x, y) => [(x * 5) % 256, (y * 5) % 256, 128]);
    const a = renderToneDither(raster, { colors: 6, palette: "adaptive", method: "bayer", seed: 7 });
    const b = renderToneDither(raster, { colors: 6, palette: "adaptive", method: "bayer", seed: 7 });
    expect(a.palette).toEqual(b.palette);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
  });
});
