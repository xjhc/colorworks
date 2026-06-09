import { describe, it, expect } from "vitest";
import {
  type RGB,
  type Raster,
  bayerMatrix,
  ditherToPalette,
  fsToPalette,
  errorDiffuseToPalette,
  DIFFUSION_KERNELS,
  yliluomaToPalette,
  flowThresholdMap,
  dedupePalette,
  renderToneDither,
  rasterToRgb01,
  rasterToGray,
  applyTone,
  srgbToOklab,
  oklabToRgb,
  kmeansPalette,
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

// ── 4a. OKLab colour space: roundtrip + perceptual palette extraction ───────────
describe("OKLab colour", () => {
  it("round-trips sRGB → OKLab → sRGB within rounding", () => {
    const samples: RGB[] = [
      [0, 0, 0],
      [255, 255, 255],
      [128, 64, 200],
      [220, 30, 30],
      [30, 180, 180],
      [17, 99, 240],
    ];
    for (const c of samples) {
      const [L, a, b] = srgbToOklab(c);
      const back = oklabToRgb(L, a, b);
      for (let k = 0; k < 3; k++) expect(Math.abs(back[k] - c[k])).toBeLessThanOrEqual(2);
    }
    // OKLab L is monotonic in luma: black darkest, white lightest.
    expect(srgbToOklab([0, 0, 0])[0]).toBeLessThan(srgbToOklab([128, 128, 128])[0]);
    expect(srgbToOklab([128, 128, 128])[0]).toBeLessThan(srgbToOklab([255, 255, 255])[0]);
  });

  it("extracts a perceptually distinct palette from a bicolour image", () => {
    // Half saturated red, half saturated teal — two clearly distinct hues.
    const w = 40;
    const h = 40;
    const raster = makeRaster(w, h, (x) => (x < w / 2 ? [220, 30, 30] : [30, 180, 180]));
    const pal = kmeansPalette(raster, 2, 42);
    expect(pal.length).toBe(2);
    // The two swatches must be far apart perceptually (not two near-identical reds).
    const [l0, a0, b0] = srgbToOklab(pal[0]);
    const [l1, a1, b1] = srgbToOklab(pal[1]);
    const dist = Math.hypot(l0 - l1, a0 - a1, b0 - b1);
    expect(dist).toBeGreaterThan(0.1);
    // Each input colour should be close to one of the recovered swatches.
    const near = (c: RGB) => {
      const [L, a, b] = srgbToOklab(c);
      return Math.min(
        Math.hypot(L - l0, a - a0, b - b0),
        Math.hypot(L - l1, a - a1, b - b1),
      );
    };
    expect(near([220, 30, 30])).toBeLessThan(0.06);
    expect(near([30, 180, 180])).toBeLessThan(0.06);
  });

  it("kmeansPalette is deterministic and returns luma-sorted swatches", () => {
    const raster = makeRaster(50, 50, (x, y) => [(x * 5) % 256, (y * 5) % 256, 128]);
    const a = kmeansPalette(raster, 6, 7);
    const b = kmeansPalette(raster, 6, 7);
    expect(a).toEqual(b);
    for (let i = 1; i < a.length; i++) expect(luma(a[i])).toBeGreaterThanOrEqual(luma(a[i - 1]));
  });
});

// ── 4b. Error-diffusion kernel pack stays in-palette; conserving kernels track tone
describe("error-diffusion kernel pack", () => {
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

  for (const method of Object.keys(DIFFUSION_KERNELS)) {
    it(`${method} stays in-palette and uses multiple levels`, () => {
      const idx = errorDiffuseToPalette(rgb01, w, h, palette, DIFFUSION_KERNELS[method]);
      for (const v of idx) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(palette.length);
      }
      expect(new Set(idx).size).toBeGreaterThan(1); // not a degenerate flat fill
    });
  }

  it("error-conserving kernels track mean tone (Atkinson is allowed to drift)", () => {
    const gray = rasterToGray(raster);
    let meanIn = 0;
    for (const g of gray) meanIn += g * 255;
    meanIn /= gray.length;

    // Atkinson under-diffuses (6/8) by design, so it's excluded from the tone check.
    for (const m of ["floyd_steinberg", "jarvis", "stucki", "burkes", "sierra"]) {
      const idx = errorDiffuseToPalette(rgb01, w, h, palette, DIFFUSION_KERNELS[m]);
      let meanOut = 0;
      for (const v of idx) meanOut += luma(palette[v]);
      meanOut /= idx.length;
      expect(Math.abs(meanOut - meanIn)).toBeLessThan(8);
    }
  });

  it("fsToPalette equals the floyd_steinberg kernel via the generic engine", () => {
    const a = fsToPalette(rgb01, w, h, palette);
    const b = errorDiffuseToPalette(rgb01, w, h, palette, DIFFUSION_KERNELS.floyd_steinberg);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

// ── 4c. Yliluoma mixes a fixed palette to approximate out-of-palette colours ────
describe("yliluoma positional dithering", () => {
  // sRGB→linear at the same gamma the renderer uses.
  const toLinear = (c: number) => Math.pow(c / 255, 2.2);

  it("mixes BOTH colours of a 2-colour palette to hit a midtone", () => {
    const w = 16;
    const h = 16;
    const raster = makeRaster(w, h, () => [128, 128, 128]); // flat mid-gray
    const res = renderToneDither(raster, {
      colors: 2,
      palette: "grayscale",
      method: "yliluoma",
      seed: 1,
    });
    // grayscale 2-colour palette is [black, white] — mixing must use both.
    expect(new Set(res.indices).size).toBe(2);

    // Gamma-correct mixing matches the target in LINEAR light (the white
    // fraction ≈ target's linear value), not in gamma space.
    let meanLin = 0;
    for (const v of res.indices) meanLin += toLinear(luma(res.palette[v]));
    meanLin /= res.indices.length;
    expect(Math.abs(meanLin - toLinear(128))).toBeLessThan(0.06);
  });

  it("snaps to nearest when the colour is already in the palette", () => {
    const w = 8;
    const h = 8;
    const raster = makeRaster(w, h, () => [0, 0, 0]); // pure black, in-palette
    const palette: RGB[] = [
      [0, 0, 0],
      [255, 255, 255],
    ];
    const idx = yliluomaToPalette(rasterToRgb01(raster), w, h, palette, 8);
    for (const v of idx) expect(v).toBe(0); // all black, no spurious mixing
  });

  it("stays in-palette and is deterministic", () => {
    const w = 24;
    const h = 18;
    const raster = makeRaster(w, h, (x, y) => [(x * 9) % 256, (y * 11) % 256, (x * y) % 256]);
    const a = renderToneDither(raster, { colors: 4, palette: "adaptive", method: "yliluoma", seed: 5 });
    const b = renderToneDither(raster, { colors: 4, palette: "adaptive", method: "yliluoma", seed: 5 });
    for (const v of a.indices) expect(v).toBeLessThan(a.palette.length);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
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

    const methods = [
      "bayer", "blue_noise", "floyd_steinberg",
      "atkinson", "jarvis", "stucki", "burkes", "sierra",
      "yliluoma", "flow", "flat",
    ] as const;
    for (const method of methods) {
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
