import { describe, it, expect } from "vitest";
import { type RGB, type Raster } from "./colorworks";
import {
  renderBlockMosaic,
  buildPresetLibrary,
  chooseCandidate,
  type BlockMosaicOptions,
} from "./blockmosaic";

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

const lumaOf = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
const meanLuma = (res: { indices: Uint16Array; palette: RGB[] }): number => {
  let s = 0;
  for (let p = 0; p < res.indices.length; p++) s += lumaOf(res.palette[res.indices[p]]);
  return s / res.indices.length;
};

// ── 1. Solid image → uniform mosaic ─────────────────────────────────────────────
describe("renderBlockMosaic — solids", () => {
  it("renders a solid image as a single colour", () => {
    const r = makeRaster(48, 48, () => [200, 60, 60]);
    const res = renderBlockMosaic(r, {
      block: 2,
      cell: 6,
      library: "solids",
      learn: 0,
      palette: "adaptive",
      colors: 4,
    });
    expect(res.width).toBe(48); // cols=round(48/12)=4 → 4*2*6
    expect(res.palette.length).toBe(1);
    expect(Array.from(res.indices).every((v) => v === 0)).toBe(true);
  });
});

// ── 2. Checkerboard target → the matching checker block is chosen ───────────────
describe("renderBlockMosaic — checker match", () => {
  it("reproduces a cell-scale checkerboard from the checker preset", () => {
    const A: RGB = [20, 20, 20];
    const B: RGB = [230, 230, 230];
    const r = makeRaster(16, 16, (x, y) => ((x + y) & 1 ? B : A));
    const res = renderBlockMosaic(r, {
      block: 2,
      cell: 1, // one source px per cell → exact
      library: "checker",
      learn: 0,
      palette: "adaptive",
      colors: 2,
    });
    expect(res.width).toBe(16);
    expect(res.palette.length).toBe(2);
    // palette[0] is the darker swatch (luma-sorted); pixel (0,0) is the dark colour,
    // so the index map equals the checkerboard parity exactly.
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        expect(res.indices[y * 16 + x]).toBe((x + y) & 1);
      }
    }
  });
});

// ── 3. Learned-only reproduces the dominant colours ─────────────────────────────
describe("renderBlockMosaic — learned blocks", () => {
  it("recovers a 4-region image with library:none + learn:4", () => {
    const quad: RGB[] = [
      [200, 50, 50],
      [50, 180, 70],
      [60, 90, 200],
      [220, 200, 40],
    ];
    const idx = (x: number, y: number) => (x < 24 ? 0 : 1) + (y < 24 ? 0 : 2);
    const r = makeRaster(48, 48, (x, y) => quad[idx(x, y)]);
    const res = renderBlockMosaic(r, {
      block: 2,
      cell: 6,
      library: "none",
      learn: 4,
      palette: "adaptive",
      colors: 4,
    });
    expect(res.width).toBe(48);
    let maxErr = 0;
    for (let y = 0; y < 48; y++) {
      for (let x = 0; x < 48; x++) {
        const c = res.palette[res.indices[y * 48 + x]];
        const e = quad[idx(x, y)];
        maxErr = Math.max(
          maxErr,
          Math.abs(c[0] - e[0]),
          Math.abs(c[1] - e[1]),
          Math.abs(c[2] - e[2]),
        );
      }
    }
    expect(maxErr).toBeLessThan(10); // only OKLab round-trip drift
  });
});

// ── 4. Diffuse conserves regional tone better than match ────────────────────────
describe("renderBlockMosaic — tone conservation", () => {
  it("diffuse tracks the mean tone closer than match", () => {
    const ramp = (x: number) => Math.round((x / 63) * 180);
    const r = makeRaster(64, 64, (x) => [ramp(x), ramp(x), ramp(x)]);
    const base: BlockMosaicOptions = {
      block: 2,
      cell: 2,
      palette: "grayscale",
      colors: 2,
      library: "solids",
      learn: 0,
    };
    const m = renderBlockMosaic(r, { ...base, method: "match" });
    const d = renderBlockMosaic(r, { ...base, method: "diffuse" });

    let target = 0;
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) target += ramp(x);
    target /= 64 * 64;

    expect(Math.abs(meanLuma(d) - target)).toBeLessThan(Math.abs(meanLuma(m) - target));
  });
});

// ── 5. library_bias prefers presets only when the gap is small (unit) ───────────
describe("chooseCandidate — library bias", () => {
  // One-cell slot at L=0.5. Preset at L=0 (dist²=0.25); learned at L=0.6 (dist²=0.01)
  // — the learned block fits better, so it wins unless the preset is discounted hard.
  const slot = Float32Array.from([0.5, 0, 0]);
  const preset = Float32Array.from([0.0, 0, 0]);
  const learned = Float32Array.from([0.6, 0, 0]);
  const cands = [preset, learned];
  const isLib = [true, false];

  it("neutral (libFactor=1) picks the better-fitting learned block", () => {
    expect(chooseCandidate(slot, cands, isLib, 1)).toBe(1);
  });
  it("a gentle discount is not enough to flip", () => {
    expect(chooseCandidate(slot, cands, isLib, 0.5)).toBe(1); // 0.25*0.5=0.125 > 0.01
  });
  it("a strong discount flips to the preset", () => {
    expect(chooseCandidate(slot, cands, isLib, 0.03)).toBe(0); // 0.25*0.03=0.0075 < 0.01
  });
});

// ── 6. Deterministic ────────────────────────────────────────────────────────────
describe("renderBlockMosaic — determinism", () => {
  it("produces identical output for identical input", () => {
    const r = makeRaster(40, 40, (x, y) => [(x * 6) & 255, (y * 9) & 255, ((x + y) * 3) & 255]);
    const opts: BlockMosaicOptions = {
      block: 2,
      cell: 4,
      library: "auto",
      learn: 6,
      palette: "adaptive",
      colors: 5,
      method: "diffuse",
    };
    const a = renderBlockMosaic(r, opts);
    const b = renderBlockMosaic(r, opts);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
    expect(a.palette).toEqual(b.palette);
  });
});

// ── 7. Output dimensions follow the slot lattice ────────────────────────────────
describe("renderBlockMosaic — output size", () => {
  it("is cols·b·cell × rows·b·cell", () => {
    const r = makeRaster(50, 30, (x, y) => [(x * 5) & 255, (y * 8) & 255, 128]);
    const res = renderBlockMosaic(r, { block: 2, cell: 6 });
    const cols = Math.max(1, Math.round(50 / 12));
    const rows = Math.max(1, Math.round(30 / 12));
    expect(res.width).toBe(cols * 2 * 6);
    expect(res.height).toBe(rows * 2 * 6);
  });
});

// ── preset library shape ────────────────────────────────────────────────────────
describe("buildPresetLibrary", () => {
  const pal: RGB[] = [
    [0, 0, 0],
    [128, 128, 128],
    [255, 255, 255],
  ];
  it("solids = one block per colour", () => {
    expect(buildPresetLibrary("solids", pal, 2).length).toBe(3);
  });
  it("checker = solids + one checkerboard per colour pair", () => {
    expect(buildPresetLibrary("checker", pal, 2).length).toBe(3 + 3); // 3 solids + C(3,2)
  });
  it("auto = solids + checkers + diagonals", () => {
    expect(buildPresetLibrary("auto", pal, 2).length).toBe(3 + 3 + 3);
  });
  it("none = empty", () => {
    expect(buildPresetLibrary("none", pal, 2).length).toBe(0);
  });
  it("a checker block alternates the two colours", () => {
    const block = buildPresetLibrary("checker", pal, 2)[3]; // first checker = pair (0,1)
    expect(block.cells[0]).toEqual(pal[0]); // (0,0)
    expect(block.cells[1]).toEqual(pal[1]); // (0,1)
    expect(block.cells[2]).toEqual(pal[1]); // (1,0)
    expect(block.cells[3]).toEqual(pal[0]); // (1,1)
  });
});
