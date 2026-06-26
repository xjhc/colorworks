import { describe, it, expect } from "vitest";
import { type RGB, type Raster } from "./colorworks";
import {
  renderKnit,
  tidyStrays,
  longestRun,
  analyzeStranding,
  pruneSortYarns,
  ditherStitches,
  type KnitOptions,
} from "./knit";

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

// ── 1. Solid image → a single yarn, flat fill ───────────────────────────────────
describe("renderKnit — solids", () => {
  it("renders a solid image as one yarn with no grid", () => {
    const r = makeRaster(48, 48, () => [200, 60, 60]);
    const res = renderKnit(r, {
      stitches: 16,
      cell: 4,
      grid: "none",
      method: "flat",
      palette: "adaptive",
      colors: 4,
    });
    expect(res.stitchesWide).toBe(16);
    expect(res.width).toBe(16 * 4);
    expect(res.height).toBe(res.rowsTall * 4);
    expect(res.yarns.length).toBe(1);
    expect(res.palette.length).toBe(1); // grid:"none" → no extra grid colour
    expect(res.yarns[0].count).toBe(16 * res.rowsTall);
    expect(res.yarns[0].symbol).toBe("A");
    expect(res.symbols).toEqual(["A"]);
    expect(Array.from(res.indices).every((v) => v === 0)).toBe(true);
  });
});

// ── 2. Row count is gauge-corrected ─────────────────────────────────────────────
describe("renderKnit — gauge", () => {
  const r = makeRaster(80, 40, (x) => [(x * 3) & 255, 100, 150]);
  it("square gauge (1.0) follows the pixel aspect", () => {
    const res = renderKnit(r, { stitches: 40, cell: 2, grid: "none", rowGauge: 1 });
    expect(res.rowsTall).toBe(20); // round(40 * 40/80 * 1.0)
  });
  it("a taller row gauge inflates the row count", () => {
    const res = renderKnit(r, { stitches: 40, cell: 2, grid: "none", rowGauge: 2 });
    expect(res.rowsTall).toBe(40); // round(40 * 40/80 * 2.0)
    expect(res.height).toBe(40 * 2);
  });
  it("can use an exact row count instead of the gauge-derived count", () => {
    const res = renderKnit(r, { stitches: 40, rows: 17, cell: 2, grid: "none", rowGauge: 2 });
    expect(res.stitchesWide).toBe(40);
    expect(res.rowsTall).toBe(17);
    expect(res.height).toBe(17 * 2);
  });
});

// ── 3. Chart grid adds one palette entry and paints the boundary ────────────────
describe("renderKnit — chart grid", () => {
  const r = makeRaster(40, 40, (x, y) => [(x * 6) & 255, (y * 9) & 255, 128]);
  const base: KnitOptions = { stitches: 20, cell: 4, palette: "adaptive", colors: 4 };
  it("appends a grid colour and paints the top-left boundary", () => {
    const none = renderKnit(r, { ...base, grid: "none" });
    const dec = renderKnit(r, { ...base, grid: "decades" });
    expect(dec.palette.length).toBe(none.palette.length + 1);
    expect(dec.palette[dec.palette.length - 1]).toEqual([82, 82, 82]);
    // (0,0) lies on the first column boundary → it's a grid pixel.
    expect(dec.indices[0]).toBe(dec.palette.length - 1);
  });
});

// ── 4. tidyStrays merges isolated stitches but preserves texture ─────────────────
describe("tidyStrays", () => {
  it("merges a lone stitch in a solid field", () => {
    const cols = 5;
    const rows = 5;
    const idx = new Uint16Array(cols * rows); // all 0
    idx[2 * cols + 2] = 1; // a single stray in the middle
    const out = tidyStrays(idx, cols, rows);
    expect(out[2 * cols + 2]).toBe(0);
  });
  it("leaves a checkerboard intact (diagonals match)", () => {
    const cols = 5;
    const rows = 5;
    const chk = new Uint16Array(cols * rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) chk[y * cols + x] = (x + y) & 1;
    const out = tidyStrays(chk, cols, rows);
    expect(Array.from(out)).toEqual(Array.from(chk));
  });
  it("leaves a two-stitch domino intact (each has a same-colour neighbour)", () => {
    const cols = 5;
    const rows = 5;
    const idx = new Uint16Array(cols * rows);
    idx[2 * cols + 1] = 1;
    idx[2 * cols + 2] = 1; // horizontal domino
    const out = tidyStrays(idx, cols, rows);
    expect(out[2 * cols + 1]).toBe(1);
    expect(out[2 * cols + 2]).toBe(1);
  });
});

// ── 5. longestRun finds the longest single-colour row run ───────────────────────
describe("longestRun", () => {
  it("measures the longest horizontal run of one colour", () => {
    // row 0: 0 0 0 1   (run 3)
    // row 1: 1 0 1 0   (run 1)
    const idx = Uint16Array.from([0, 0, 0, 1, 1, 0, 1, 0]);
    expect(longestRun(idx, 4, 2)).toBe(3);
  });
});

// ── 5b. analyzeStranding measures real row-carried floats ──────────────────────
describe("analyzeStranding", () => {
  it("counts yarns per row and between-use float gaps", () => {
    const idx = Uint16Array.from([
      0, 0, 1, 0, 0,
      0, 1, 2, 1, 0,
    ]);
    const stats = analyzeStranding(idx, 5, 2, 3);
    expect(stats.longestSolidRun).toBe(2);
    expect(stats.maxFloat).toBe(3); // yarn 0 in row 1: x0 -> x4
    expect(stats.maxYarnsPerRow).toBe(3);
    expect(stats.rowsWithThreePlusYarns).toBe(1);
    expect(stats.rowsWithLongFloats).toBe(0);
  });

  it("flags rows with long carried gaps", () => {
    const idx = Uint16Array.from([0, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
    const stats = analyzeStranding(idx, 10, 1, 2);
    expect(stats.maxFloat).toBe(8);
    expect(stats.rowsWithLongFloats).toBe(1);
  });
});

// ── 6. pruneSortYarns drops unused yarns and sorts main-colour first ────────────
describe("pruneSortYarns", () => {
  it("removes unused colours and orders by descending count", () => {
    const palette: RGB[] = [
      [0, 0, 0], // index 0, used once
      [255, 0, 0], // index 1, unused
      [0, 255, 0], // index 2, used three times
    ];
    const idx = Uint16Array.from([0, 2, 2, 2]);
    const out = pruneSortYarns(idx, palette);
    expect(out.palette).toEqual([
      [0, 255, 0], // most used first
      [0, 0, 0],
    ]);
    expect(out.counts).toEqual([3, 1]);
    expect(Array.from(out.idx)).toEqual([1, 0, 0, 0]); // remapped to the new order
  });
});

// ── 7. ditherStitches "flat" assigns the nearest yarn (no diffusion) ────────────
describe("ditherStitches — flat", () => {
  it("maps each stitch to its nearest palette colour", () => {
    const palette: RGB[] = [
      [0, 0, 0],
      [255, 255, 255],
    ];
    // two stitches: near-black and near-white
    const rgb = Float32Array.from([0.05, 0.05, 0.05, 0.95, 0.95, 0.95]);
    const idx = ditherStitches(rgb, 2, 1, palette, "flat");
    expect(idx[0]).toBe(0);
    expect(idx[1]).toBe(1);
  });
});

// ── 8. Deterministic ────────────────────────────────────────────────────────────
describe("renderKnit — determinism", () => {
  it("produces identical output for identical input", () => {
    const r = makeRaster(60, 40, (x, y) => [(x * 6) & 255, (y * 9) & 255, ((x + y) * 3) & 255]);
    const opts: KnitOptions = {
      stitches: 30,
      cell: 4,
      palette: "adaptive",
      colors: 5,
      method: "floyd_steinberg",
      grid: "decades",
    };
    const a = renderKnit(r, opts);
    const b = renderKnit(r, opts);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
    expect(a.palette).toEqual(b.palette);
    expect(a.yarns.map((y) => y.count)).toEqual(b.yarns.map((y) => y.count));
  });
});

// ── 9. Symbol mode can hide symbol annotations ─────────────────────────────────
describe("renderKnit — symbols", () => {
  it("can omit symbols while keeping yarn stats", () => {
    const r = makeRaster(32, 32, (x) => (x < 16 ? [0, 0, 0] : [255, 255, 255]));
    const res = renderKnit(r, { stitches: 16, cell: 4, grid: "none", method: "flat", colors: 2, symbols: "none" });
    expect(res.symbols).toEqual([]);
    expect(res.yarns.map((y) => y.symbol)).toEqual(["A", "B"]);
  });
});
