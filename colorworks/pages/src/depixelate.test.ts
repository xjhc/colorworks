import { describe, it, expect } from "vitest";
import { type RGB, type Raster } from "./colorworks";
import { detectGrid, reduceToTiles, renderDepixelate, type Grid } from "./depixelate";

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

/** A 6x6 native sprite of distinct solid cells, upscaled by `factor` (nearest). */
function nativeCells(): RGB[][] {
  // deterministic, high-contrast neighbours so the grid is unambiguous
  const palette: RGB[] = [
    [20, 20, 20],
    [230, 230, 230],
    [210, 90, 60],
    [60, 120, 200],
    [240, 200, 40],
    [40, 170, 110],
  ];
  const cells: RGB[][] = [];
  for (let y = 0; y < 6; y++) {
    cells.push([]);
    for (let x = 0; x < 6; x++) cells[y].push(palette[(x * 2 + y * 3) % palette.length]);
  }
  return cells;
}

function upscaled(cells: RGB[][], factor: number): Raster {
  const h = cells.length;
  const w = cells[0].length;
  return makeRaster(w * factor, h * factor, (x, y) => cells[(y / factor) | 0][(x / factor) | 0]);
}

describe("depixelate", () => {
  it("detects the upscale factor as the grid pitch", () => {
    const r = upscaled(nativeCells(), 10);
    const grid = detectGrid(r);
    expect(Math.round(grid.pitchX)).toBe(10);
    expect(Math.round(grid.pitchY)).toBe(10);
  });

  it("recovers native solid cells as uniform tiles (block=2)", () => {
    const cells = nativeCells();
    const r = upscaled(cells, 10);
    const grid: Grid = { pitchX: 10, pitchY: 10, phaseX: 0, phaseY: 0, confidence: 1 };
    const out = reduceToTiles(r, grid, 2, { tau: 45 });
    expect(out.width).toBe(6 * 2);
    expect(out.height).toBe(6 * 2);
    // each solid cell -> uniform 2x2 tile; top-left subpixel recovers the native colour
    for (let cy = 0; cy < 6; cy++) {
      for (let cx = 0; cx < 6; cx++) {
        const i = ((cy * 2) * out.width + cx * 2) * 4;
        expect([out.data[i], out.data[i + 1], out.data[i + 2]]).toEqual(cells[cy][cx]);
      }
    }
  });

  it("scales output by the tile size", () => {
    const r = upscaled(nativeCells(), 10);
    const grid: Grid = { pitchX: 10, pitchY: 10, phaseX: 0, phaseY: 0, confidence: 1 };
    for (const block of [2, 3, 4]) {
      const out = reduceToTiles(r, grid, block, { tau: 45 });
      expect(out.width).toBe(6 * block);
      expect(out.height).toBe(6 * block);
    }
  });

  it("renders a two-colour cell as the o/x checkerboard at block=2", () => {
    // one cell spanning a black half and a white half
    const r = makeRaster(20, 10, (x) => (x < 10 ? [0, 0, 0] : [255, 255, 255]));
    const grid: Grid = { pitchX: 20, pitchY: 10, phaseX: 0, phaseY: 0, confidence: 1 };
    const out = reduceToTiles(r, grid, 2, { tau: 45 });
    expect(out.width).toBe(2);
    expect(out.height).toBe(2);
    const at = (x: number, y: number): RGB => {
      const i = (y * out.width + x) * 4;
      return [out.data[i], out.data[i + 1], out.data[i + 2]];
    };
    expect(at(0, 0)).toEqual(at(1, 1)); // diagonal equal
    expect(at(0, 1)).toEqual(at(1, 0)); // anti-diagonal equal
    expect(at(0, 0)).not.toEqual(at(0, 1)); // the two colours differ
  });

  it("keepMarks floors a sparse mark to 1 subpixel; off lets it stay solid", () => {
    // one cell that is mostly black with a small (1/9) white mark
    const r = makeRaster(30, 30, (x, y) => (x < 10 && y < 10 ? [255, 255, 255] : [0, 0, 0]));
    const grid: Grid = { pitchX: 30, pitchY: 30, phaseX: 0, phaseY: 0, confidence: 1 };
    const countWhite = (out: Raster): number => {
      let n = 0;
      for (let p = 0; p < out.width * out.height; p++) if (out.data[p * 4] > 200) n++;
      return n;
    };
    // frac ~= 1/9 -> round(0.11*4)=0 -> proportional drops it; keepMarks re-floors to 1
    expect(countWhite(reduceToTiles(r, grid, 2, { tau: 45, keepMarks: false }))).toBe(0);
    expect(countWhite(reduceToTiles(r, grid, 2, { tau: 45, keepMarks: true }))).toBeGreaterThan(0);
  });

  it("fillMult scales how fast coverage lights subpixels", () => {
    // a single cell that is ~30% white
    const r = makeRaster(20, 20, (_x, y) => (y < 6 ? [255, 255, 255] : [0, 0, 0]));
    const grid: Grid = { pitchX: 20, pitchY: 20, phaseX: 0, phaseY: 0, confidence: 1 };
    const whites = (m: number): number => {
      const out = reduceToTiles(r, grid, 2, { tau: 45, fillMult: m });
      let n = 0;
      for (let p = 0; p < out.width * out.height; p++) if (out.data[p * 4] > 200) n++;
      return n;
    };
    expect(whites(2)).toBeGreaterThan(whites(1)); // 2x fills more for the same coverage
  });

  it("quantises to a limited palette when a palette mode is given", () => {
    const r = upscaled(nativeCells(), 10);
    const res = renderDepixelate(r, { block: 2, pitch: 10, palette: "grayscale", colors: 3 });
    // grayscale-3 palette -> at most 3 distinct colours in the output
    expect(res.palette.length).toBeLessThanOrEqual(3);
    for (const c of res.palette) expect(c[0]).toBe(c[1]); // gray: R==G==B
  });

  it("renderDepixelate returns an indexed result the studio can paint", () => {
    const r = upscaled(nativeCells(), 10);
    const res = renderDepixelate(r, { block: 2, tau: 45, pitch: 10 });
    expect(res.width).toBe(12);
    expect(res.height).toBe(12);
    expect(res.indices.length).toBe(12 * 12);
    expect(res.palette.length).toBeGreaterThan(0);
    // every index points at a real palette entry
    for (const ix of res.indices) expect(ix).toBeLessThan(res.palette.length);
  });
});
