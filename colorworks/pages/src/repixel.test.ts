import { describe, it, expect } from "vitest";
import { type RGB, type Raster } from "./colorworks";
import { type RenderResult } from "./colorworks";
import { detectGrid as detectGridRepixel, detectSubjectGrid, detectCandidates, renderRepixel, toGlyphText } from "./repixel";
import { detectGrid as detectGridWeighted } from "./depixelate";

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

const BG: RGB = [20, 20, 20];

function outColor(res: { indices: Uint16Array; palette: RGB[]; width: number }, x: number, y: number): RGB {
  return res.palette[res.indices[y * res.width + x]];
}
const eq = (a: RGB, b: RGB): boolean => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
function litCount(res: { indices: Uint16Array; palette: RGB[]; width: number; height: number }, bg: RGB): number {
  let n = 0;
  for (let i = 0; i < res.indices.length; i++) if (!eq(res.palette[res.indices[i]], bg)) n++;
  return n;
}

describe("repixel — grid detection", () => {
  // The decisive test: a MIXED image where colourful blocks (high saturation,
  // coarse 14px pitch) coexist with a grey braille dot field (low saturation,
  // fine 7px pitch). depixelate's saturation-weighted detector suppresses the
  // grey braille and locks onto the 14px blocks; repixel's luminance-only
  // detector must lock onto the fine 7px lattice instead.
  function mixed(): Raster {
    const grey: RGB = [190, 190, 190];
    const col0: RGB = [210, 90, 60];
    const col1: RGB = [60, 120, 200];
    return makeRaster(112, 112, (x, y) => {
      if (x < 56 && y < 56) {
        // 14px solid colour squares → >2000 saturated px → weighting turns ON
        return (((x / 14) | 0) + ((y / 14) | 0)) % 2 === 0 ? col0 : col1;
      }
      // grey braille: a 2x2 dot at offset (1,1) of every 7px cell
      const ux = x % 7;
      const uy = y % 7;
      return (ux === 1 || ux === 2) && (uy === 1 || uy === 2) ? grey : BG;
    });
  }

  it("locks onto the fine 7px lattice (luminance profile), not the 14px blocks", () => {
    const r = mixed();
    const g = detectGridRepixel(r);
    expect(Math.round(g.pitchX)).toBe(7);
    expect(Math.round(g.pitchY)).toBe(7);
  });

  it("documents the bug: the saturation-weighted detector mis-locks onto ~14px", () => {
    // Proves the fixture actually triggers the weighting failure that repixel fixes.
    const r = mixed();
    const g = detectGridWeighted(r);
    expect(Math.round(g.pitchX)).not.toBe(7);
    expect(Math.round(g.pitchX)).toBeGreaterThanOrEqual(12);
  });

  it("the subject detector finds the coarse colour-sprite pitch (~14px), not the fine 7px", () => {
    const r = mixed();
    const g = detectSubjectGrid(r);
    expect(Math.round(g.pitchX)).toBeGreaterThanOrEqual(12);
    expect(Math.round(g.pitchX)).toBeLessThanOrEqual(16);
  });

  it("detectCandidates reports both scales (fine ~7 vs subject ~14)", () => {
    const c = detectCandidates(mixed());
    expect(Math.round(c.fine)).toBe(7);
    expect(c.subject).toBeGreaterThan(c.fine + 3); // the two scales are distinct
  });

  it("target:subject recovers at the coarser pitch (fewer cells than fine)", () => {
    const r = mixed();
    const fine = renderRepixel(r, { target: "fine" });
    const subject = renderRepixel(r, { target: "subject" });
    expect(subject.width).toBeLessThan(fine.width); // coarser pitch → fewer columns
  });
});

describe("repixel — recovery", () => {
  // A 6x6 mosaic of distinct, high-contrast cells (no two neighbours equal, none
  // == BG), so every cell boundary is an edge and the grid locks cleanly. Each
  // cell upscaled by `factor` px — i.e. drawn as a solid "block" glyph.
  const PALETTE: RGB[] = [
    [230, 230, 230],
    [210, 90, 60],
    [60, 120, 200],
    [240, 200, 40],
    [40, 170, 110],
    [200, 60, 180],
  ];
  function mosaicCells(): RGB[][] {
    const cells: RGB[][] = [];
    for (let y = 0; y < 6; y++) {
      cells.push([]);
      for (let x = 0; x < 6; x++) cells[y].push(PALETTE[(x * 2 + y * 3) % PALETTE.length]);
    }
    return cells;
  }
  function upscale(cells: RGB[][], factor: number): Raster {
    return makeRaster(cells[0].length * factor, cells.length * factor, (x, y) => cells[(y / factor) | 0][(x / factor) | 0]);
  }

  it("recovers each glyph cell as exactly one pixel of the right colour (position-preserving)", () => {
    const cells = mosaicCells();
    // Black background is distinct from every (bright) cell colour, so no cell is
    // ever read as background — each output pixel is its source cell's colour,
    // not a dithered/merged tile (which is the whole point vs depixelate).
    const res = renderRepixel(upscale(cells, 7), { pitch: 7, shade: false, bgMode: "custom", bgColor: "#000000" });
    expect(res.width).toBe(6);
    expect(res.height).toBe(6);
    for (let y = 0; y < 6; y++) {
      for (let x = 0; x < 6; x++) expect(outColor(res, x, y)).toEqual(cells[y][x]);
    }
  });

  it("shade recovers size-modulated halftone tone (dot size → brightness)", () => {
    // cell0 fully white (coverage 1); cell1 half-white (coverage ~0.5) on black.
    const W: RGB = [255, 255, 255];
    const r = makeRaster(14, 7, (x, y) => (x < 7 ? W : x >= 7 && y < 3 ? W : BG));
    const lumOf = (res: { indices: Uint16Array; palette: RGB[]; width: number }, cx: number) => {
      const c = outColor(res, cx, 0);
      return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    };
    const shaded = renderRepixel(r, { pitch: 7, shade: true, bgMode: "custom", bgColor: "#000000" });
    expect(lumOf(shaded, 0)).toBeGreaterThan(lumOf(shaded, 1)); // solid brighter than half
    expect(lumOf(shaded, 1)).toBeGreaterThan(30); // half-cell is a mid tone, not bg
    const flat = renderRepixel(r, { pitch: 7, shade: false, bgMode: "custom", bgColor: "#000000" });
    expect(Math.abs(lumOf(flat, 0) - lumOf(flat, 1))).toBeLessThan(20); // both → solid ink
  });

  it("min_lit rejects sparse speckle but keeps real dots", () => {
    // a uniform braille-like field: a 2x2 (=4px) light dot in every 7px cell
    const fg: RGB = [230, 230, 230];
    const field = makeRaster(70, 70, (x, y) => (x % 7 < 2 && y % 7 < 2 ? fg : BG));
    const keep = renderRepixel(field, { pitch: 7, minLit: 2, bgMode: "custom", bgColor: "#141414" });
    const drop = renderRepixel(field, { pitch: 7, minLit: 8, bgMode: "custom", bgColor: "#141414" });
    expect(litCount(keep, BG)).toBeGreaterThanOrEqual(50); // ~all 100 dots recovered as pixels
    expect(litCount(drop, BG)).toBe(0); // a 4px dot < 8 → every cell drops to background
  });

  it("bgMode custom drives the background fill (and differs from auto-detect)", () => {
    // a 2x2 patch of explicit background-coloured cells in the corner, so the
    // tested cell (5,5) is interior to the patch — its sample window only ever
    // bleeds neighbouring background cells, never a bright cell.
    const cells = mosaicCells();
    cells[5][5] = cells[5][4] = cells[4][5] = cells[4][4] = [20, 20, 20];
    const r = upscale(cells, 7);
    // custom bg "#1e1e1e" (=30,30,30) is within tau of the cell's 20,20,20, so the
    // cell reads as empty and is painted the CUSTOM colour…
    const custom = renderRepixel(r, { pitch: 7, shade: false, bgMode: "custom", bgColor: "#1e1e1e" });
    expect(outColor(custom, 5, 5)).toEqual([30, 30, 30]);
    // …whereas auto-detect's background is the modal mosaic colour, so the cell's
    // own 20,20,20 survives — proving the knob actually changes the fill.
    const auto = renderRepixel(r, { pitch: 7, shade: false, bgMode: "auto" });
    expect(outColor(auto, 5, 5)).toEqual([20, 20, 20]);
    expect(outColor(custom, 0, 0)).toEqual(cells[0][0]); // other cells unaffected
  });

  it("adaptive palette is built from the foreground, recovering the source inks", () => {
    // 6 distinct bright cells; with 6 fg clusters (colors=7 → colors-1) each ink
    // gets its own cluster, so adaptive recovers them rather than spending slots
    // on the (here, black) background.
    const cells = mosaicCells();
    const res = renderRepixel(upscale(cells, 7), { pitch: 7, shade: false, palette: "adaptive", colors: 7, bgMode: "custom", bgColor: "#000000" });
    expect(res.palette.length).toBeLessThanOrEqual(7);
    const near = (a: RGB, b: RGB) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));
    for (let y = 0; y < 6; y++) {
      for (let x = 0; x < 6; x++) expect(near(outColor(res, x, y), cells[y][x])).toBeLessThanOrEqual(2);
    }
  });

  it("quantises to a limited palette when a palette mode is given", () => {
    const res = renderRepixel(upscale(mosaicCells(), 7), { pitch: 7, palette: "grayscale", colors: 3 });
    expect(res.palette.length).toBeLessThanOrEqual(3);
    for (const c of res.palette) expect(c[0]).toBe(c[1]); // gray: R==G==B
  });

  it("toGlyphText re-encodes the bitmap as braille + block glyphs", () => {
    const make = (w: number, h: number, lit: Array<[number, number]>): RenderResult => {
      const indices = new Uint16Array(w * h); // 0 = background
      for (const [x, y] of lit) indices[y * w + x] = 1;
      return { indices, palette: [[0, 0, 0], [255, 255, 255]], width: w, height: h };
    };
    // a fully-lit 2x4 cell (with a clear bg majority below) → the full block char
    const block: Array<[number, number]> = [];
    for (let y = 0; y < 4; y++) for (let x = 0; x < 2; x++) block.push([x, y]);
    expect(toGlyphText(make(2, 12, block)).startsWith("█")).toBe(true);
    // a single lit dot at the top-left sub-cell → braille ⠁ (U+2801)
    expect(toGlyphText(make(2, 4, [[0, 0]]))).toBe("⠁\n");
    // empty bitmap → blank
    expect(toGlyphText(make(2, 4, [])).trim()).toBe("");
  });

  it("returns an indexed result the studio can paint", () => {
    const res = renderRepixel(upscale(mosaicCells(), 7), { pitch: 7 });
    expect(res.width).toBe(6);
    expect(res.height).toBe(6);
    expect(res.indices.length).toBe(36);
    expect(res.palette.length).toBeGreaterThan(0);
    for (const ix of res.indices) expect(ix).toBeLessThan(res.palette.length);
  });
});

describe("repixel — composite target", () => {
  // Braille grey dot field (7px fine lattice) everywhere, with a solid colour
  // sprite block overlaid in the centre. The composite target must keep the fine
  // background dots AND repaint the sprite region in its own colour — the two
  // scales a single-grid target cannot hold at once.
  function scene(): Raster {
    const grey: RGB = [190, 190, 190];
    const salmon: RGB = [210, 95, 65];
    return makeRaster(112, 112, (x, y) => {
      if (x >= 40 && x < 72 && y >= 40 && y < 72) return salmon; // colour sprite block
      const ux = x % 7;
      const uy = y % 7;
      return (ux === 1 || ux === 2) && (uy === 1 || uy === 2) ? grey : BG; // braille dots
    });
  }

  it("recovers the braille background AND overlays the colour sprite", () => {
    const res = renderRepixel(scene(), { target: "composite", shade: false, bgMode: "custom", bgColor: "#141414" });
    // Composite renders at SOURCE resolution (gapped dots, not one px per cell), so
    // the dots keep the inter-dot gap that reads as braille (not solid blocks).
    expect(res.width).toBe(112);
    expect(res.height).toBe(112);
    let grey = 0;
    let salmon = 0;
    let bgGap = 0;
    for (let i = 0; i < res.indices.length; i++) {
      const c = res.palette[res.indices[i]];
      if (Math.abs(c[0] - c[1]) < 25 && Math.abs(c[1] - c[2]) < 25 && c[0] > 140) grey++; // braille dot
      if (c[0] > 150 && c[0] > c[2] + 40) salmon++; // sprite ink
      if (c[0] < 40 && c[1] < 40 && c[2] < 40) bgGap++; // background (gaps between dots)
    }
    expect(grey).toBeGreaterThan(20); // background dots recovered outside the sprite
    expect(salmon).toBeGreaterThan(4); // sprite block overlaid in its own colour
    expect(bgGap).toBeGreaterThan(grey); // gaps dominate → dots are separated, not solid
  });
});
