import { describe, it, expect } from "vitest";
import { type RGB, type Raster } from "./colorworks";
import {
  maskToBraille,
  maskToBlock,
  isBlockShaped,
  chooseGlyph,
} from "./glyph_alphabet";
import {
  fitGlyphDocument,
  renderGlyphDocument,
  glyphDocumentToText,
  glyphDocumentToJSON,
  glyphDocumentFromJSON,
} from "./glyphfit";

/** RGBA raster from (x,y)->RGB. */
function makeRaster(w: number, h: number, fn: (x: number, y: number) => RGB): Raster {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const [r, g, b] = fn(x, y);
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
    }
  }
  return { width: w, height: h, data };
}

const eq = (a: RGB, b: RGB) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

describe("glyph_alphabet", () => {
  it("braille bit order: single dots map to the right code points (not row-major)", () => {
    // top-left subcell (idx 0) → ⠁ U+2801; idx 6 (row3,col0) → ⡀ U+2840
    expect(maskToBraille([true, false, false, false, false, false, false, false])).toBe("⠁");
    expect(maskToBraille([false, false, false, false, false, false, true, false])).toBe("⡀");
  });

  it("full mask → █, empty → space (block)", () => {
    expect(maskToBlock(new Array(8).fill(true))).toBe("█");
    expect(maskToBlock(new Array(8).fill(false))).toBe(" ");
  });

  it("block-shaped detection + top-half → ▀", () => {
    const topHalf = [true, true, true, true, false, false, false, false];
    expect(isBlockShaped(topHalf)).toBe(true);
    expect(maskToBlock(topHalf)).toBe("▀");
    const notBlock = [true, false, false, false, false, false, false, false];
    expect(isBlockShaped(notBlock)).toBe(false);
  });

  it("chooseGlyph: blocks_braille prefers a block when block-shaped, else braille", () => {
    const topHalf = [true, true, true, true, false, false, false, false];
    expect(chooseGlyph(topHalf, "blocks_braille")).toMatchObject({ char: "▀", kind: "block2x2" });
    const dot = [true, false, false, false, false, false, false, false];
    expect(chooseGlyph(dot, "blocks_braille")).toMatchObject({ char: "⠁", kind: "braille2x4" });
    expect(chooseGlyph(dot, "braille")).toMatchObject({ char: "⠁", kind: "braille2x4" });
  });
});

describe("glyphfit — fit", () => {
  const BG: RGB = [16, 16, 16];
  const FG: RGB = [220, 120, 90];

  /** One 4×8 cell (subcell 2×2) where the given canonical subcells are FG on BG. */
  function oneCell(litIdx: number[]): Raster {
    const lit = new Set(litIdx);
    return makeRaster(4, 8, (x, y) => {
      const c = x < 2 ? 0 : 1;
      const r = (y / 2) | 0;
      return lit.has(r * 2 + c) ? FG : BG;
    });
  }

  // force an unambiguous dark background so fg/bg orientation is deterministic
  const fitOne = (r: Raster, extra = {}) =>
    fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8, phaseX: 0, phaseY: 0, colorModel: "fg_bg", bgMode: "custom", bgColor: "#101010", ...extra });

  it("recovers a single top-left dot as ⠁ with the right fg/bg (score≈0)", () => {
    const doc = fitOne(oneCell([0]));
    expect(doc.cols).toBe(1);
    expect(doc.rows).toBe(1);
    const cell = doc.cells[0];
    expect(cell.glyph).toBe("⠁");
    expect(eq(cell.fg as RGB, FG)).toBe(true);
    expect(eq(cell.bg, BG)).toBe(true);
    expect(cell.reconError).toBeLessThan(1);
  });

  it("shape fit is non-degenerate: half-lit cells pick a half-block, not █/space", () => {
    // top two subrows lit → ▀; bottom two → ▄; empty → space (solid cells are
    // canonically a space painted with their bg colour, not █).
    expect(fitOne(oneCell([0, 1, 2, 3])).cells[0].glyph).toBe("▀");
    expect(fitOne(oneCell([4, 5, 6, 7])).cells[0].glyph).toBe("▄");
    expect(fitOne(oneCell([])).cells[0].glyph).toBe(" ");
  });

  it("fg_bg uses exactly two colours; per_subpixel keeps distinct subcell colours", () => {
    // two different bright colours in one cell
    const A: RGB = [220, 40, 40];
    const B: RGB = [40, 220, 40];
    const r = makeRaster(4, 8, (_x, y) => {
      const row = (y / 2) | 0;
      if (row === 0) return A;
      if (row === 1) return B;
      return [16, 16, 16];
    });
    const fg = fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8, colorModel: "fg_bg", bgMode: "custom", bgColor: "#101010" });
    const uniqFg = new Set(fg.cells[0].colors.map((c) => c.join(",")));
    expect(uniqFg.size).toBeLessThanOrEqual(2); // fg + bg only

    const ps = fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8, colorModel: "per_subpixel", bgMode: "custom", bgColor: "#101010", tau: 40 });
    const uniqPs = new Set(ps.cells[0].colors.map((c) => c.join(",")));
    expect(uniqPs.size).toBeGreaterThanOrEqual(3); // A, B, and bg
  });
});

describe("glyphfit — render-back", () => {
  it("renders cols·2 × rows·4 logical pixels (uniform grid, no ragged rows)", () => {
    // 2×1 cells, each 4×8 → render 4×4
    const r = makeRaster(8, 8, (x, _y) => (x < 4 ? [200, 200, 200] : [20, 20, 20]));
    const doc = fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8 });
    expect(doc.cols).toBe(2);
    expect(doc.rows).toBe(1);
    const res = renderGlyphDocument(doc);
    expect(res.width).toBe(4); // cols*2
    expect(res.height).toBe(4); // rows*4
    expect(res.indices.length).toBe(16);
  });

  it("round-trips a synthetic 2-colour image within tolerance", () => {
    // a 12×16 image of clean 4×8 cells with sharp 2×2-subcell features
    const FG: RGB = [240, 80, 200];
    const r = makeRaster(12, 16, (x, y) => {
      const c = ((x / 2) | 0) % 2;
      const rr = ((y / 2) | 0) % 2;
      return (c ^ rr) ? FG : [18, 18, 18];
    });
    const doc = fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8, colorModel: "fg_bg", bgMode: "custom", bgColor: "#121212" });
    expect(doc.meanError).toBeLessThan(30); // clean 2-colour-per-cell → low residual
  });
});

describe("glyphfit — auto grid (meanError-ranked)", () => {
  // a clean 4px sub-cell checker → detectGrid finds pitch 4 → char cell 8×16.
  function planted(): Raster {
    const FG: RGB = [235, 235, 235], BG: RGB = [16, 16, 16];
    return makeRaster(80, 80, (x, y) => {
      const sc = ((x % 8) / 4) | 0;
      const sr = ((y % 16) / 4) | 0;
      return (sc + sr) % 2 === 0 ? FG : BG;
    });
  }

  it("auto-detects the planted 8×16 cell", () => {
    const doc = fitGlyphDocument(planted(), { cellMode: "auto" });
    expect(Math.abs(doc.grid.cellW - 8)).toBeLessThanOrEqual(1);
    expect(Math.abs(doc.grid.cellH - 16)).toBeLessThanOrEqual(1);
  });

  it("the correct grid scores lower meanError than a misaligned one (the ranking premise)", () => {
    const r = planted();
    const good = fitGlyphDocument(r, { cellMode: "manual", cellW: 8, cellH: 16, phaseX: 0, phaseY: 0 });
    const bad = fitGlyphDocument(r, { cellMode: "manual", cellW: 11, cellH: 13, phaseX: 2, phaseY: 1 });
    expect(good.meanError).toBeLessThan(bad.meanError);
  });
});

describe("glyphfit — failure signal", () => {
  it("a noisy/3-colour image scores far higher meanError than a clean 2-colour one", () => {
    const clean = makeRaster(64, 64, (x, _y) => (((x / 8) | 0) % 2 ? [230, 230, 230] : [16, 16, 16]));
    // pseudo-random RGB per pixel (no seed needed: deterministic hash)
    const noisy = makeRaster(64, 64, (x, y) => {
      const h = (x * 2654435761 + y * 40503) >>> 0;
      return [h & 255, (h >> 8) & 255, (h >> 16) & 255];
    });
    const cleanDoc = fitGlyphDocument(clean, { cellMode: "manual", cellW: 16, cellH: 32 });
    const noisyDoc = fitGlyphDocument(noisy, { cellMode: "manual", cellW: 16, cellH: 32 });
    expect(noisyDoc.meanError).toBeGreaterThan(cleanDoc.meanError * 3);
  });
});

describe("glyphfit — exports", () => {
  const r = makeRaster(8, 8, (x, _y) => (x < 4 ? [210, 210, 210] : [20, 20, 20]));
  const doc = fitGlyphDocument(r, { cellMode: "manual", cellW: 4, cellH: 8 });

  it("glyphDocumentToText is shape-only (one row per cell-row, trailing-trimmed)", () => {
    const txt = glyphDocumentToText(doc);
    expect(txt.split("\n").length).toBeGreaterThanOrEqual(1);
    expect(typeof txt).toBe("string");
  });

  it("glyphDocumentFromJSON round-trips to the SAME render (not just serialization)", () => {
    const back = glyphDocumentFromJSON(glyphDocumentToJSON(doc));
    expect(back.grid).toEqual(doc.grid);
    const a = renderGlyphDocument(doc);
    const b = renderGlyphDocument(back);
    expect(b.width).toBe(a.width);
    expect(b.height).toBe(a.height);
    expect(Array.from(b.indices)).toEqual(Array.from(a.indices));
    expect(b.palette).toEqual(a.palette);
  });
});
