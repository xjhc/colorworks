import { describe, it, expect } from "vitest";
import {
  type RGB,
  type Raster,
  type PinnedColor,
  kmeansPalette,
  buildTonePalette,
  parseColor,
  srgbToOklab,
  excludePinLabs,
  rgbNearAnyLab,
} from "./colorworks";
import { renderBlockMosaic } from "./blockmosaic";

// ── helpers ───────────────────────────────────────────────────────────────────
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

const eq = (a: RGB, b: RGB) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
const has = (pal: RGB[], c: RGB) => pal.some((p) => eq(p, c));
// OKLab a-axis: red/orange are +a, green/teal are −a — a luma-independent way to
// tell the two families apart no matter how dark/light each shade is.
const greenish = (pal: RGB[]) => pal.filter((c) => srgbToOklab(c)[1] < 0).length;

// A red majority (left 80%) with a teal minority (right 20%), each a vertical
// luma gradient so there are several distinguishable shades to allocate.
const redTeal = makeRaster(40, 40, (x, y) =>
  x < 32 ? [200 - y * 3, 24, 24] : [24, 200 - y * 3, 200 - y * 3],
);
const halves = makeRaster(40, 1, (x) => (x < 20 ? [220, 30, 30] : [30, 180, 180]));

describe("pinned adaptive palette", () => {
  it("no pins ≡ pins:[] ≡ pins:undefined (delegation is a no-op)", () => {
    const base = kmeansPalette(redTeal, 4);
    expect(kmeansPalette(redTeal, 4, 0, 16, [])).toEqual(base);
    expect(kmeansPalette(redTeal, 4, 0, 16, undefined)).toEqual(base);
  });

  it("lock forces the exact colour into the palette", () => {
    const yellow: RGB = [255, 224, 0];
    const pins: PinnedColor[] = [{ hex: "#ffe000", mode: "lock" }];
    const pal = buildTonePalette(redTeal, 4, "adaptive", "#161616", "#f4ebd9", 0, pins);
    expect(pal).toHaveLength(4);
    expect(has(pal, yellow)).toBe(true); // exact hue, not a round-tripped approximation
  });

  it("two locks coexist and both appear exactly", () => {
    const pins: PinnedColor[] = [
      { hex: "#ffe000", mode: "lock" },
      { hex: "#7b2ff7", mode: "lock" },
    ];
    const pal = buildTonePalette(redTeal, 5, "adaptive", "#161616", "#f4ebd9", 0, pins);
    expect(pal).toHaveLength(5);
    expect(has(pal, [255, 224, 0])).toBe(true);
    expect(has(pal, [123, 47, 247])).toBe(true);
  });

  it("fix-palette: locks == colors yields exactly the locked colours", () => {
    const pins: PinnedColor[] = [
      { hex: "#ff0000", mode: "lock" },
      { hex: "#00ff00", mode: "lock" },
      { hex: "#0000ff", mode: "lock" },
    ];
    const pal = buildTonePalette(redTeal, 3, "adaptive", "#161616", "#f4ebd9", 0, pins);
    expect(pal).toHaveLength(3);
    for (const hex of ["#ff0000", "#00ff00", "#0000ff"]) {
      expect(has(pal, parseColor(hex))).toBe(true);
    }
  });

  it("excess locks beyond `colors` are clamped, not duplicated", () => {
    const pins: PinnedColor[] = [
      { hex: "#ff0000", mode: "lock" },
      { hex: "#00ff00", mode: "lock" },
      { hex: "#0000ff", mode: "lock" },
    ];
    const pal = buildTonePalette(redTeal, 2, "adaptive", "#161616", "#f4ebd9", 0, pins);
    expect(pal).toHaveLength(2);
  });

  it("exclude keeps a hue out while its complement survives", () => {
    const pins: PinnedColor[] = [{ hex: "#dc1e1e", mode: "exclude" }]; // ≈ the red family
    const pal = buildTonePalette(halves, 4, "adaptive", "#161616", "#f4ebd9", 0, pins);
    const labs = excludePinLabs(pins);
    expect(pal.every((c) => !rgbNearAnyLab(c, labs))).toBe(true); // no red-family slot
    expect(greenish(pal)).toBeGreaterThan(0); // teal still represented
  });

  it("boost claims more palette slots for the boosted hue", () => {
    const base = greenish(buildTonePalette(redTeal, 5, "adaptive", "#161616", "#f4ebd9", 0));
    const boosted = greenish(
      buildTonePalette(redTeal, 5, "adaptive", "#161616", "#f4ebd9", 0, [
        { hex: "#1ec8c8", mode: "boost", weight: 16 },
      ]),
    );
    expect(boosted).toBeGreaterThanOrEqual(base);
    expect(boosted).toBeGreaterThanOrEqual(2); // a strong boost forces multiple teal shades
  });

  it("grayscale/duotone ignore pins (fixed ramps)", () => {
    const pins: PinnedColor[] = [{ hex: "#ffe000", mode: "lock" }];
    const gray = buildTonePalette(redTeal, 4, "grayscale", "#161616", "#f4ebd9", 0, pins);
    expect(gray.every((c) => c[0] === c[1] && c[1] === c[2])).toBe(true);
    expect(has(gray, [255, 224, 0])).toBe(false);
  });
});

describe("pins flow through the block mosaic", () => {
  it("fix-palette locks define the mosaic's whole vocabulary", () => {
    // colors == locks, solids only, no learned blocks → the candidate set is
    // exactly the two locked solids, so every output cell is one of them.
    const res = renderBlockMosaic(halves, {
      colors: 2,
      palette: "adaptive",
      library: "solids",
      learn: 0,
      pins: [
        { hex: "#dc1e1e", mode: "lock" }, // ≈ the red half
        { hex: "#1eb4b4", mode: "lock" }, // ≈ the teal half
      ],
    });
    expect(res.palette.every((c) => eq(c, [220, 30, 30]) || eq(c, [30, 180, 180]))).toBe(true);
    expect(has(res.palette, [220, 30, 30])).toBe(true);
    expect(has(res.palette, [30, 180, 180])).toBe(true);
  });

  it("an excluded hue never appears in the mosaic output (presets and learned)", () => {
    const pins: PinnedColor[] = [{ hex: "#dc1e1e", mode: "exclude" }];
    const res = renderBlockMosaic(redTeal, { colors: 4, palette: "adaptive", learn: 8, pins });
    const labs = excludePinLabs(pins);
    expect(res.palette.every((c) => !rgbNearAnyLab(c, labs))).toBe(true);
  });
});
