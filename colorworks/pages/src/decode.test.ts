import { describe, it, expect } from "vitest";
import { classifyFile, decodeLabel, svgRasterSize } from "./decode";

describe("classifyFile", () => {
  it("routes by MIME type", () => {
    expect(classifyFile({ name: "a.png", type: "image/png" })).toBe("raster");
    expect(classifyFile({ name: "a", type: "image/svg+xml" })).toBe("svg");
    expect(classifyFile({ name: "a", type: "image/tiff" })).toBe("tiff");
    expect(classifyFile({ name: "a", type: "image/heic" })).toBe("heic");
    // HEIF is the container HEIC sits in — both take the HEIC decode path.
    expect(classifyFile({ name: "a", type: "image/heif" })).toBe("heic");
  });

  it("falls back to the extension when the MIME type is empty or wrong", () => {
    // HEIC files routinely arrive with an empty type from the OS.
    expect(classifyFile({ name: "IMG_1234.HEIC", type: "" })).toBe("heic");
    expect(classifyFile({ name: "logo.svg", type: "" })).toBe("svg");
    expect(classifyFile({ name: "scan.tif", type: "application/octet-stream" })).toBe("tiff");
    expect(classifyFile({ name: "scan.tiff", type: "" })).toBe("tiff");
  });

  it("treats AVIF and other unknown rasters as the native path", () => {
    expect(classifyFile({ name: "shot.avif", type: "image/avif" })).toBe("raster");
    expect(classifyFile({ name: "photo.jpg", type: "image/jpeg" })).toBe("raster");
  });
});

describe("decodeLabel", () => {
  it("names the slow formats and falls back to a generic label", () => {
    expect(decodeLabel("heic")).toMatch(/HEIC/);
    expect(decodeLabel("tiff")).toMatch(/TIFF/);
    expect(decodeLabel("svg")).toMatch(/SVG/);
    expect(decodeLabel("raster")).toBe("Decoding…");
  });
});

describe("svgRasterSize", () => {
  it("scales the longest side to the target, preserving viewBox aspect", () => {
    expect(svgRasterSize('<svg viewBox="0 0 200 100"></svg>', 1000)).toEqual({ w: 1000, h: 500 });
    expect(svgRasterSize('<svg viewBox="0 0 50 200"></svg>', 1000)).toEqual({ w: 250, h: 1000 });
  });

  it("prefers viewBox over declared width/height", () => {
    expect(svgRasterSize('<svg width="10" height="10" viewBox="0 0 4 2"></svg>', 800)).toEqual({ w: 800, h: 400 });
  });

  it("uses width/height (ignoring units) when there is no viewBox", () => {
    expect(svgRasterSize('<svg width="300px" height="150pt"></svg>', 600)).toEqual({ w: 600, h: 300 });
  });

  it("falls back to a square when the SVG declares no usable size", () => {
    expect(svgRasterSize("<svg></svg>", 512)).toEqual({ w: 512, h: 512 });
    expect(svgRasterSize("not even svg", 512)).toEqual({ w: 512, h: 512 });
  });
});
