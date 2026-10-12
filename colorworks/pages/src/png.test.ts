import { describe, expect, it } from "vitest";
import { bitDepthFor, crc32, encodeIndexedPng, isLosslessWebp, packScanlines } from "./png";
import type { RGB } from "./colorworks";

interface Chunk { type: string; data: Uint8Array }

/** Split a PNG into chunks, checking every CRC on the way. */
function readChunks(bytes: Uint8Array): Chunk[] {
  expect(Array.from(bytes.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: Chunk[] = [];
  for (let o = 8; o < bytes.length; ) {
    const len = view.getUint32(o);
    const type = String.fromCharCode(...bytes.subarray(o + 4, o + 8));
    const data = bytes.subarray(o + 8, o + 8 + len);
    expect(view.getUint32(o + 8 + len)).toBe(crc32(bytes.subarray(o + 4, o + 8 + len)));
    chunks.push({ type, data });
    o += 12 + len;
  }
  return chunks;
}

async function inflate(parts: Uint8Array[]): Promise<Uint8Array> {
  const stream = new Blob(parts as Uint8Array<ArrayBuffer>[]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Decode an indexed PNG back to its index map (filter 0 only, as we write). */
async function decodeIndexed(bytes: Uint8Array) {
  const chunks = readChunks(bytes);
  const ihdr = new DataView(chunks[0].data.buffer, chunks[0].data.byteOffset);
  const width = ihdr.getUint32(0);
  const height = ihdr.getUint32(4);
  const depth = chunks[0].data[8];
  const plte = chunks.find((c) => c.type === "PLTE")!.data;
  const raw = await inflate(chunks.filter((c) => c.type === "IDAT").map((c) => c.data));
  const rowBytes = Math.ceil((width * depth) / 8);
  const indices: number[] = [];
  for (let y = 0; y < height; y++) {
    expect(raw[y * (rowBytes + 1)]).toBe(0);
    const row = raw.subarray(y * (rowBytes + 1) + 1, (y + 1) * (rowBytes + 1));
    for (let x = 0; x < width; x++) {
      const bit = x * depth;
      indices.push((row[bit >> 3] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1));
    }
  }
  return { types: chunks.map((c) => c.type), width, height, depth, colorType: chunks[0].data[9], plte, indices };
}

describe("indexed PNG encoder", () => {
  it("picks the smallest bit depth for the palette", () => {
    expect([2, 3, 4, 5, 16, 17, 256].map(bitDepthFor)).toEqual([1, 2, 2, 4, 4, 8, 8]);
  });

  it("matches the standard CRC-32", () => {
    expect(crc32(new TextEncoder().encode("IEND"))).toBe(0xae426082);
  });

  it("round-trips an index map at every bit depth, enlarged by repetition", async () => {
    for (const n of [2, 4, 16, 40]) {
      const w = 7, h = 3, scale = 3;
      const indices = new Uint16Array(w * h).map((_, i) => (i * 5) % n);
      const palette: RGB[] = Array.from({ length: n }, (_, i) => [i, 255 - i, (i * 7) % 256]);
      const blob = await encodeIndexedPng(indices, w, h, palette, scale);
      expect(blob.type).toBe("image/png");
      const png = await decodeIndexed(new Uint8Array(await blob.arrayBuffer()));
      expect(png.types).toEqual(["IHDR", "PLTE", "IDAT", "IEND"]);
      expect([png.width, png.height, png.depth, png.colorType]).toEqual([w * scale, h * scale, bitDepthFor(n), 3]);
      expect(Array.from(png.plte)).toEqual(palette.flat());
      const expected: number[] = [];
      for (let y = 0; y < h * scale; y++) {
        for (let x = 0; x < w * scale; x++) expected.push(indices[((y / scale) | 0) * w + ((x / scale) | 0)]);
      }
      expect(png.indices).toEqual(expected);
    }
  });

  it("packs sub-byte pixels MSB first with a None filter byte per row", () => {
    const rows = packScanlines(new Uint16Array([1, 0, 1, 1, 0, 0, 0, 0, 1]), 9, 1, 1);
    expect(Array.from(rows)).toEqual([0, 0b10110000, 0b10000000]);
  });
});

describe("isLosslessWebp", () => {
  const riff = (...chunks: [string, number][]) => {
    const parts: number[] = [..."RIFF"].map((c) => c.charCodeAt(0)).concat([0, 0, 0, 0], [..."WEBP"].map((c) => c.charCodeAt(0)));
    for (const [type, size] of chunks) {
      parts.push(...[...type].map((c) => c.charCodeAt(0)), size & 255, size >> 8, 0, 0, ...new Array(size + (size & 1)).fill(0));
    }
    return new Uint8Array(parts);
  };
  it("tells lossless from lossy, including the extended (VP8X) layout", () => {
    expect(isLosslessWebp(riff(["VP8L", 5]))).toBe(true);
    expect(isLosslessWebp(riff(["VP8 ", 10]))).toBe(false);
    expect(isLosslessWebp(riff(["VP8X", 10], ["VP8L", 4]))).toBe(true);
    expect(isLosslessWebp(riff(["VP8X", 10], ["ALPH", 3], ["VP8 ", 4]))).toBe(false);
    expect(isLosslessWebp(new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]))).toBe(false);
  });
});
