/**
 * Indexed-colour PNG encoder (colour type 3) for index-map renders.
 *
 * A dither is a handful of inks, so storing palette indices at 1/2/4 bits per
 * pixel is several times smaller than the 32-bit RGBA PNG a canvas encodes —
 * and exactly as lossless. Compression is the browser's own zlib
 * (CompressionStream "deflate"), so this module only lays out the bytes.
 */
import type { RGB } from "./colorworks";

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** length · type · data · CRC(type + data). */
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** The smallest PNG bit depth whose palette holds `n` colours. */
export function bitDepthFor(n: number): 1 | 2 | 4 | 8 {
  return n <= 2 ? 1 : n <= 4 ? 2 : n <= 16 ? 4 : 8;
}

/** Raw scanlines: filter byte 0 (None, the PNG spec's advice for palette
 *  images) then the packed indices, MSB first. `scale` repeats every pixel
 *  scale×scale — a nearest-neighbour enlargement. */
export function packScanlines(
  indices: Uint16Array,
  width: number,
  height: number,
  depth: 1 | 2 | 4 | 8,
  scale = 1,
): Uint8Array {
  const W = width * scale;
  const rowBytes = Math.ceil((W * depth) / 8);
  const stride = rowBytes + 1;
  const out = new Uint8Array(stride * height * scale);
  const perByte = 8 / depth;
  for (let y = 0; y < height; y++) {
    const o = y * scale * stride;
    const row = out.subarray(o + 1, o + stride);
    const src = y * width;
    for (let x = 0; x < W; x++) {
      const v = indices[src + ((x / scale) | 0)];
      if (depth === 8) row[x] = v;
      else row[(x / perByte) | 0] |= v << (8 - depth * ((x % perByte) + 1));
    }
    for (let r = 1; r < scale; r++) out.copyWithin(o + r * stride, o, o + stride);
  }
  return out;
}

async function deflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Whether this browser can encode indexed PNGs (CompressionStream support). */
export function canEncodeIndexedPng(): boolean {
  return typeof CompressionStream === "function";
}

/** Encode an index map + palette (≤ 256 colours) as an indexed PNG, enlarged
 *  `scale`× by pixel repetition. */
export async function encodeIndexedPng(
  indices: Uint16Array,
  width: number,
  height: number,
  palette: RGB[],
  scale = 1,
): Promise<Blob> {
  if (palette.length < 1 || palette.length > 256) throw new Error("indexed PNG needs 1–256 colours");
  const depth = bitDepthFor(palette.length);
  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width * scale);
  view.setUint32(4, height * scale);
  ihdr[8] = depth;
  ihdr[9] = 3; // colour type: palette
  // compression, filter, and interlace methods are all 0
  const plte = new Uint8Array(palette.length * 3);
  palette.forEach((c, i) => plte.set(c, i * 3));
  const idat = await deflate(packScanlines(indices, width, height, depth, scale));
  const parts = [SIGNATURE, chunk("IHDR", ihdr), chunk("PLTE", plte), chunk("IDAT", idat), chunk("IEND", new Uint8Array(0))];
  return new Blob(parts as Uint8Array<ArrayBuffer>[], { type: "image/png" });
}

/** True when a WebP file is lossless (has a VP8L chunk). Browsers vary in
 *  what canvas.toBlob("image/webp", 1) produces, so the export checks rather
 *  than assumes. */
export function isLosslessWebp(bytes: Uint8Array): boolean {
  const tag = (o: number) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  if (bytes.length < 16 || tag(0) !== "RIFF" || tag(8) !== "WEBP") return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let o = 12; o + 8 <= bytes.length; ) {
    const type = tag(o);
    if (type === "VP8L") return true;
    if (type === "VP8 ") return false;
    const size = view.getUint32(o + 4, true);
    o += 8 + size + (size & 1);
  }
  return false;
}
