import { linearToOklabInto, type Raster, type RenderResult } from "./colorworks";

export interface ProofMetrics {
  usedColors: number;
  /** Mean OKLab ΔE×100 once both images are blurred to viewing distance. */
  viewingError: number;
  /** Mean per-channel difference over 8px blocks, as a percentage. */
  lowFrequencyError: number;
}

/** Blur radius, in output pixels, at which neighbouring dots merge — the
 *  distance at which a dither is meant to be seen. Palette spread was tuned
 *  against this same measure. */
const VIEW_SIGMA = 1.5;
/** Sample points for the viewing-distance error; enough for a stable mean. */
const VIEW_SAMPLES = 20_000;

const SRGB_TO_LINEAR = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return t;
})();

/** Mirror an out-of-range coordinate back inside [0, n) without repeating the
 *  edge pixel, so a fine pattern keeps its parity at the border. */
function reflect(i: number, n: number): number {
  if (n === 1) return 0;
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

/** Mean OKLab ΔE×100 between source and render after both are blurred in
 *  linear light (how light mixes in the eye). A pixel-by-pixel comparison
 *  punishes every dither for its texture and ranks a flat poster first; this
 *  measures what the dots add up to. Evaluated at a grid of sample points. */
function viewingDistanceError(source: Raster, output: RenderResult): number {
  const { width: w, height: h } = source;
  const radius = Math.ceil(3 * VIEW_SIGMA);
  const kernel = new Float32Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    kernel[i + radius] = Math.exp(-(i * i) / (2 * VIEW_SIGMA * VIEW_SIGMA));
    ksum += kernel[i + radius];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const pal = new Float32Array(output.palette.length * 3);
  output.palette.forEach((c, j) => {
    pal[j * 3] = SRGB_TO_LINEAR[c[0]];
    pal[j * 3 + 1] = SRGB_TO_LINEAR[c[1]];
    pal[j * 3 + 2] = SRGB_TO_LINEAR[c[2]];
  });
  const src = source.data;
  const idx = output.indices;
  const a = new Float32Array(3);
  const b = new Float32Array(3);
  const step = Math.max(1, Math.round(Math.sqrt((w * h) / VIEW_SAMPLES)));
  let delta = 0;
  let samples = 0;
  for (let y = step >> 1; y < h; y += step) {
    for (let x = step >> 1; x < w; x += step) {
      let sr = 0, sg = 0, sb = 0, dr = 0, dg = 0, db = 0;
      for (let ky = -radius; ky <= radius; ky++) {
        const row = reflect(y + ky, h) * w;
        const wy = kernel[ky + radius];
        for (let kx = -radius; kx <= radius; kx++) {
          const i = row + reflect(x + kx, w);
          const wt = wy * kernel[kx + radius];
          sr += wt * SRGB_TO_LINEAR[src[i * 4]];
          sg += wt * SRGB_TO_LINEAR[src[i * 4 + 1]];
          sb += wt * SRGB_TO_LINEAR[src[i * 4 + 2]];
          const p = idx[i] * 3;
          dr += wt * pal[p];
          dg += wt * pal[p + 1];
          db += wt * pal[p + 2];
        }
      }
      linearToOklabInto(sr, sg, sb, a, 0);
      linearToOklabInto(dr, dg, db, b, 0);
      delta += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      samples++;
    }
  }
  return samples ? (delta / samples) * 100 : 0;
}

/** A compact receipt, not a claim of print-device accuracy. */
export function computeProofMetrics(source: Raster, output: RenderResult): ProofMetrics | null {
  if (source.width !== output.width || source.height !== output.height) return null;
  const n = source.width * source.height;
  const used = new Set<number>();
  for (let i = 0; i < n; i++) used.add(output.indices[i]);

  const block = 8;
  let tone = 0;
  let blocks = 0;
  for (let by = 0; by < source.height; by += block) {
    for (let bx = 0; bx < source.width; bx += block) {
      let sr = 0, sg = 0, sb = 0, dr = 0, dg = 0, db = 0, count = 0;
      for (let y = by; y < Math.min(source.height, by + block); y++) {
        for (let x = bx; x < Math.min(source.width, bx + block); x++) {
          const i = y * source.width + x;
          const c = output.palette[output.indices[i]] ?? [0, 0, 0];
          sr += source.data[i * 4]; sg += source.data[i * 4 + 1]; sb += source.data[i * 4 + 2];
          dr += c[0]; dg += c[1]; db += c[2];
          count++;
        }
      }
      tone += (Math.abs(sr - dr) + Math.abs(sg - dg) + Math.abs(sb - db)) / (count * 3 * 255);
      blocks++;
    }
  }

  return {
    usedColors: used.size,
    viewingError: viewingDistanceError(source, output),
    lowFrequencyError: blocks ? (tone / blocks) * 100 : 0,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
