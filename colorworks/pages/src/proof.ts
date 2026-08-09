import { srgbToOklab, type RGB, type Raster, type RenderResult } from "./colorworks";

export interface ProofMetrics {
  usedColors: number;
  perceptualError: number;
  lowFrequencyError: number;
}

/** A compact receipt, not a claim of print-device accuracy. */
export function computeProofMetrics(source: Raster, output: RenderResult): ProofMetrics | null {
  if (source.width !== output.width || source.height !== output.height) return null;
  const n = source.width * source.height;
  const used = new Set<number>();
  const stride = Math.max(1, Math.floor(n / 100_000));
  let delta = 0;
  let samples = 0;
  for (let i = 0; i < n; i++) {
    used.add(output.indices[i]);
    if (i % stride !== 0) continue;
    const src: RGB = [source.data[i * 4], source.data[i * 4 + 1], source.data[i * 4 + 2]];
    const dst = output.palette[output.indices[i]] ?? [0, 0, 0];
    const a = srgbToOklab(src);
    const b = srgbToOklab(dst);
    delta += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    samples++;
  }

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
    perceptualError: samples ? (delta / samples) * 100 : 0,
    lowFrequencyError: blocks ? (tone / blocks) * 100 : 0,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10 * 1024 ? 1 : 0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
