/* ============================================================================
   Output-size geometry — shared, DOM-free so it can be unit-tested.

   The studio's "Output size" control (max W / max H / fit mode) defines the
   exact pixel dimensions of the preview and the exported PNG. `boxFit` resolves
   how a source maps into that box (mirroring the Python server's
   `resize_for_output` so both apps agree), and `conformIndexed` nearest-neighbour
   resamples an indexed render onto it — used for depixelate, whose recovered
   native grid is otherwise decoupled from the requested output size.
   ========================================================================== */
import type { RenderResult } from "./colorworks";

export type FitMode = "fit" | "cover" | "stretch";

export interface BoxFit {
  canvasW: number; // final raster width
  canvasH: number; // final raster height
  drawW: number; // scaled source width
  drawH: number; // scaled source height
  dx: number; // source x-offset within the canvas (≤ 0 → centre-crop, cover)
  dy: number;
}

/** Resolve how an (sw×sh) source maps into the output-size box. With
 *  `allowUpscale` the fit-mode "never upscale" cap is lifted — depixelate recovers
 *  a small native grid that we *do* want to scale up to the requested export size. */
export function boxFit(
  sw: number,
  sh: number,
  mw: number | null,
  mh: number | null,
  fit: FitMode,
  allowUpscale: boolean,
): BoxFit {
  if (mw === null && mh === null) {
    return { canvasW: sw, canvasH: sh, drawW: sw, drawH: sh, dx: 0, dy: 0 };
  }
  if (fit === "stretch") {
    const cw = mw ?? sw;
    const ch = mh ?? sh;
    return { canvasW: cw, canvasH: ch, drawW: cw, drawH: ch, dx: 0, dy: 0 };
  }
  let scale: number;
  if (mw !== null && mh !== null) {
    scale = fit === "fit" ? Math.min(mw / sw, mh / sh) : Math.max(mw / sw, mh / sh);
  } else if (mw !== null) {
    scale = mw / sw;
  } else {
    scale = (mh as number) / sh;
  }
  if (!allowUpscale && fit === "fit" && scale >= 1) scale = 1; // never upscale in fit mode
  const drawW = Math.max(1, Math.round(sw * scale));
  const drawH = Math.max(1, Math.round(sh * scale));
  if (fit === "cover" && mw !== null && mh !== null) {
    return { canvasW: mw, canvasH: mh, drawW, drawH, dx: Math.round((mw - drawW) / 2), dy: Math.round((mh - drawH) / 2) };
  }
  return { canvasW: drawW, canvasH: drawH, drawW, drawH, dx: 0, dy: 0 };
}

/** Nearest-neighbour resample an indexed render onto the output-size box so the
 *  exported PNG matches the requested size exactly. Returns the input unchanged
 *  when no size is set, or when it already matches. Nearest-neighbour keeps the
 *  crisp pixel blocks and the palette intact. */
export function conformIndexed(
  res: RenderResult,
  mw: number | null,
  mh: number | null,
  fit: FitMode,
): RenderResult {
  if (mw === null && mh === null) return res; // no size set → keep native grid
  const { canvasW, canvasH, drawW, drawH, dx, dy } = boxFit(res.width, res.height, mw, mh, fit, true);
  if (canvasW === res.width && canvasH === res.height && dx === 0 && dy === 0) return res;

  const sw = res.width;
  const sh = res.height;
  const src = res.indices;
  const out = new Uint16Array(canvasW * canvasH);
  for (let y = 0; y < canvasH; y++) {
    let sy = Math.floor(((y - dy) / drawH) * sh);
    sy = sy < 0 ? 0 : sy >= sh ? sh - 1 : sy;
    const srow = sy * sw;
    const orow = y * canvasW;
    for (let x = 0; x < canvasW; x++) {
      let sx = Math.floor(((x - dx) / drawW) * sw);
      sx = sx < 0 ? 0 : sx >= sw ? sw - 1 : sx;
      out[orow + x] = src[srow + sx];
    }
  }
  return { indices: out, palette: res.palette, width: canvasW, height: canvasH };
}
