/* ============================================================================
   Colorworks — source decoding (browser-only).
   Turns an arbitrary image File into an ImageBitmap the studio can rasterise.

   Standard raster formats go straight through createImageBitmap; the ones
   browsers can't decode natively take dedicated paths:
     • SVG  — rasterised at a fixed resolution via an <img> + canvas.
     • HEIC/HEIF — decoded with heic2any (libheif compiled to WASM).
     • TIFF — decoded with UTIF.js (pure JS).
   The heavy HEIC/TIFF decoders are dynamically imported so they only ship to
   users who actually open one of those files — the base bundle stays lean.
   ========================================================================== */

export type SourceKind = "svg" | "heic" | "tiff" | "raster";

/** Longest side (px) to rasterise resolution-free formats (SVG) at. */
const VECTOR_RASTER = 1024;

/** Classify a file by MIME type, falling back to its extension — HEIC files
 *  frequently arrive with an empty `type`. Pure (unit-tested). */
export function classifyFile(file: { name: string; type: string }): SourceKind {
  const type = file.type.toLowerCase();
  const ext = (file.name.match(/\.([^.]+)$/)?.[1] ?? "").toLowerCase();
  if (type === "image/svg+xml" || ext === "svg") return "svg";
  if (type.includes("heic") || type.includes("heif") || ext === "heic" || ext === "heif") return "heic";
  if (type === "image/tiff" || ext === "tif" || ext === "tiff") return "tiff";
  return "raster";
}

/** Short label for the decode spinner — HEIC/TIFF can take a beat. */
export function decodeLabel(kind: SourceKind): string {
  if (kind === "heic") return "Decoding HEIC…";
  if (kind === "tiff") return "Decoding TIFF…";
  if (kind === "svg") return "Rasterising SVG…";
  return "Decoding…";
}

/** Pick a raster size for an SVG: preserve the viewBox/declared aspect ratio,
 *  scaling the longest side to `target` so the dither has pixels to chew on
 *  regardless of the vector's declared units. Pure (unit-tested) — reads only
 *  the root <svg> tag's attributes, so it works without a DOM. */
export function svgRasterSize(svgText: string, target = VECTOR_RASTER): { w: number; h: number } {
  const tag = svgText.match(/<svg\b[^>]*>/i)?.[0] ?? "";
  const attr = (name: string): string =>
    tag.match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']*)["']`, "i"))?.[1] ?? "";

  let aw = 0;
  let ah = 0;
  const vb = attr("viewBox").split(/[\s,]+/).map(Number).filter((n) => Number.isFinite(n));
  if (vb.length === 4 && vb[2] > 0 && vb[3] > 0) {
    aw = vb[2];
    ah = vb[3];
  } else {
    aw = parseFloat(attr("width")) || 0;
    ah = parseFloat(attr("height")) || 0;
  }
  if (!(aw > 0) || !(ah > 0)) return { w: target, h: target };
  const s = target / Math.max(aw, ah);
  return { w: Math.max(1, Math.round(aw * s)), h: Math.max(1, Math.round(ah * s)) };
}

// ── DOM-backed decode helpers ───────────────────────────────────────────────
/** Resolve an <img> for a URL (SVG path + the raster <img> fallback). */
function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image element failed to load"));
    img.src = url;
  });
}

/** Snapshot a drawable (img/canvas) into an ImageBitmap at the given size. */
async function bitmapFromDrawable(src: CanvasImageSource, w: number, h: number): Promise<ImageBitmap> {
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(src, 0, 0, w, h);
  return createImageBitmap(cv);
}

async function decodeSvg(file: File): Promise<ImageBitmap> {
  const { w, h } = svgRasterSize(await file.text());
  const url = URL.createObjectURL(file);
  try {
    return await bitmapFromDrawable(await loadImage(url), w, h);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function decodeHeic(file: File): Promise<ImageBitmap> {
  const { default: heic2any } = await import("heic2any");
  const out = await heic2any({ blob: file, toType: "image/png" });
  const png = Array.isArray(out) ? out[0] : out;
  return createImageBitmap(png, { imageOrientation: "from-image" });
}

async function decodeTiff(file: File): Promise<ImageBitmap> {
  const UTIF = (await import("utif")).default;
  const buf = await file.arrayBuffer();
  const ifds = UTIF.decode(buf);
  if (!ifds.length) throw new Error("TIFF has no images");
  const page = ifds[0];
  UTIF.decodeImage(buf, page);
  const rgba = new Uint8ClampedArray(UTIF.toRGBA8(page)); // copy → ImageData-ready
  const { width, height } = page;
  const cv = document.createElement("canvas");
  cv.width = width;
  cv.height = height;
  const ctx = cv.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  return createImageBitmap(cv);
}

/** <img>-based fallback for raster formats createImageBitmap can't build
 *  directly but the <img> tag can decode (e.g. AVIF on Safari). The tag
 *  applies EXIF orientation itself, matching the createImageBitmap path. */
async function decodeViaImage(file: File): Promise<ImageBitmap> {
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    if (!img.naturalWidth || !img.naturalHeight) throw new Error("decoded image has no size");
    return await bitmapFromDrawable(img, img.naturalWidth, img.naturalHeight);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Decode any supported source file into an ImageBitmap. Rejects if the format
 *  can't be decoded in this browser. */
export async function decodeToBitmap(file: File): Promise<ImageBitmap> {
  switch (classifyFile(file)) {
    case "svg":
      return decodeSvg(file);
    case "heic":
      return decodeHeic(file);
    case "tiff":
      return decodeTiff(file);
    default:
      // PNG/JPEG/WebP/GIF/BMP/AVIF/ICO — native decode, with an <img> fallback
      // for formats the bitmap factory rejects but the browser can still paint.
      try {
        return await createImageBitmap(file, { imageOrientation: "from-image" });
      } catch {
        return decodeViaImage(file);
      }
  }
}
