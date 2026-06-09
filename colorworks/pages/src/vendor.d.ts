// Minimal ambient types for UTIF.js (ships no declarations). Only the three
// entry points decode.ts uses are described; `width`/`height` land on an IFD
// after UTIF.decodeImage runs.
declare module "utif" {
  interface UTIFImage {
    width: number;
    height: number;
    [key: string]: unknown;
  }
  const UTIF: {
    decode(buffer: ArrayBuffer | Uint8Array): UTIFImage[];
    decodeImage(buffer: ArrayBuffer | Uint8Array, ifd: UTIFImage): void;
    toRGBA8(ifd: UTIFImage): Uint8Array;
  };
  export default UTIF;
}
