/* ============================================================================
   Glyph alphabet — the *allowed* characters glyphfit may emit, and the mapping
   between a canonical 2x4 subcell MASK and its character.

   Canonical mask: boolean[8], index = row*2 + col (row-major, 2 cols × 4 rows):

        idx 0 1        col 0  col 1
        idx 2 3   →    row 0:  0   1
        idx 4 5        row 1:  2   3
        idx 6 7        row 2:  4   5
                       row 3:  6   7

   Braille (2x4) can encode ANY 8-bit pattern exactly; block glyphs (2x2,
   promoted to 2x4 by pairing rows) cover the 16 quadrant patterns. See
   GLYPHFIT_PLAN.md §2/§4.
   ========================================================================== */
import { brailleBit } from "./repixel";

export type GlyphKind = "block2x2" | "braille2x4";
export type Alphabet = "blocks" | "braille" | "blocks_braille";

/** Canonical 2x4 mask → Unicode braille char (U+2800..U+28FF). Bit order is the
 *  braille dot layout (via `brailleBit`), NOT row-major — do not shortcut it. */
export function maskToBraille(mask: boolean[]): string {
  let bits = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 2; c++) {
      if (mask[r * 2 + c]) bits |= 1 << brailleBit(c, r);
    }
  }
  return String.fromCharCode(0x2800 + bits);
}

/** 2x2 quadrant (tl,tr,bl,br) → block char. All 16 combinations are expressible. */
const QUAD: Record<string, string> = {
  "0000": " ", "1000": "▘", "0100": "▝", "1100": "▀",
  "0010": "▖", "1010": "▌", "0110": "▞", "1110": "▛",
  "0001": "▗", "1001": "▚", "0101": "▐", "1101": "▜",
  "0011": "▄", "1011": "▙", "0111": "▟", "1111": "█",
};

const bit = (v: boolean): string => (v ? "1" : "0");

/** A 2x4 mask is "block-shaped" when its row pairs match (rows 0≡1, 2≡3) — i.e. it
 *  is the vertical promotion of a 2x2 quadrant and can be drawn as a block char. */
export function isBlockShaped(m: boolean[]): boolean {
  return m[0] === m[2] && m[1] === m[3] && m[4] === m[6] && m[5] === m[7];
}

/** Block char for a (block-shaped) 2x4 mask — reads the top (row 0) and bottom
 *  (row 2) quadrants. */
export function maskToBlock(m: boolean[]): string {
  return QUAD[bit(m[0]) + bit(m[1]) + bit(m[4]) + bit(m[5])];
}

/** Snap any 2x4 mask to the nearest block-shaped mask: a quadrant is lit if either
 *  sub-row of its pair is lit (favours presence). Used only for the blocks-only
 *  alphabet, which is inherently lossy vs braille. */
export function snapToBlock(m: boolean[]): boolean[] {
  const tl = m[0] || m[2];
  const tr = m[1] || m[3];
  const bl = m[4] || m[6];
  const br = m[5] || m[7];
  return [tl, tr, tl, tr, bl, br, bl, br];
}

export interface ChosenGlyph {
  char: string;
  kind: GlyphKind;
  /** The mask actually represented (== input for braille; snapped for blocks). */
  mask: boolean[];
}

/** Pick the glyph for a binary subcell mask under the chosen alphabet:
 *  - "braille": exact (any pattern, 256 chars).
 *  - "blocks": snap to the nearest block-shaped mask, emit a block char.
 *  - "blocks_braille": a block char when the mask is already block-shaped (the
 *    cleaner terminal char), otherwise braille. */
export function chooseGlyph(mask: boolean[], alphabet: Alphabet): ChosenGlyph {
  if (alphabet === "braille") {
    return { char: maskToBraille(mask), kind: "braille2x4", mask };
  }
  if (alphabet === "blocks") {
    const sm = snapToBlock(mask);
    return { char: maskToBlock(sm), kind: "block2x2", mask: sm };
  }
  if (isBlockShaped(mask)) return { char: maskToBlock(mask), kind: "block2x2", mask };
  return { char: maskToBraille(mask), kind: "braille2x4", mask };
}
