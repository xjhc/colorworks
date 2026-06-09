#!/usr/bin/env python3
"""Unified inverse-renderer: screenshot -> colored half-block (▀) terminal art.

One character cell = 8px wide x 16px tall = two stacked 8x8 colour pixels.
The glyph is always U+2580 ▀ (upper half block); the TOP pixel is the
foreground colour, the BOTTOM pixel is the background colour. This represents
the grayscale dot-art background AND the colour pixel-art character in a single
block-glyph grid.
"""
import sys
import numpy as np
from PIL import Image

CW, CH = 8, 16            # char cell: 8 wide, 16 tall (two 8x8 halves)
SAT_THR = 0.35           # above -> coloured sprite (use smooth mean colour)
DOT_THR = 70             # grayscale dot "on" luma (keeps dots crisp)

def cell_color(block):
    """One 8x8 patch -> an (r,g,b). Sprite stays smooth; dot-art stays crisp."""
    rgb = block.reshape(-1, 3)
    mx, mn = rgb.max(1), rgb.min(1)
    sat = ((mx - mn) / (mx + 1e-6))
    if np.median(sat) > SAT_THR:                 # coloured sprite pixel
        return tuple(np.median(rgb, 0).astype(int))
    lum = rgb.mean(1)                            # grayscale dot region
    peak = lum.max()                            # max keeps a single dot bright
    if peak < DOT_THR:
        return (0, 0, 0)
    g = int(min(255, peak * 1.25))
    return (g, g, g)

def render(path):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    H, W, _ = a.shape
    cols, rows = W // CW, H // (CH)
    lines = []
    for r in range(rows):
        out = []
        for c in range(cols):
            x, y = c * CW, r * CH
            top = cell_color(a[y:y+8,       x:x+CW])
            bot = cell_color(a[y+8:y+CH,    x:x+CW])
            if top == (0, 0, 0) and bot == (0, 0, 0):
                out.append('\x1b[0m ')           # both dark -> blank
            else:
                out.append(f'\x1b[38;2;{top[0]};{top[1]};{top[2]}m'
                           f'\x1b[48;2;{bot[0]};{bot[1]};{bot[2]}m▀')
        lines.append(''.join(out) + '\x1b[0m')
    return '\n'.join(lines), cols, rows

if __name__ == '__main__':
    txt, cols, rows = render(sys.argv[1] if len(sys.argv) > 1 else
        '/home/xjhc/Screenshots/Screenshot_20260604_144154.png')
    sys.stdout.write(txt + '\n')
    sys.stderr.write(f'{cols} x {rows} block-glyph cells\n')
