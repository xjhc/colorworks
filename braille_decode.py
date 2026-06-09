#!/usr/bin/env python3
"""Minimal inverse-renderer: terminal Braille-dot screenshot -> Unicode Braille text.

Each Braille glyph is a 2x4 dot cell. We find the dot lattice, sample each dot,
and pack the on/off bits back into 0x2800-based code points.
"""
import sys
import numpy as np
from PIL import Image

# Dot -> Braille bit (col, row) within a 2-wide x 4-tall cell.
BIT = {(0,0):0x01,(0,1):0x02,(0,2):0x04,(0,3):0x40,
       (1,0):0x08,(1,1):0x10,(1,2):0x20,(1,3):0x80}

def decode(path, pitch=7.98, ox=5.1, oy=6.1, lum_thr=60, sat_thr=0.35):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    H, W, _ = a.shape
    lum = 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]
    mx, mn = a.max(2), a.min(2)
    sat = (mx-mn)/(mx+1e-6)

    cw, ch = pitch*2, pitch*4
    cols = int((W-ox)//cw)
    rows = int((H-oy)//ch)

    def on(cx, cy):
        x, y = int(round(cx)), int(round(cy))
        ls = lum[max(0,y-1):y+2, max(0,x-1):x+2]   # tight 3x3 at dot centre
        ss = sat[max(0,y-1):y+2, max(0,x-1):x+2]
        usable = ss < sat_thr          # ignore coloured foreground
        if not usable.any():
            return False
        return ls[usable].mean() > lum_thr

    lines = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            mask = 0
            for dy in range(4):
                for dx in range(2):
                    x = ox + c*cw + dx*pitch
                    y = oy + r*ch + dy*pitch
                    if on(x, y):
                        mask |= BIT[(dx,dy)]
            chars.append(chr(0x2800+mask))
        lines.append(''.join(chars).rstrip('⠀'))
    return '\n'.join(lines)

if __name__ == '__main__':
    print(decode(sys.argv[1] if len(sys.argv) > 1 else
                 '/home/xjhc/Screenshots/Screenshot_20260604_144154.png'))
