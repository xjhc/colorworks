#!/usr/bin/env python3
"""Region-spec-driven decoder.

Everything a region needs is in one object:
    Region { name, kind, grid(x0,y0,px,py,cols,rows), clip, params }
  * grid   = pure geometry: where the pixels are and how big they are
  * kind   = which decoder turns clean pixels into glyphs ('braille' | 'sprite')
  * params = that decoder's config: palette + thresholds + shade cuts
  * clip   = optional source bbox the region is allowed to occupy

Pipeline per region:  crop+snap to the grid  ->  decode to glyphs  ->  draw.
render() iterates the region list in order (later regions composite on top), so
adding a second sprite is just appending another Region. Specs are static here;
auto-fitting them would only repopulate grid fields.
"""
import sys, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_erosion

# ---- glyph tables -------------------------------------------------------------
BIT = {(0,0):0x01,(0,1):0x02,(0,2):0x04,(0,3):0x40,
       (1,0):0x08,(1,1):0x10,(1,2):0x20,(1,3):0x80}
INV = {v:k for k,v in BIT.items()}

# ---- the one spec object ------------------------------------------------------
class Region:
    def __init__(s, name, kind, x0, y0, px, py, cols, rows, clip=None, **params):
        s.name, s.kind, s.clip, s.p = name, kind, clip, params
        s.x0, s.y0, s.px, s.py, s.cols, s.rows = x0, y0, px, py, cols, rows
    def cell(s, i, j):                       # source bbox of pixel (col i, row j)
        x, y = s.x0 + i*s.px, s.y0 + j*s.py
        return int(round(x)), int(round(y)), int(round(x+s.px)), int(round(y+s.py))

# ---- per-region scene masks (built once, from the region's own thresholds) ----
def sprite_masks(a, p, clip):
    lum = a.mean(2); mx, mn = a.max(2), a.min(2); sat = (mx-mn)/(mx+1e-6)
    sal = (sat > p['sat']) & (a[:,:,0] >  a[:,:,2])
    blu = (sat > p['sat']) & (a[:,:,2] >= a[:,:,0])
    if clip:                                 # confine detection to the region's box
        x0,y0,x1,y1 = clip; keep = np.zeros(sal.shape, bool); keep[y0:y1, x0:x1] = True
        sal &= keep; blu &= keep
    body = binary_fill_holes(sal | blu)
    eye  = binary_erosion(body, iterations=p['erode']) & (lum < p['eye_lum'])
    return dict(sal=sal, blu=blu, body=body, eye=eye)

# ---- decoders: clean pixels -> glyph data ------------------------------------
def decode_sprite(a, reg):
    """crop+snap the sprite to its native grid -> labels[rows][cols] in palette|None."""
    m = sprite_masks(a, reg.p, reg.clip); P = reg.p['palette']
    art = []
    for j in range(reg.rows):
        row = []
        for i in range(reg.cols):
            x0,y0,x1,y1 = reg.cell(i,j)
            if m['body'][y0:y1,x0:x1].mean() < reg.p['body']: row.append(None); continue
            if m['eye'][y0:y1,x0:x1].mean() > reg.p['eye_cov']: row.append(P['dark']); continue
            row.append(P['blu'] if m['blu'][y0:y1,x0:x1].sum() >= m['sal'][y0:y1,x0:x1].sum()
                       else P['sal'])
        art.append(row)
    return art

def decode_braille(a, reg):
    """group the dot lattice 2x4 -> (mask, shade) per cell; shade by mean brightness."""
    lum = a.mean(2); mx, mn = a.max(2), a.min(2); sat = (mx-mn)/(mx+1e-6)
    on, midc, hic = reg.p['shades']; cols, rows = reg.cols//2, reg.rows//4
    grid = []
    for cr in range(rows):
        line = []
        for cc in range(cols):
            mask = 0; br = []
            for dy in range(4):
                for dx in range(2):
                    x,y,_,_ = reg.cell(2*cc+dx, 4*cr+dy)   # lattice point = dot centre
                    ls = lum[max(0,y-1):y+2, max(0,x-1):x+2]; ss = sat[max(0,y-1):y+2, max(0,x-1):x+2]
                    u = ss < reg.p['sat']
                    if u.any() and ls[u].mean() > on: mask |= BIT[(dx,dy)]; br.append(ls[u].mean())
            if mask == 0: line.append(None); continue
            b = float(np.mean(br)); sh = 2 if b>=hic else 1 if b>=midc else 0
            line.append((mask, reg.p['shade_cols'][sh]))
        grid.append(line)
    return grid

# ---- draw onto the shared source-resolution canvas ---------------------------
def draw_sprite(canvas, reg, art):
    pw = int(round(reg.px)) + 1
    for j, row in enumerate(art):
        for i, lab in enumerate(row):
            if lab is None: continue
            x0,y0,_,_ = reg.cell(i,j); canvas[y0:y0+pw, x0:x0+pw] = lab

def draw_braille(canvas, reg, grid, dot):
    for cr, line in enumerate(grid):
        for cc, cell in enumerate(line):
            if cell is None: continue
            mask, col = cell
            for bit,(dx,dy) in INV.items():
                if mask & bit:
                    x,y,_,_ = reg.cell(2*cc+dx, 4*cr+dy)   # lattice point = dot centre
                    cx,cy = x-dot//2, y-dot//2
                    canvas[max(0,cy):cy+dot, max(0,cx):cx+dot] = col

# ---- text emitters (each region is its own grid) -----------------------------
def sprite_text(art):
    return '\n'.join(''.join('\x1b[0m ' if l is None else
        f'\x1b[38;2;{l[0]};{l[1]};{l[2]}m█' for l in row)+'\x1b[0m' for row in art)
def braille_text(grid):
    out=[]
    for line in grid:
        s=''
        for cell in line:
            if cell is None: s+='\x1b[0m '
            else: m,col=cell; s+=f'\x1b[0;38;2;{col[0]};{col[1]};{col[2]}m{chr(0x2800+m)}'
        out.append(s+'\x1b[0m')
    return '\n'.join(out)

DECODERS = {'braille': (decode_braille, draw_braille, braille_text),
            'sprite':  (decode_sprite,  draw_sprite,  sprite_text)}

# ---- single render loop over all regions -------------------------------------
def render(path, regions, dot=6):
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    H, W, _ = a.shape; canvas = np.zeros((H, W, 3), np.uint8); texts = {}
    for reg in regions:                       # order = compositing order
        dec, drw, txt = DECODERS[reg.kind]
        data = dec(a, reg)
        drw(canvas, reg, data, dot) if reg.kind == 'braille' else drw(canvas, reg, data)
        texts[reg.name] = txt(data)
    return canvas, texts

# ---- the static spec (everything that defines this screenshot) ----------------
SHADE3 = [(112,112,112),(180,180,180),(245,245,245)]
PALETTE = {'sal':(195,105,85), 'blu':(36,73,110), 'dark':(0,0,0)}

def default_regions(W, H, shades=(60,105,175)):
    return [
        Region('bg', 'braille', 5.1, 6.1, 7.98, 7.98,
               cols=int((W-5.1)//7.98), rows=int((H-6.1)//7.98),
               shades=shades, shade_cols=SHADE3, sat=0.35),
        Region('char', 'sprite', 674.0, 595.0, 12.4, 12.4, cols=23, rows=22,
               clip=(664,588,964,876), palette=PALETTE,
               sat=0.30, body=0.45, eye_cov=0.25, eye_lum=45, erode=2),
    ]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image', nargs='?', default='/home/xjhc/Screenshots/Screenshot_20260604_144154.png')
    ap.add_argument('--shades', default='60,105,175'); ap.add_argument('--dot', type=int, default=6)
    ap.add_argument('--png', default='/tmp/composite.png')
    A = ap.parse_args(); sh = tuple(int(x) for x in A.shades.split(','))
    W, H = Image.open(A.image).size
    regions = default_regions(W, H, shades=sh)
    canvas, texts = render(A.image, regions, dot=A.dot)
    Image.fromarray(canvas).save(A.png)
    sys.stdout.write(texts['char'] + '\n')    # character layer to stdout
    sys.stderr.write(f"composite -> {A.png}  |  regions: " +
                     ", ".join(f"{r.name}[{r.kind} {r.cols}x{r.rows}]" for r in regions) + '\n')
