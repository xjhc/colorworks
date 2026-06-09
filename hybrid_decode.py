#!/usr/bin/env python3
"""Color-routed hybrid decoder: Braille dot-art background + colour pixel-art sprite.

Master grid = the Braille cell lattice (2 dots wide x 4 dots tall).
Each cell is routed by the character's body silhouette (filled salmon|blue mask):
  * outside the sprite -> Unicode Braille glyph, tinted into 3 grey SHADES by brightness
  * inside the sprite  -> QUADRANT block (2x2 sub-cells) in salmon/blue/black, with the
                          silhouette defining clean edges (band, legs) and dark interior
                          pixels detected by low luma as the eyes.

Tunables (CLI):
  --shades A,B,C   brightness cuts. A = dot on-threshold ("thickness"); cells with
                   mean on-dot luma >=C render bright, >=B mid, else dim. (def 60,105,175)
  --dot N          rendered Braille dot size in px for --png ("thickness"). (def 6)
  --png FILE       also rasterise a verification PNG.
"""
import sys, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_erosion

BIT = {(0,0):0x01,(0,1):0x02,(0,2):0x04,(0,3):0x40,
       (1,0):0x08,(1,1):0x10,(1,2):0x20,(1,3):0x80}
# quadrant glyph by 4-bit mask: TL=1 TR=2 BL=4 BR=8
QUAD = {0:' ',1:'▘',2:'▝',3:'▀',4:'▖',5:'▌',6:'▞',7:'▛',
        8:'▗',9:'▚',10:'▐',11:'▜',12:'▄',13:'▙',14:'▟',15:'█'}
SAL=(195,105,85); BLU=(36,73,110); DARK=(0,0,0)
SHADE3=[(112,112,112),(180,180,180),(245,245,245)]   # dim, mid, bright

def _quad(eye, sal, blu, bod):
    """sub-quadrant boolean masks -> salmon / blue / dark / None(background).
    Foreground vs background is decided by the (clean) body silhouette, so thin
    parts like the headphone band and legs survive; eye needs solid coverage."""
    if bod.mean() < 0.40: return None        # outside the character -> transparent
    if eye.mean() > 0.30: return DARK        # solid black eye pixel
    return BLU if blu.sum() >= sal.sum() else SAL

def decode(path, shades=(60,105,175), pitch=7.98, ox=5.1, oy=6.1, sat_thr=0.35):
    on_thr,midc,hic = shades
    a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)
    H,W,_=a.shape; lum=a.mean(2); mx,mn=a.max(2),a.min(2); sat=(mx-mn)/(mx+1e-6)
    # global sprite masks: salmon body, blue headphones, and eyes (dark inside body)
    salmask=(sat>0.30)&(a[:,:,0]>a[:,:,2])
    blumask=(sat>0.30)&(a[:,:,2]>=a[:,:,0])
    body=binary_fill_holes(salmask|blumask)
    inner=binary_erosion(body, iterations=2)   # drop boundary gaps, keep deep interior
    eyemask=inner&(lum<45)       # near-black eyes: low luma (sat is unreliable here)
    cw,ch=pitch*2,pitch*4; cols,rows=int((W-ox)//cw),int((H-oy)//ch)
    grid=[]
    for r in range(rows):
        line=[]
        for c in range(cols):
            x0,y0=int(ox+c*cw),int(oy+r*ch)
            cell=a[y0:y0+int(ch), x0:x0+int(cw)]
            bo=body[y0:y0+int(ch), x0:x0+int(cw)]
            if bo.mean()>0.12:                  # routed by silhouette, not per-cell sat
                # ----- colour sprite -> quadrant block -----
                ey=eyemask[y0:y0+int(ch), x0:x0+int(cw)]
                sa=salmask[y0:y0+int(ch), x0:x0+int(cw)]
                bl=blumask[y0:y0+int(ch), x0:x0+int(cw)]
                hh,ww=cell.shape[0]//2, cell.shape[1]//2
                sub=lambda M: (M[:hh,:ww],M[:hh,ww:],M[hh:,:ww],M[hh:,ww:])
                quad=[_quad(e,s,b,d) for e,s,b,d in zip(sub(ey),sub(sa),sub(bl),sub(bo))]
                qr=[DARK if q is None else q for q in quad]   # None -> black background
                if all(q is None for q in quad): line.append(('blank',)); continue
                cnt={}
                for q in qr: cnt[q]=cnt.get(q,0)+1
                order=sorted(cnt, key=lambda k:-cnt[k])   # bg=most common, fg=next
                c0=order[0]; c1=order[1] if len(order)>1 else c0
                mask=sum(b for b,q in zip((1,2,4,8),qr) if q==c1)
                line.append(('quad',mask,c1,c0))
            else:
                # ----- dot-art -> shaded braille -----
                mask=0; brights=[]
                for dy in range(4):
                    for dx in range(2):
                        x=int(round(ox+c*cw+dx*pitch)); y=int(round(oy+r*ch+dy*pitch))
                        ls=lum[max(0,y-1):y+2, max(0,x-1):x+2]
                        ss=sat[max(0,y-1):y+2, max(0,x-1):x+2]; u=ss<sat_thr
                        if u.any():
                            v=ls[u].mean()
                            if v>on_thr: mask|=BIT[(dx,dy)]; brights.append(v)
                if mask==0: line.append(('blank',)); continue
                b=float(np.mean(brights)); sh=2 if b>=hic else 1 if b>=midc else 0
                line.append(('brl',mask,SHADE3[sh]))
        grid.append(line)
    return grid,cols,rows

def to_ansi(grid):
    out=[]
    for line in grid:
        s=[]
        for cell in line:
            if cell[0]=='blank': s.append('\x1b[0m ')
            elif cell[0]=='brl':
                _,m,col=cell
                s.append(f'\x1b[0;38;2;{col[0]};{col[1]};{col[2]}m{chr(0x2800+m)}')
            else:
                _,m,fg,bg=cell
                s.append(f'\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m'
                         f'\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m{QUAD[m]}')
        out.append(''.join(s)+'\x1b[0m')
    return '\n'.join(out)

def to_png(grid, path, dot=6, CW=16, CH=32):
    rows=len(grid); cols=max(len(l) for l in grid)
    img=np.zeros((rows*CH, cols*CW, 3), np.uint8)
    INV={v:k for k,v in BIT.items()}; off=(8-dot)//2
    for r,line in enumerate(grid):
        for c,cell in enumerate(line):
            y0,x0=r*CH,c*CW
            if cell[0]=='blank': continue
            if cell[0]=='brl':
                _,m,col=cell
                for bit,(dx,dy) in INV.items():
                    if m&bit:
                        yy,xx=y0+dy*8+off, x0+dx*8+off
                        img[yy:yy+dot, xx:xx+dot]=col
            else:
                _,m,fg,bg=cell
                img[y0:y0+CH, x0:x0+CW]=bg
                if m&1: img[y0:y0+CH//2,     x0:x0+CW//2]=fg
                if m&2: img[y0:y0+CH//2,     x0+CW//2:x0+CW]=fg
                if m&4: img[y0+CH//2:y0+CH,  x0:x0+CW//2]=fg
                if m&8: img[y0+CH//2:y0+CH,  x0+CW//2:x0+CW]=fg
    Image.fromarray(img).save(path)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('image', nargs='?', default='/home/xjhc/Screenshots/Screenshot_20260604_144154.png')
    ap.add_argument('--shades', default='60,105,175')
    ap.add_argument('--dot', type=int, default=6)
    ap.add_argument('--png')
    A=ap.parse_args()
    sh=tuple(int(x) for x in A.shades.split(','))
    grid,cols,rows=decode(A.image, shades=sh)
    sys.stdout.write(to_ansi(grid)+'\n')
    if A.png: to_png(grid, A.png, dot=A.dot)
    sys.stderr.write(f'{cols} x {rows} | shades(on,mid,hi)={sh} dot={A.dot}\n')
