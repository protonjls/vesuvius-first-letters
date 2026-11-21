#!/usr/bin/env python3
"""Do the ink forms on one mesh reappear on another mesh of the same place?

  reappear.py <meshA> <mapsA> <meshB> <mapsB> <out_dir> --rows r0:r1 --cols c0:c1
              [--offB x0,y0] [--ppcA N] [--ppcB N] [--maxd 10] [--models reader-v3,rectina-m4] [--shift 0]

meshA / meshB: tifxyz dirs (x.tif y.tif z.tif meta.json), e.g. a HerculeX piece and a GrowPatch mesh, or two
GrowPatch meshes grown from different seeds. mapsA / mapsB: output dirs of ../followup/confirm_region.sh (or
../herculex/piece_confirm.sh): ct_d0.png and maps/d0/<model>_{fwd,rev}.png, rendered at 1 voxel per pixel with grid
node (r, c) at render pixel (r * ppc, c * ppc), ppc = 1 / meta.json scale (16 for HerculeX pieces, 20 for GrowPatch).
--offB x0,y0: render-pixel origin of the mapsB window (the x0 y0 given to confirm_region.sh); default 0,0.

For every pixel of A's window (rows/cols are A grid rows/cols) the A surface point is located on B: nearest B node,
refined to a fractional B grid position on B's local tangent plane; points more than --maxd voxels from B along B's
normal are dropped. Each B map is sampled there (bilinear), so it lands in A's pixel frame. Valid pixels = A's
ct_d0 > 0 and the resampled ct_d0 > 0. For every model present on both meshes and every pair of depth orders
(A fwd/rev x B fwd/rev; the meshes' normals may point opposite ways) the Pearson r over valid pixels is reported.
--shift N rolls the resampled B maps by N pixels along x and y first: a negative control (use about half a letter).
Same ink on the same surface gives high r in one order pair; another sheet, a misplaced mesh, or ink that is not
tied to the surface gives r near 0.

Writes <out_dir>/reappear.tsv, reappear.json (dn statistics, r table, best pair per model) and reappear_sheet.jpg
(A map | resampled B map, best order pair per model, same window, one shared contrast per panel).
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def load_mesh(d):
    return np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)


def mesh_ppc(d):
    m = json.load(open(f'{d}/meta.json'))
    return 1.0 / float(m['scale'][0])


def locate(A, B, r0, r1, c0, c1, maxd):
    """Fractional B grid position (fr, fc) and normal distance dn for every A node in the window (NaN = not on B)."""
    vb = B[..., 2] > 0
    bi = np.argwhere(vb)
    tree = cKDTree(B[vb])
    W = A[r0:r1, c0:c1]
    ok = W[..., 2] > 0
    fr = np.full(ok.shape, np.nan); fc = np.full(ok.shape, np.nan); dn = np.full(ok.shape, np.nan)
    if not ok.any():
        return fr, fc, dn, np.array([])
    d, idx = tree.query(W[ok])
    rb, cb = bi[idx, 0], bi[idx, 1]
    H, Wd = B.shape[:2]
    rc = np.clip(rb, 1, H - 2); cc = np.clip(cb, 1, Wd - 2)
    dc = (B[rc, cc + 1] - B[rc, cc - 1]) / 2
    dr = (B[rc + 1, cc] - B[rc - 1, cc]) / 2
    res = W[ok] - B[rb, cb]
    sol = np.einsum('nij,nj->ni', np.linalg.pinv(np.stack([dc, dr], -1)), res)
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    fr[ok] = rb + np.clip(sol[:, 1], -1, 1)
    fc[ok] = cb + np.clip(sol[:, 0], -1, 1)
    dn[ok] = np.abs(np.sum(res * n, -1))
    bad = ~np.isfinite(dn) | (dn > maxd)
    fr[bad] = np.nan; fc[bad] = np.nan
    return fr, fc, dn, d


def pixel_coords(fr, fc, ppcA, ppcB, offB):
    """B render-pixel coordinates (my, mx) for every A pixel of the window, renderer convention on both sides:
    A pixel (y, x) <-> A grid (y / ppcA, x / ppcA) relative to the window origin node."""
    oh, ow = int(round(fr.shape[0] * ppcA)), int(round(fr.shape[1] * ppcA))
    Y, X = np.meshgrid(np.arange(oh) / ppcA, np.arange(ow) / ppcA, indexing='ij')
    inside = np.isfinite(fr).astype(np.float32)
    w = map_coordinates(inside, [Y, X], order=1, mode='nearest')
    g = lambda a: map_coordinates(np.nan_to_num(a, nan=0.0).astype(np.float64), [Y, X], order=1, mode='nearest')
    # bilinear weights of a NaN node would pull coordinates toward 0: keep only pixels whose 4 nodes are all on B
    my = g(fr) * ppcB - offB[1]; mx = g(fc) * ppcB - offB[0]
    good = w > 0.999
    return my, mx, good


def sample(img, my, mx, good):
    out = map_coordinates(img, [my, mx], order=1, mode='constant', cval=0.0)
    inb = (my >= 0) & (mx >= 0) & (my <= img.shape[0] - 1) & (mx <= img.shape[1] - 1)
    out[~(good & inb)] = 0
    return out


def gray(f):
    return np.asarray(Image.open(f).convert('L')).astype(np.float32)


def pearson(a, b, m):
    if m.sum() < 100:
        return float('nan')
    x = a[m]; y = b[m]
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def run(meshA, mapsA, meshB, mapsB, out, rows, cols, offB=(0, 0), ppcA=None, ppcB=None, maxd=10.0,
        models=('reader-v3', 'rectina-m4'), shift=0, sheet=True):
    os.makedirs(out, exist_ok=True)
    ppcA = ppcA or mesh_ppc(meshA); ppcB = ppcB or mesh_ppc(meshB)
    r0, r1 = rows; c0, c1 = cols
    A = load_mesh(meshA); B = load_mesh(meshB)
    fr, fc, dn, dnode = locate(A, B, r0, r1, c0, c1, maxd)
    my, mx, good = pixel_coords(fr, fc, ppcA, ppcB, offB)
    h, w = my.shape
    y0, x0 = int(round(r0 * ppcA)), int(round(c0 * ppcA))
    crop = lambda a: np.pad(a, ((0, max(0, y0 + h - a.shape[0])), (0, max(0, x0 + w - a.shape[1]))))[y0:y0 + h, x0:x0 + w]
    ctA = crop(gray(f'{mapsA}/ct_d0.png'))
    ctB = sample(gray(f'{mapsB}/ct_d0.png'), my, mx, good)
    valid = (ctA > 0) & (ctB > 0)
    fin = dn[np.isfinite(dn)]
    info = {'meshA': meshA, 'meshB': meshB, 'rows': [r0, r1], 'cols': [c0, c1], 'ppcA': ppcA, 'ppcB': ppcB,
            'offB': list(offB), 'maxd': maxd, 'shift': shift,
            'normal_dn_median': float(np.median(fin)) if fin.size else None,
            'normal_dn_p90': float(np.percentile(fin, 90)) if fin.size else None,
            'nearest_node_dist_median': float(np.median(dnode)) if len(dnode) else None,
            'valid_pixels': int(valid.sum()), 'window_pixels': int(h * w), 'r': {}, 'best': {}}
    panels = []
    for m in models:
        for oa in ('fwd', 'rev'):
            fa = f'{mapsA}/maps/d0/{m}_{oa}.png'
            if not os.path.exists(fa):
                continue
            a = crop(gray(fa))
            for ob in ('fwd', 'rev'):
                fb = f'{mapsB}/maps/d0/{m}_{ob}.png'
                if not os.path.exists(fb):
                    continue
                b = sample(gray(fb), my, mx, good)
                if shift:
                    b = np.roll(b, (shift, shift), (0, 1))
                r = pearson(a, b, valid)
                info['r'][f'{m} A{oa} B{ob}'] = r
                bst = info['best'].get(m)
                if np.isfinite(r) and (bst is None or r > bst['r']):
                    info['best'][m] = {'r': r, 'A': oa, 'B': ob}
                    panels = [p for p in panels if p[0] != m] + [(m, f'{m} A{oa}', a, f'{m} B{ob} r={r:.2f}', b)]
    with open(f'{out}/reappear.tsv', 'w') as fh:
        fh.write(f'# A {meshA} rows {r0}:{r1} cols {c0}:{c1} | B {meshB} offB {offB[0]},{offB[1]} | shift {shift}\n')
        fh.write(f"# normal distance A->B median {info['normal_dn_median']} p90 {info['normal_dn_p90']} vox; "
                 f"valid pixels {info['valid_pixels']} of {info['window_pixels']}\n")
        fh.write('pair\tr\n')
        for k, v in info['r'].items():
            fh.write(f'{k}\t{v:.3f}\n')
    json.dump(info, open(f'{out}/reappear.json', 'w'), indent=1)
    if sheet and panels:
        def stretch(a):
            v = a[valid] if valid.sum() > 100 else a.ravel()
            lo, hi = np.percentile(v, 50), np.percentile(v, 99.7)
            return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255 * valid).astype(np.uint8)
        sc = min(1.0, 700.0 / max(h, w)); th, tw = max(1, int(h * sc)), max(1, int(w * sc))
        tiles = [t for p in panels for t in ((p[1], p[2]), (p[3], p[4]))]
        im = Image.new('L', (tw * len(tiles), th + 20), 0); dr = ImageDraw.Draw(im)
        for i, (lab, a) in enumerate(tiles):
            im.paste(Image.fromarray(stretch(a)).resize((tw, th)), (i * tw, 20)); dr.text((i * tw + 4, 4), lab, fill=255)
        im.save(f'{out}/reappear_sheet.jpg', quality=90)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('meshA'); ap.add_argument('mapsA'); ap.add_argument('meshB'); ap.add_argument('mapsB'); ap.add_argument('out')
    ap.add_argument('--rows', required=True); ap.add_argument('--cols', required=True)
    ap.add_argument('--offB', default='0,0'); ap.add_argument('--ppcA', type=float); ap.add_argument('--ppcB', type=float)
    ap.add_argument('--maxd', type=float, default=10.0); ap.add_argument('--models', default='reader-v3,rectina-m4')
    ap.add_argument('--shift', type=int, default=0)
    a = ap.parse_args()
    rr = tuple(map(int, a.rows.split(':'))); cc = tuple(map(int, a.cols.split(':')))
    off = tuple(map(int, a.offB.split(',')))
    info = run(a.meshA, a.mapsA, a.meshB, a.mapsB, a.out, rr, cc, off, a.ppcA, a.ppcB, a.maxd, a.models.split(','), a.shift)
    print(open(f'{a.out}/reappear.tsv').read(), end='')
    print('best:', json.dumps(info['best']))
    if not info['best']:
        sys.exit('no model maps present on both meshes')


if __name__ == '__main__':
    main()
