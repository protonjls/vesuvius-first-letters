#!/usr/bin/env python3
"""Show a map of mesh B in the grid frame of mesh A (same sheet, different GrowPatch/fit).

  mesh_resample.py <meshA> <meshB> <mapB.png> <out.png> --rows r0:r1 --cols c0:c1 [--ppcA 10] [--ppcB 10] [--maxd 25]

For each node of A in rows/cols, the nearest node of B is refined to a fractional B grid position by projecting
the residual onto B's local tangent basis (dP/dcol, dP/drow). The fractional map is upsampled to ppcA pixels per A cell
and mapB (ppcB pixels per B cell) is sampled there. Pixels whose A node lies > maxd voxels from B's surface are 0.
Prints the median/p90 of the normal-direction distance between the two surfaces over the window.
"""
import argparse
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def load(d):
    return np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('A'); ap.add_argument('B'); ap.add_argument('mapB'); ap.add_argument('out')
    ap.add_argument('--rows', required=True); ap.add_argument('--cols', required=True)
    ap.add_argument('--ppcA', type=float, default=10); ap.add_argument('--ppcB', type=float, default=10)
    ap.add_argument('--maxd', type=float, default=25)
    a = ap.parse_args()
    A = load(a.A); B = load(a.B)
    r0, r1 = map(int, a.rows.split(':')); c0, c1 = map(int, a.cols.split(':'))
    vb = B[..., 0] > 0; bi = np.argwhere(vb); tree = cKDTree(B[vb])
    W = A[r0:r1, c0:c1]; ok = W[..., 0] > 0
    fr = np.full(ok.shape, np.nan); fc = np.full(ok.shape, np.nan); dn = np.full(ok.shape, np.nan)
    d, idx = tree.query(W[ok])
    rb, cb = bi[idx, 0], bi[idx, 1]
    H, Wd = B.shape[:2]
    rbc = np.clip(rb, 1, H - 2); cbc = np.clip(cb, 1, Wd - 2)
    dc = (B[rbc, cbc + 1] - B[rbc, cbc - 1]) / 2; dr = (B[rbc + 1, cbc] - B[rbc - 1, cbc]) / 2
    res = W[ok] - B[rb, cb]
    M = np.stack([dc, dr], -1)  # (n,3,2)
    sol = np.einsum('nij,nj->ni', np.linalg.pinv(M), res)
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    fr[ok] = rb + np.clip(sol[:, 1], -1, 1); fc[ok] = cb + np.clip(sol[:, 0], -1, 1)
    dn[ok] = np.abs(np.sum(res * n, -1))
    print('normal distance A->B: median %.1f p90 %.1f vox; nearest-node dist median %.1f' %
          (np.nanmedian(dn), np.nanpercentile(dn, 90), np.median(d)))
    bad = ~ok | (dn > a.maxd)
    fr[bad] = np.nan; fc[bad] = np.nan
    s = a.ppcA
    oh, ow = int((r1 - r0) * s), int((c1 - c0) * s)
    # upsample with cell centres: A pixel (y,x) <-> A grid (r0 + y/s, c0 + x/s)
    def up(g):  # bilinear resize to (oh, ow), pixel-centre aligned, edges replicated
        g = np.nan_to_num(g, nan=-1e4).astype(np.float32)
        yy = (np.arange(oh) + 0.5) * g.shape[0] / oh - 0.5
        xx = (np.arange(ow) + 0.5) * g.shape[1] / ow - 0.5
        Y, X = np.meshgrid(yy, xx, indexing='ij')
        return map_coordinates(g, [Y, X], order=1, mode='nearest')
    my = up(fr) * a.ppcB; mx = up(fc) * a.ppcB
    im = np.asarray(Image.open(a.mapB).convert('L')).astype(np.float32)
    out = map_coordinates(im, [my, mx], order=1, mode='constant', cval=0)
    out[(my < 0) | (mx < 0)] = 0
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(a.out)
    np.save(a.out.rsplit('.', 1)[0] + '_dn.npy', dn.astype(np.float32))
    print('wrote', a.out, out.shape)


if __name__ == '__main__':
    main()
