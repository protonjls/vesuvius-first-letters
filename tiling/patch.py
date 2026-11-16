#!/usr/bin/env python3
"""Mask a GrowPatch mesh to the part that stays on the predicted sheet, and mark the seeds it covers (tiling step 2).

  patch.py mask <grow_dir> <m7_render.zarr | .npy> <seeds.tsv> <seed_idx> <out_dir> <covered.txt>
               [--thr 128] [--dedup 5] [--up 5] [--minarea 0.3] [--um 9.362] [--keep-open 3]

<grow_dir>: tifxyz (x/y/z.tif, meta.json; scale s => one node per 1/s vox, render px = node / s at --scale 1).
<m7 render>: vc_render_tifxyz of the surface-prediction zarr along the mesh, NL layers (layer NL//2 = mesh surface).
On-sheet (render px) = max over the central 5 layers >= thr, then a k x k morphological opening; the connected component
(8-conn) holding the seed's nearest node is kept. Node mask = fraction of its render block (1/s px around the node)
inside the component >= 0.5. Off-mask nodes are set to -1, the grid is cropped to the mask bbox, meta.json bbox/area
updated, written to <out_dir> (skipped if on-sheet area < --minarea cm2). Seeds of seeds.tsv within --dedup vox of the
masked surface (nodes bilinearly densified x --up) are appended to covered.txt (the seed itself always is).
Prints one JSON line of stats.
"""
import argparse, json, os, sys
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


def load_mesh(d):
    P = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float32)
    return P, json.load(open(f'{d}/meta.json'))


def load_render(p):
    if p.endswith('.npy'):
        return np.load(p)
    import zarr
    return np.asarray(zarr.open(p, mode='r')['0'])


def densify(P, ok, up):
    """Bilinear densification of valid quads (all 4 corners valid) -> (N,3) points."""
    H, W, _ = P.shape
    q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    ii, jj = np.nonzero(q)
    if len(ii) == 0:
        return P[ok].reshape(-1, 3)
    t = (np.arange(up) / up).astype(np.float32)
    a, b = np.meshgrid(t, t, indexing='ij'); a = a.ravel()[None, :, None]; b = b.ravel()[None, :, None]
    p00 = P[ii, jj][:, None]; p10 = P[ii + 1, jj][:, None]; p01 = P[ii, jj + 1][:, None]; p11 = P[ii + 1, jj + 1][:, None]
    D = (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
    return np.concatenate([D.reshape(-1, 3), P[ok].reshape(-1, 3)])


def node_mask(comp, s_px, shape):
    """Fraction of each node's render block inside comp; node (i,j) <-> render px (i*s_px, j*s_px)."""
    H, W = shape
    Hr, Wr = comp.shape
    h = s_px // 2
    pad = np.zeros((H * s_px + s_px, W * s_px + s_px), np.float32)
    inb = np.zeros_like(pad)                      # render px that exist (a first-row/col node's block is half outside)
    hh, ww = min(Hr, pad.shape[0] - h), min(Wr, pad.shape[1] - h)
    pad[h:h + hh, h:h + ww] = comp[:hh, :ww]; inb[h:h + hh, h:h + ww] = 1
    num = pad[:H * s_px, :W * s_px].reshape(H, s_px, W, s_px).sum((1, 3))
    den = inb[:H * s_px, :W * s_px].reshape(H, s_px, W, s_px).sum((1, 3))
    return num / np.maximum(den, 1)


def cmd_mask(a):
    P, meta = load_mesh(a.grow_dir)
    ok = P[..., 0] != -1
    H, W, _ = P.shape
    s = float(meta.get('scale', [0.05, 0.05])[0]); s_px = int(round(1 / s))
    R = load_render(a.render)
    NL = R.shape[0]; c = NL // 2
    M = R[max(0, c - 2):c + 3].max(0) >= a.thr
    if a.keep_open > 1:
        M = ndimage.binary_opening(M, structure=np.ones((a.keep_open, a.keep_open), bool))
    seeds = [l.rstrip('\n').split('\t') for l in open(a.seeds)][1:]
    sd = {int(r[0]): np.array([float(r[3]), float(r[4]), float(r[5])], np.float32) for r in seeds}
    S0 = sd[a.seed_idx]
    # nearest valid node to the seed
    vi = np.argwhere(ok)
    dn = np.linalg.norm(P[ok] - S0, axis=1); k = int(np.argmin(dn)); si, sj = vi[k]
    lab, n = ndimage.label(M, structure=np.ones((3, 3), bool))
    # component at the seed node's render px (search a small neighbourhood for a labelled px)
    r0, c0 = int(si * s_px), int(sj * s_px)
    win = lab[max(0, r0 - s_px):r0 + s_px + 1, max(0, c0 - s_px):c0 + s_px + 1]
    labs, cnt = np.unique(win[win > 0], return_counts=True)
    st = dict(seed=a.seed_idx, grid=[H, W], valid_nodes=int(ok.sum()), seed_node=[int(si), int(sj)],
              seed_node_dist=round(float(dn[k]), 2), render=list(R.shape), onsheet_px_frac=round(float(M.mean()), 4))
    covered = {a.seed_idx}
    if len(labs) == 0:
        st.update(action='no_component', onsheet_nodes=0, area_cm2=0.0)
    else:
        L = labs[np.argmax(cnt)]
        comp = (lab == L).astype(np.float32)
        nm = (node_mask(comp, s_px, (H, W)) >= 0.5) & ok
        area = float(nm.sum()) * (s_px * a.um * 1e-4) ** 2
        st.update(onsheet_nodes=int(nm.sum()), area_cm2=round(area, 3), comp_px=int(comp.sum()))
        if nm.sum() >= 4:
            D = densify(P, nm, a.up)
            tree = cKDTree(D)
            idx = np.array(sorted(sd)); X = np.stack([sd[i] for i in idx])
            d, _ = tree.query(X, distance_upper_bound=a.dedup + 1e-3)
            covered |= set(int(i) for i in idx[np.isfinite(d) & (d <= a.dedup)])
        if area >= a.minarea:
            rr, cc = np.nonzero(nm)
            ra, rb, ca, cb = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            Q = P[ra:rb, ca:cb].copy(); Q[~nm[ra:rb, ca:cb]] = -1
            os.makedirs(a.out_dir, exist_ok=True)
            for i, ch in enumerate('xyz'):
                tifffile.imwrite(f'{a.out_dir}/{ch}.tif', Q[..., i].astype(np.float32))
            V = Q[nm[ra:rb, ca:cb]]
            m2 = dict(meta)
            m2['bbox'] = [V.min(0).tolist(), V.max(0).tolist()]
            m2['area_cm2'] = area
            m2['tiling'] = dict(seed=a.seed_idx, crop=[int(ra), int(rb), int(ca), int(cb)], thr=a.thr)
            json.dump(m2, open(f'{a.out_dir}/meta.json', 'w'), indent=1)
            st.update(action='kept', crop=[int(ra), int(rb), int(ca), int(cb)])
        else:
            st.update(action='small')
    st['covered_new'] = len(covered)
    with open(a.covered, 'a') as f:
        f.write(''.join(f'{i}\n' for i in sorted(covered)))
    print(json.dumps(st))


def cmd_next(a):
    """Print the next seed line (TSV fields idx x y z) not done/covered and in this part, or nothing."""
    seeds = [l.rstrip('\n').split('\t') for l in open(a.seeds)][1:]
    skip = set()
    for f in a.skip:
        if os.path.exists(f):
            skip |= {int(x) for x in open(f).read().split()}
    if a.part:
        k, K = map(int, a.part.split('/'))
        tc = sorted({int(r[2]) for r in seeds}); n = len(tc)
        lo, hi = tc[0] + (tc[-1] - tc[0] + 1) * k // K, tc[0] + (tc[-1] - tc[0] + 1) * (k + 1) // K
        seeds = [r for r in seeds if lo <= int(r[2]) < hi]
    rest = [r for r in seeds if int(r[0]) not in skip]
    if rest:
        r = rest[0]
        print(r[0], r[3], r[4], r[5], len(rest))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    m = sp.add_parser('mask')
    for k in ('grow_dir', 'render', 'seeds'):
        m.add_argument(k)
    m.add_argument('seed_idx', type=int); m.add_argument('out_dir'); m.add_argument('covered')
    m.add_argument('--thr', type=int, default=128); m.add_argument('--dedup', type=float, default=5.0)
    m.add_argument('--up', type=int, default=5); m.add_argument('--minarea', type=float, default=0.3)
    m.add_argument('--um', type=float, default=9.362); m.add_argument('--keep-open', type=int, default=3)
    n = sp.add_parser('next')
    n.add_argument('seeds'); n.add_argument('skip', nargs='*'); n.add_argument('--part', default='')
    a = ap.parse_args()
    {'mask': cmd_mask, 'next': cmd_next}[a.cmd](a)


if __name__ == '__main__':
    main()
