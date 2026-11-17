#!/usr/bin/env python3
"""How much of a tiling run's on-sheet surface the rank.py windows can show a reviewer.

  review_coverage.py <rank.npz> <patches> [<patches> ...] [--bin 20] [--tol 15] [--top 36] [--px-per-cell 10]

<patches> = tile_grow.sh output dir(s) holding wNNNN/{x,y,z}.tif (off-sheet nodes -1), or tar(s) of them.
rank.npz = rank.py --out-npz (every window that passed --min-valid: winding, y, x, side in PNG px). With 20-voxel mesh
cells and png_ds 2 a cell is 10 PNG px (--px-per-cell). A node is 'scored' if it lies inside a scored window of its own
patch (either order). The on-sheet surface is counted once in 3D: one representative node per --bin voxel cube over all
patches. Reports the share of representatives within --tol voxels of a scored node, and of a node in a top-N window.
"""
import argparse, io, os, re, tarfile
import numpy as np
import tifffile
from scipy.spatial import cKDTree


def patches(srcs):
    for src in srcs:
        if os.path.isdir(src):
            for root, _, files in sorted(os.walk(src)):
                if all(f'{c}.tif' in files for c in 'xyz') and re.fullmatch(r'w\d+', os.path.basename(root)):
                    yield os.path.basename(root), np.stack([tifffile.imread(f'{root}/{c}.tif') for c in 'xyz'], -1)
        else:
            with tarfile.open(src) as t:
                mem = {}
                for m in t.getmembers():
                    g = re.search(r'(?:^|/)(w\d+)/([xyz])\.tif$', m.name)
                    if g:
                        mem.setdefault(g.group(1), {})[g.group(2)] = m
                for w, d in sorted(mem.items()):
                    if len(d) == 3:
                        yield w, np.stack([tifffile.imread(io.BytesIO(t.extractfile(d[c]).read())) for c in 'xyz'], -1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('patches', nargs='+')
    ap.add_argument('--bin', type=float, default=20.0); ap.add_argument('--tol', type=float, default=15.0)
    ap.add_argument('--top', type=int, default=36); ap.add_argument('--px-per-cell', type=float, default=10.0)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    W = {k: z[k].tolist() for k in ('winding', 'y', 'x', 'side', 'score')}
    wins = {}
    for rank, i in enumerate(np.argsort(-np.array(W['score']), kind='stable')):
        wins.setdefault(W['winding'][i], []).append((W['y'][i], W['x'][i], W['side'][i], rank))
    allp, scp, topp, seen, nsc = [], [], [], set(), 0
    for w, P in patches(a.patches):
        if w in seen:
            continue
        seen.add(w)
        ok = (P > 0).all(-1)
        S = np.zeros(ok.shape, bool); T = np.zeros(ok.shape, bool)
        for (y, x, s, rank) in wins.get(w, []):
            r0, r1 = int(y / a.px_per_cell), int(np.ceil((y + s) / a.px_per_cell))
            c0, c1 = int(x / a.px_per_cell), int(np.ceil((x + s) / a.px_per_cell))
            S[r0:r1, c0:c1] = True
            if rank < a.top:
                T[r0:r1, c0:c1] = True
        nsc += w in wins
        allp.append(P[ok]); scp.append(P[ok & S]); topp.append(P[ok & T])
    A = np.concatenate(allp).astype(np.float32)
    _, idx = np.unique(np.floor(A / a.bin).astype(np.int64), axis=0, return_index=True)
    R = A[idx]
    res = {'patches': len(seen), 'patches_scored': nsc, 'onsheet_nodes': int(len(A)), 'representatives': int(len(R))}
    for name, L in (('scored', scp), (f'top{a.top}', topp)):
        B = np.concatenate(L).astype(np.float32)
        res[f'frac_{name}'] = float((cKDTree(B).query(R, distance_upper_bound=a.tol)[0] <= a.tol).mean()) if len(B) else 0.0
    print(res)


if __name__ == '__main__':
    main()
