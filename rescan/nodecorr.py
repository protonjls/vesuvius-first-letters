#!/usr/bin/env python3
"""Node correspondence between two tifxyz meshes of one sheet in different volumes of a scroll (rescan/ known-answer demo).
  nodecorr.py --a <mesh A dir> --b <mesh B dir> [--affine transform.json[:inv]] --out corr.json [--step 4]
Maps A's vertices by the affine (XYZ, villa transform.json; ':inv' inverts) into B's voxel space, finds for every
--step-th valid A node the nearest B node (KD-tree), and fits a 2D affine node_B(row,col) ~ M @ [row_A, col_A, 1]
(least squares on pairs with 3D distance < --maxd). Reports 3D distance stats and the 2D fit residual."""
import argparse, json
import numpy as np, tifffile
from scipy.spatial import cKDTree


def load(d):
    x, y, z = (tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz')
    P = np.stack([x, y, z], -1); ok = np.isfinite(P).all(-1) & (P[..., 2] > 0) & ~(P == -1).all(-1)
    return P, ok


def aff(spec):
    p, inv = (spec[:-4], True) if spec.endswith(':inv') else (spec, False)
    M = np.array(json.load(open(p))['transformation_matrix'], float)
    if M.shape == (3, 4): M = np.vstack([M, [0, 0, 0, 1]])
    return np.linalg.inv(M) if inv else M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--affine', default=''); ap.add_argument('--out', required=True); ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--maxd', type=float, default=30.0)
    a = ap.parse_args()
    PA, okA = load(a.a); PB, okB = load(a.b)
    if a.affine:
        M = aff(a.affine); PA = PA @ M[:3, :3].T + M[:3, 3]
    rb, cb = np.nonzero(okB); tree = cKDTree(PB[rb, cb])
    ra, ca = np.nonzero(okA); sel = np.arange(0, len(ra), a.step); ra, ca = ra[sel], ca[sel]
    d, i = tree.query(PA[ra, ca])
    good = d < a.maxd
    X = np.c_[ra[good], ca[good], np.ones(good.sum())]
    Y = np.c_[rb[i[good]], cb[i[good]]]
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = np.linalg.norm(X @ W - Y, axis=1)
    out = {'a': a.a, 'b': a.b, 'affine': a.affine, 'n_pairs': int(len(d)), 'n_good': int(good.sum()),
           'dist3d_median': float(np.median(d)), 'dist3d_p90': float(np.percentile(d, 90)),
           'fit_nodeB_from_nodeA (rows: [rowA,colA,1] -> [rowB,colB])': W.tolist(),
           'fit_resid_nodesB_median': float(np.median(res)), 'fit_resid_nodesB_p95': float(np.percentile(res, 95)),
           'shapeA': list(okA.shape), 'shapeB': list(okB.shape)}
    json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
