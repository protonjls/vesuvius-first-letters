#!/usr/bin/env python3
"""How much of a reference sheet A (tifxyz) a grown patch B covers: fraction of A's nodes within --tol voxels of B.

  coverage.py <meshA_ref> <meshB> [--rows r0:r1] [--cols c0:c1] [--tol 5] [--json out.json]

For each valid node of A in the row/column window, the nearest valid node of B is found and the residual is projected
on B's local normal (central differences of B's grid). A node counts as covered when that normal distance is <= tol and
the in-plane residual is under 1.5 B cells (otherwise the nearest node is just B's edge); nodes whose nearest B node
has an invalid (-1) neighbour do not count. Pure geometry, no maps. Prints one JSON line:
  {"frac_within_tol", "median_dn_within_tol", "n_ref_nodes", "B_valid_nodes"}.
Example (PHerc0306B awarded region): A = the winners' claim mesh, --rows 82:137 --cols 121:341.
"""
import argparse, json
import numpy as np
import tifffile
from scipy.spatial import cKDTree


def load(d):
    return np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)


def coverage(A, B, rows, cols, tol):
    vb = B[..., 0] > 0
    bi = np.argwhere(vb); tree = cKDTree(B[vb])
    W = A[rows[0]:rows[1], cols[0]:cols[1]]; ok = W[..., 0] > 0
    P = W[ok]
    _, idx = tree.query(P); rb, cb = bi[idx, 0], bi[idx, 1]
    H, Wd = B.shape[:2]
    rbc = np.clip(rb, 1, H - 2); cbc = np.clip(cb, 1, Wd - 2)
    dc = (B[rbc, cbc + 1] - B[rbc, cbc - 1]) / 2; dr = (B[rbc + 1, cbc] - B[rbc - 1, cbc]) / 2
    n = np.cross(dc, dr); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    badn = (B[rbc, cbc + 1, 0] <= 0) | (B[rbc, cbc - 1, 0] <= 0) | (B[rbc + 1, cbc, 0] <= 0) | (B[rbc - 1, cbc, 0] <= 0)
    res = P - B[rb, cb]
    dn = np.abs(np.sum(res * n, -1))
    inpl = np.linalg.norm(res - np.sum(res * n, -1)[:, None] * n, axis=-1)
    cell = np.median(np.linalg.norm(dc, axis=-1))
    dn[badn | (inpl > 1.5 * cell)] = np.inf
    within = dn <= tol
    return {'frac_within_tol': float(within.mean()) if within.size else 0.0,
            'median_dn_within_tol': float(np.median(dn[within])) if within.any() else None,
            'n_ref_nodes': int(ok.sum()), 'B_valid_nodes': int(vb.sum()), 'tol': tol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('A'); ap.add_argument('B')
    ap.add_argument('--rows', default=''); ap.add_argument('--cols', default='')
    ap.add_argument('--tol', type=float, default=5.0); ap.add_argument('--json', default='')
    a = ap.parse_args()
    A = load(a.A); B = load(a.B)
    rows = tuple(map(int, a.rows.split(':'))) if a.rows else (0, A.shape[0])
    cols = tuple(map(int, a.cols.split(':'))) if a.cols else (0, A.shape[1])
    out = coverage(A, B, rows, cols, a.tol)
    out.update({'A': a.A, 'B': a.B, 'rows': list(rows), 'cols': list(cols)})
    print(json.dumps(out))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
