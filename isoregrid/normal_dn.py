#!/usr/bin/env python3
"""normal_dn.py <mesh A tifxyz dir> <mesh B tifxyz dir> [--json out.json]

Normal distance of the valid nodes of mesh A to the surface of mesh B: for each A node, the nearest valid B node and
the distance along B's grid normal at that node. Grid nodes are typically 20 voxels apart, so the plain nearest-node
distance overstates how far two meshes of one sheet are apart; the normal distance measures whether A lies on B's
surface. B's normal is the cross product of its column and row tangents (mean of the forward and backward differences
that join valid nodes). Prints and optionally writes median, p10, p90, fraction <= 1 voxel, and the median nearest-node
distance. A nodes past B's edge still get the distance to B's tangent plane at its nearest edge node.
"""
import json, argparse
import numpy as np
import tifffile
from scipy.spatial import cKDTree


def load(d):
    return np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)


def normals(G):
    """Unit grid normals of an (H, W, 3) surface (NaN where a node has no valid neighbour along an axis)."""
    X = np.where((G[..., 2] > 0)[..., None], G, np.nan)
    t = []
    for ax in (1, 0):
        d = np.diff(X, axis=ax)
        pad = [(0, 0)] * 3
        pad[ax] = (0, 1); fwd = np.pad(d, pad, constant_values=np.nan)
        pad[ax] = (1, 0); bwd = np.pad(d, pad, constant_values=np.nan)
        s = np.nan_to_num(fwd) + np.nan_to_num(bwd)
        k = (~np.isnan(fwd[..., :1])).astype(float) + (~np.isnan(bwd[..., :1])).astype(float)
        t.append(np.where(k > 0, s / np.maximum(k, 1), np.nan))
    n = np.cross(t[0], t[1])
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def dn(B, P):
    """Normal distance of points P (N, 3) to surface B: (|normal distance|, nearest-node distance, B node (row, col))."""
    N = normals(B)
    ok = (B[..., 2] > 0) & np.all(np.isfinite(N), -1)
    V = B[ok]; Nv = N[ok]; rc = np.argwhere(ok)
    d, i = cKDTree(V).query(P)
    s = np.abs(np.sum((P - V[i]) * Nv[i], 1))
    return s, d, rc[i]


def summary(s, d):
    return dict(nodes=int(len(s)), dn_median=float(np.median(s)), dn_p10=float(np.percentile(s, 10)),
                dn_p90=float(np.percentile(s, 90)), dn_frac_le1=float((s <= 1).mean()),
                node_dist_median=float(np.median(d)))


def main():
    ap = argparse.ArgumentParser(description='normal distance of mesh A nodes to mesh B')
    ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--json', default='')
    a = ap.parse_args()
    A = load(a.a); B = load(a.b)
    P = A[A[..., 2] > 0]
    s, d, _ = dn(B, P)
    r = summary(s, d)
    if a.json:
        json.dump(r, open(a.json, 'w'), indent=1)
    print(json.dumps(r))


if __name__ == '__main__':
    main()
