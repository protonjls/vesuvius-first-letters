#!/usr/bin/env python3
"""Normal-distance place check between two tifxyz meshes (e.g. a claim mesh and a HerculeX piece or a regrow).

  place_check.py <ref tifxyz> <cand tifxyz> [--rows r0:r1] [--cols c0:c1] [--near 25]
  place_check.py --size <tifxyz>        (prints: rows cols stride, stride = round(1 / meta.json scale))

For every valid node of <ref> in the row/col box: the nearest valid node of <cand> and the distance ALONG <cand>'s local
surface normal there (normal dn). Nodes whose nearest <cand> node is farther than --near voxels are outside the overlap.
Reports the overlap share, the median / p10 / p90 normal dn and the share <= 5 voxels over the overlap, the raw
nearest-node distance for comparison, and the <cand> grid cells the overlap maps to (rows, cols, and render pixels =
cells * stride) so a feature on <ref> can be found on <cand>.

Why normal distance: tifxyz nodes are often 16-20 voxels apart, so two meshes of the SAME sheet read about half a node
spacing apart by nearest-node distance (we measured 8.5-8.9 voxels median on pairs whose normal distance was 0.3-0.7).
A sheet pitch is ~10-15 voxels in many scrolls, so node distance cannot tell 'same sheet' from 'next sheet'.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree


def load(d):
    import tifffile
    return np.stack([tifffile.imread(os.path.join(d, c + '.tif')) for c in 'xyz'], -1).astype(np.float64)


def stride_of(d):
    try:
        s = json.load(open(os.path.join(d, 'meta.json'))).get('scale', [1.0])
        return int(round(1.0 / float(s[0] if isinstance(s, (list, tuple)) else s)))
    except (OSError, ValueError, TypeError, ZeroDivisionError):
        return 1


def normals(G):
    dc = np.gradient(G, axis=1); dr = np.gradient(G, axis=0)
    n = np.cross(dc, dr)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def normal_dn(cand, P):
    """cand: (H, W, 3) grid (invalid nodes z <= 0); P: (n, 3) points -> normal dn, node dist, (row, col) of nearest node."""
    ok = cand[..., 2] > 0
    N = normals(cand)[ok]; V = cand[ok]; idx = np.argwhere(ok)
    d, i = cKDTree(V).query(P)
    s = np.abs(np.sum((P - V[i]) * N[i], 1))
    return s, d, idx[i]


def box(spec, n):
    if not spec:
        return 0, n
    a, b = spec.split(':')
    return int(a or 0), int(b or n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('cand', nargs='?')
    ap.add_argument('--rows'); ap.add_argument('--cols'); ap.add_argument('--near', type=float, default=25.0)
    ap.add_argument('--size', action='store_true')
    a = ap.parse_args()
    if a.size:
        import tifffile
        x = tifffile.imread(os.path.join(a.ref, 'x.tif'))
        print(x.shape[0], x.shape[1], stride_of(a.ref)); return
    R = load(a.ref); C = load(a.cand)
    r0, r1 = box(a.rows, R.shape[0]); c0, c1 = box(a.cols, R.shape[1])
    P = R[r0:r1, c0:c1].reshape(-1, 3); P = P[P[:, 2] > 0]
    s, d, rc = normal_dn(C, P)
    ov = d <= a.near; st = stride_of(a.cand)
    out = {'ref_nodes': int(len(P)), 'overlap_nodes': int(ov.sum()), 'overlap_share': round(float(ov.mean()), 3)}
    if ov.any():
        so = s[ov]; q = rc[ov]
        out.update({'normal_dn_median': round(float(np.median(so)), 2), 'normal_dn_p10': round(float(np.percentile(so, 10)), 2),
                    'normal_dn_p90': round(float(np.percentile(so, 90)), 2), 'share_le5': round(float((so <= 5).mean()), 3),
                    'node_dist_median': round(float(np.median(d[ov])), 2),
                    'cand_rows': [int(q[:, 0].min()), int(q[:, 0].max())], 'cand_cols': [int(q[:, 1].min()), int(q[:, 1].max())],
                    'cand_render_box_xyxy': [int(q[:, 1].min()) * st, int(q[:, 0].min()) * st,
                                             int(q[:, 1].max() + 1) * st, int(q[:, 0].max() + 1) * st]})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
