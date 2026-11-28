#!/usr/bin/env python3
"""iso_regrid.py <in tifxyz dir> <out tifxyz dir> [--step 20] [--iters 200]

Isometric re-grid of one tifxyz surface. Spiral-fit windings (villa fit_spiral) keep a regular grid in (angle, z), so
their column edges vary from about 15 to 53 voxels in 3D, and a render at one pixel per grid unit squashes the text
horizontally. This tool re-samples the same surface on a grid whose edges are all <step> voxels long:
  1. mass-spring flatten: 2D positions uv of the valid nodes minimising sum (|uv_i - uv_j| - |X_i - X_j|)^2 over grid
     edges (row, column and both diagonals of every valid quad), scipy least_squares (trf, sparse Jacobian); init =
     per-row arc length from the middle column (u) and per-column arc length from the middle row (v); two nodes are
     softly anchored (gauge);
  2. rotate uv so the 3D z gradient points along the input's row direction (input meta z_direction kept), and keep the
     input's left-right sense;
  3. sample x/y/z on a regular uv grid of spacing <step> by linear (barycentric) interpolation over the input grid's
     own triangles (two per valid quad). Triangles whose 2D orientation disagrees with the majority (fold-overs) are
     dropped and counted; points outside every kept triangle are invalid (-1, the tifxyz convention);
  4. write x/y/z.tif (float32), meta.json (input meta, scale 1/step, bbox, source note) and iso_info.json, and print
     the distortion |edge3D/step - 1| (median, p95 over row and column edges between valid nodes) of the input
     (against its own meta scale) and of the output. This is the distortion measure of a First Letters package's
     isometric 2D flattening.
"""
import os, json, argparse
import numpy as np
import tifffile
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix


def load(d):
    """tifxyz dir -> (H, W, 3) float64; invalid nodes have z <= 0 (tifxyz writes -1)."""
    return np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)


def distortion(G, step):
    """|edge3D/step - 1| over row and column edges between valid nodes: (median, p95, number of edges)."""
    m = G[..., 2] > 0
    e = []
    for ax in (0, 1):
        d = np.linalg.norm(np.diff(G, axis=ax), axis=-1)
        mm = (m[1:] & m[:-1]) if ax == 0 else (m[:, 1:] & m[:, :-1])
        e.append(d[mm])
    e = np.abs(np.concatenate(e) / step - 1)
    return float(np.median(e)), float(np.percentile(e, 95)), len(e)


def flatten(G, iters=200):
    H, W = G.shape[:2]
    m = G[..., 2] > 0
    idx = -np.ones((H, W), int); idx[m] = np.arange(m.sum()); n = int(m.sum())
    P = G[m]
    edges = []
    for (dr, dc) in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = max(0, -dr), H - max(0, dr); c0, c1 = max(0, -dc), W - max(0, dc)
        a = idx[r0:r1, c0:c1]; b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        k = (a >= 0) & (b >= 0); edges.append(np.stack([a[k], b[k]], 1))
    E = np.concatenate(edges); L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    # init: per-row arc length from the middle column, per-column arc length from the middle row
    cm, rm = W // 2, H // 2
    dcol = np.linalg.norm(np.diff(G, axis=1), axis=-1) * (m[:, 1:] & m[:, :-1])
    drow = np.linalg.norm(np.diff(G, axis=0), axis=-1) * (m[1:] & m[:-1])
    cu = np.concatenate([np.zeros((H, 1)), np.cumsum(dcol, 1)], 1); u0 = cu - cu[:, cm:cm + 1]
    cv = np.concatenate([np.zeros((1, W)), np.cumsum(drow, 0)], 0); v0 = cv - cv[rm:rm + 1, :]
    x0 = np.stack([u0[m], v0[m]], 1).ravel()
    anc = np.array([idx[m][0], idx[m][-1]]); A = x0.reshape(-1, 2)[anc].copy(); wa = 1e-3
    ne = len(E)

    def fun(x):
        U = x.reshape(-1, 2); d = U[E[:, 0]] - U[E[:, 1]]
        r = np.linalg.norm(d, axis=1) - L
        return np.concatenate([r, wa * (U[anc] - A).ravel()])

    def jac(x):
        U = x.reshape(-1, 2); d = U[E[:, 0]] - U[E[:, 1]]; nd = np.linalg.norm(d, axis=1) + 1e-12; g = d / nd[:, None]
        rows = np.repeat(np.arange(ne), 4)
        cols = np.stack([2 * E[:, 0], 2 * E[:, 0] + 1, 2 * E[:, 1], 2 * E[:, 1] + 1], 1).ravel()
        vals = np.stack([g[:, 0], g[:, 1], -g[:, 0], -g[:, 1]], 1).ravel()
        ar = ne + np.arange(4); ac = np.array([2 * anc[0], 2 * anc[0] + 1, 2 * anc[1], 2 * anc[1] + 1])
        return csr_matrix((np.concatenate([vals, np.full(4, wa)]), (np.concatenate([rows, ar]), np.concatenate([cols, ac]))),
                          shape=(ne + 4, 2 * n))
    res = least_squares(fun, x0, jac=jac, method='trf', tr_solver='lsmr', max_nfev=iters, x_scale=1.0)
    U = res.x.reshape(-1, 2)
    rel = np.abs(np.linalg.norm(U[E[:, 0]] - U[E[:, 1]], axis=1) / L - 1)
    info = dict(nodes=n, edges=ne, nfev=int(res.nfev), status=int(res.status),
                edge_rel_err_median=float(np.median(rel)), edge_rel_err_p95=float(np.percentile(rel, 95)))
    return U, m, idx, info


def regrid(G, step=20.0, iters=200):
    """(H, W, 3) surface -> (out (h, w, 3) float32 with invalid = 0, info dict)."""
    U, m, idx, info = flatten(G, iters)
    P = G[m]
    # rotate so z increases along +v (input rows follow z the same way)
    A = np.c_[U, np.ones(len(U))]; cz = np.linalg.lstsq(A, P[:, 2], rcond=None)[0][:2]
    ang = np.arctan2(cz[0], cz[1]); c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]]); U = U @ R.T
    sign_in = np.sign(np.nanmedian(np.diff(np.where(m, G[..., 2], np.nan), axis=0)))
    if sign_in < 0:
        U[:, 1] = -U[:, 1]
    # keep the input's left-right sense: u should increase with the input column index
    cols = np.argwhere(m)[:, 1]
    if np.corrcoef(cols, U[:, 0])[0, 1] < 0:
        U[:, 0] = -U[:, 0]
    # triangles of the input grid (two per valid quad)
    q = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    k = (q[0] >= 0) & (q[1] >= 0) & (q[2] >= 0) & (q[3] >= 0)
    tri = np.concatenate([np.stack([q[0][k], q[1][k], q[3][k]], 1), np.stack([q[0][k], q[3][k], q[2][k]], 1)])
    # orientation: keep triangles with the majority sign (fold-overs are dropped and counted)
    a_, b_, c_ = U[tri[:, 0]], U[tri[:, 1]], U[tri[:, 2]]
    sa = (b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1]) - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0])
    maj = np.sign(np.median(sa)); keep = np.sign(sa) == maj
    info['flipped_triangles'] = int((~keep).sum()); info['triangles'] = int(len(tri))
    tri = tri[keep]
    u_min, v_min = U.min(0); u_max, v_max = U.max(0)
    gu = np.arange(u_min, u_max + 1e-6, step); gv = np.arange(v_min, v_max + 1e-6, step)
    out = np.zeros((len(gv), len(gu), 3), np.float32)
    for t3 in tri:  # rasterise each triangle by barycentric coordinates
        qq = U[t3]
        lo = np.floor((qq.min(0) - (u_min, v_min)) / step).astype(int)
        hi = np.ceil((qq.max(0) - (u_min, v_min)) / step).astype(int)
        ii, jj = np.meshgrid(np.arange(max(lo[1], 0), min(hi[1], len(gv) - 1) + 1),
                             np.arange(max(lo[0], 0), min(hi[0], len(gu) - 1) + 1), indexing='ij')
        if ii.size == 0:
            continue
        pu = gu[jj.ravel()]; pv = gv[ii.ravel()]
        T = np.array([[qq[1, 0] - qq[0, 0], qq[2, 0] - qq[0, 0]], [qq[1, 1] - qq[0, 1], qq[2, 1] - qq[0, 1]]])
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        if abs(det) < 1e-9:
            continue
        du = pu - qq[0, 0]; dv = pv - qq[0, 1]
        l1 = (T[1, 1] * du - T[0, 1] * dv) / det; l2 = (-T[1, 0] * du + T[0, 0] * dv) / det; l0 = 1 - l1 - l2
        ins = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        if not ins.any():
            continue
        X3 = P[t3]
        val = l0[ins, None] * X3[0] + l1[ins, None] * X3[1] + l2[ins, None] * X3[2]
        out[ii.ravel()[ins], jj.ravel()[ins]] = val
    bad = np.any(out == 0, -1); out[bad] = 0
    info['out_shape'] = list(out.shape[:2]); info['out_valid'] = float((~bad).mean())
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--step', type=float, default=20.0, help='output grid edge length in voxels (default 20)')
    ap.add_argument('--iters', type=int, default=200, help='least-squares function evaluations (default 200)')
    a = ap.parse_args()
    G = load(a.inp); meta = json.load(open(f'{a.inp}/meta.json'))
    out, info = regrid(G, a.step, a.iters)
    bad = ~(out[..., 2] > 0)
    os.makedirs(a.out, exist_ok=True)
    w = out.copy(); w[bad] = -1
    for ch, c_ in enumerate('xyz'):
        tifffile.imwrite(f'{a.out}/{c_}.tif', w[..., ch])
    v = out[~bad]
    meta2 = dict(meta); meta2['scale'] = [1 / a.step, 1 / a.step]
    meta2['bbox'] = [v.min(0).tolist(), v.max(0).tolist()]
    meta2['uuid'] = str(meta.get('uuid', 'mesh')) + '_iso'
    meta2['source'] = f"iso_regrid.py of {os.path.basename(os.path.normpath(a.inp))} (mass-spring flatten, step {a.step})"
    json.dump(meta2, open(f'{a.out}/meta.json', 'w'), indent=1)
    dmed, dp95, ne = distortion(out.astype(np.float64), a.step)
    din = distortion(G, 1 / meta['scale'][0])
    info.update(in_step=1 / meta['scale'][0], in_distortion=din[:2], out_step=a.step,
                out_distortion_median=dmed, out_distortion_p95=dp95, out_edges=ne)
    json.dump(info, open(f'{a.out}/iso_info.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
