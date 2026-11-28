#!/usr/bin/env python3
"""Synthetic tests for isoregrid/ (CPU, seconds):
  1. a curved sheet (a developable spiral-like curl, isometric to a plane) sampled with irregular columns (edges
     0.4x-1.6x of the step) -> iso_regrid.py (CLI) gives a low-distortion grid that lies on the true surface;
  2. the same sheet sampled isometrically stays isometric and on the surface (no-op);
  3. squash.py keeps the surface and makes the distortion large;
  4. normal_dn.py recovers a known offset between two parallel curved sheets, and is not fooled by node spacing.
  python3 isoregrid/tests/test_iso_regrid.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.dirname(HERE)
sys.path.insert(0, KIT)
import iso_regrid as ir  # noqa: E402
import normal_dn as nd  # noqa: E402
import squash as sq  # noqa: E402

R0, K = 300.0, 0.0008   # curl: radius of curvature R(s) = R0 / (1 + K s), arc length s along the sheet


def curl(s, z):
    """Developable sheet: a planar curve of arc length s (curvature growing with s), extruded along z (voxels)."""
    ss = np.linspace(0, s.max() + 1, 20001); kap = (1 + K * ss) / R0
    th = np.concatenate([[0], np.cumsum((kap[1:] + kap[:-1]) / 2 * np.diff(ss))])
    cx = np.concatenate([[0], np.cumsum((np.cos(th[1:]) + np.cos(th[:-1])) / 2 * np.diff(ss))])
    cy = np.concatenate([[0], np.cumsum((np.sin(th[1:]) + np.sin(th[:-1])) / 2 * np.diff(ss))])
    return np.stack([2000 + np.interp(s, ss, cx), 2000 + np.interp(s, ss, cy), 5000 + z], -1), (ss, cx + 2000, cy + 2000)


def dist_to_curl(P, curve):
    """Distance of points to the true sheet (nearest point of the dense planar curve; the sheet is extruded along z)."""
    from scipy.spatial import cKDTree
    ss, cx, cy = curve
    d, _ = cKDTree(np.c_[cx, cy]).query(P[:, :2])
    return d


def write(d, G, scale=0.05):
    os.makedirs(d, exist_ok=True)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(f'{d}/{c}.tif', G[..., i].astype(np.float32))
    json.dump({'format': 'tifxyz', 'type': 'seg', 'scale': [scale, scale], 'uuid': 'synthetic'}, open(f'{d}/meta.json', 'w'))


def sheet(cols_s, H=40, step=20.0):
    z = np.arange(H) * step
    S, Z = np.meshgrid(cols_s, z)
    G, curve = curl(S, Z)
    G[:3, :5] = -1          # a few invalid nodes, as on real meshes
    return G, curve


def test_irregular_columns(tmp):
    j = np.arange(70)
    cols = np.concatenate([[0], np.cumsum(20.0 * (1 + 0.6 * np.sin(j[:-1] / 5.0)))])
    G, curve = sheet(cols)
    din = ir.distortion(G, 20.0)
    assert din[1] >= 0.5, din
    write(f'{tmp}/irr', G)
    r = subprocess.run([sys.executable, f'{KIT}/iso_regrid.py', f'{tmp}/irr', f'{tmp}/irr_iso', '--step', '20'],
                       check=True, capture_output=True, text=True)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    O = ir.load(f'{tmp}/irr_iso')
    assert tifffile.imread(f'{tmp}/irr_iso/z.tif').min() == -1      # invalid nodes written as -1
    om, op, _ = ir.distortion(O, 20.0)
    assert abs(om - info['out_distortion_median']) < 1e-6
    assert om < 0.02 and op < 0.06, (om, op)
    P = O[O[..., 2] > 0]
    dd = dist_to_curl(P, curve)
    assert np.median(dd) < 0.5 and np.percentile(dd, 95) < 1.0, (np.median(dd), np.percentile(dd, 95))
    # output spans about the same sheet area as the input (arc length x height)
    assert info['out_valid'] > 0.8 and abs(O.shape[1] * 20 - cols[-1]) < 60, (info, O.shape, cols[-1])
    # rows still follow z, columns still follow arc length
    zc = O[..., 2].copy(); zc[zc <= 0] = np.nan
    assert np.nanmedian(np.diff(zc, axis=0)) > 15
    print(f'irregular columns ok: input {din[0]:.3f}/{din[1]:.3f} -> output {om:.4f}/{op:.4f}; '
          f'distance to true sheet median {np.median(dd):.3f}')


def test_noop(tmp):
    cols = np.arange(60) * 20.0
    G, curve = sheet(cols)
    din = ir.distortion(G, 20.0)
    out, info = ir.regrid(G, 20.0)
    om, op, _ = ir.distortion(out.astype(np.float64), 20.0)
    assert om < 0.02 and op < 0.06, (om, op)
    P = out[out[..., 2] > 0].astype(np.float64)
    s, _, _ = nd.dn(G, P)
    dd = dist_to_curl(P, curve)
    assert np.median(s) < 0.5 and np.median(dd) < 0.5, (np.median(s), np.median(dd))
    print(f'no-op ok: input {din[0]:.4f}/{din[1]:.4f} -> output {om:.4f}/{op:.4f}; dn to input median {np.median(s):.3f}')


def test_squash(tmp):
    cols = np.arange(60) * 20.0
    G, curve = sheet(cols)
    Q, pos = sq.squash(G, 0.6, 6.0)
    e = np.diff(pos)
    assert 0.39 < e.min() < 0.45 and 1.55 < e.max() < 1.61, (e.min(), e.max())
    dq = ir.distortion(Q, 20.0)
    assert dq[1] >= 0.5, dq
    P = Q[Q[..., 2] > 0]
    dd = dist_to_curl(P, curve)
    assert np.median(dd) < 0.2, np.median(dd)
    assert Q.shape[0] == G.shape[0] and np.allclose(Q[3:, 0], G[3:, 0])   # rows unchanged; column 0 kept
    assert np.all(Q[:3, 0, 2] == -1)                                       # invalid input nodes stay invalid
    print(f'squash ok: distortion {dq[0]:.3f}/{dq[1]:.3f}, distance to true sheet median {np.median(dd):.3f}')


def test_normal_dn(tmp):
    cols = np.arange(60) * 20.0
    G, curve = sheet(cols)
    N = nd.normals(G)
    G2 = G.copy(); ok = G[..., 2] > 0
    G2[ok] = G[ok] + 3.0 * N[ok]                      # parallel sheet 3 voxels away along the normal
    s0, _, _ = nd.dn(G, G2[ok])                       # at the nodes: exact
    assert abs(np.median(s0) - 3.0) < 0.05, np.median(s0)
    # half-node shift along the sheet (chord midpoints): the nearest-node distance is about 10 voxels, the normal
    # distance stays near 3; its curvature bias (tangent plane 10 voxels from the touching point, radius 140-300
    # voxels, plus the chord sagitta) is a few tenths of a voxel and documented in the README
    P = 0.5 * (G2[5:, 5:-1] + G2[5:, 6:])
    s, d, _ = nd.dn(G, P.reshape(-1, 3))
    assert abs(np.median(s) - 3.0) < 0.6, np.median(s)
    assert np.median(d) > 9, np.median(d)
    print(f'normal_dn ok: known offset 3.00 -> {np.median(s0):.3f} at nodes, {np.median(s):.3f} half a node away '
          f'(nearest-node distance {np.median(d):.1f})')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as t:
        test_irregular_columns(t); test_noop(t); test_squash(t); test_normal_dn(t)
    print('all isoregrid tests passed')
