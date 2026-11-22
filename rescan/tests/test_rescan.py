#!/usr/bin/env python3
"""Synthetic tests for rescan/: nodecorr.py recovers a known node correspondence between two meshes of one surface in
two volumes, and kat_compare.py moves a C-frame image into the team window.
  python3 rescan/tests/test_rescan.py"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.dirname(HERE)
sys.path.insert(0, KIT)
import kat_compare as kc  # noqa: E402


def surface(r, c):
    """A curved sheet in volume-B voxels; (r, c) in mesh-B node units (20 voxels per node)."""
    x = 20.0 * c; y = 20.0 * r
    z = 3000.0 + 40.0 * np.sin(c / 25.0) + 25.0 * np.cos(r / 40.0)
    return np.stack([x, y, z], -1)


def write_mesh(d, P, scale):
    import tifffile
    os.makedirs(d, exist_ok=True)
    for i, k in enumerate('xyz'):
        tifffile.imwrite(f'{d}/{k}.tif', P[..., i].astype(np.float32))
    json.dump({'format': 'tifxyz', 'type': 'seg', 'scale': [scale, scale]}, open(f'{d}/meta.json', 'w'))


def test_nodecorr(tmp):
    # mesh B: 240 x 260 nodes at scale 0.05 in volume B; mesh A: every 4th B node (offset 1, 2) expressed in volume A,
    # where volume A = 0.25 * volume B + t (the transform maps B (moving) -> A (fixed); nodecorr gets it with ':inv').
    rb, cb = np.mgrid[0:240, 0:260].astype(float)
    write_mesh(f'{tmp}/B', surface(rb, cb), 0.05)
    ra, ca = np.mgrid[0:58, 0:63].astype(float)
    PA_B = surface(4 * ra + 1, 4 * ca + 2)
    t = np.array([-12.0, 7.0, 5.0]); PA = 0.25 * PA_B + t
    write_mesh(f'{tmp}/A', PA, 0.05)
    M = [[0.25, 0, 0, t[0]], [0, 0.25, 0, t[1]], [0, 0, 0.25, t[2]]]
    json.dump({'transformation_matrix': M}, open(f'{tmp}/T.json', 'w'))
    subprocess.run([sys.executable, f'{KIT}/nodecorr.py', '--a', f'{tmp}/A', '--b', f'{tmp}/B', '--affine',
                    f'{tmp}/T.json:inv', '--out', f'{tmp}/corr.json', '--step', '1'], check=True, capture_output=True)
    c = json.load(open(f'{tmp}/corr.json'))
    W = np.array(c['fit_nodeB_from_nodeA (rows: [rowA,colA,1] -> [rowB,colB])'])
    exp = np.array([[4, 0], [0, 4], [1, 2]], float)
    assert np.abs(W - exp).max() < 0.05, W
    assert c['dist3d_median'] < 0.01, c['dist3d_median']
    print('nodecorr ok', np.round(W, 3).tolist())


def test_resample(tmp):
    rng = np.random.default_rng(0)
    import cv2
    F = cv2.GaussianBlur(rng.random((900, 1000)).astype(np.float32), (0, 0), 6)      # full team map
    W = [[3.95, 0.01], [-0.02, 3.90], [1.44, 2.08]]                                      # node fit A -> B
    json.dump({'fit_nodeB_from_nodeA (rows: [rowA,colA,1] -> [rowB,colB])': W}, open(f'{tmp}/corr2.json', 'w'))
    sA, sa, sb = 3.9, 0.05, 0.05
    A2 = np.array([[W[0][0] * sa / sA, W[1][0] * sa / sA, W[2][0]], [W[0][1] * sa / sA, W[1][1] * sa / sA, W[2][1]]]) / sb
    win = (300, 350, 250, 300)                                                           # R0 C0 H W in team px
    x0, y0 = 250, 230                                                                    # C crop origin
    pr, pc = np.mgrid[0:400, 0:450].astype(np.float64)
    tr = A2[0, 0] * (pr + y0) + A2[0, 1] * (pc + x0) + A2[0, 2]
    tc = A2[1, 0] * (pr + y0) + A2[1, 1] * (pc + x0) + A2[1, 2]
    C = cv2.remap(F, tc.astype(np.float32), tr.astype(np.float32), cv2.INTER_LINEAR)       # C-frame image of F
    got = kc.resample_into_team(C, A2, win, (x0, y0))
    R0, C0, H, Wd = win
    truth = F[R0:R0 + H, C0:C0 + Wd]
    inner = (slice(5, H - 5), slice(5, Wd - 5))
    rr = np.corrcoef(got[inner].ravel(), truth[inner].ravel())[0, 1]
    assert rr > 0.99, rr
    assert np.allclose(kc.px_c_to_team(f'{tmp}/corr2.json', sA, sa, sb), A2)
    print('resample ok r=%.4f' % rr)


def test_read_team(tmp):
    """read_team on a tiled, deflate-compressed uint8 TIFF like the published team maps."""
    import tifffile
    img = np.random.default_rng(1).integers(0, 256, (700, 900), dtype=np.uint8)
    tifffile.imwrite(f'{tmp}/team.tif', img, tile=(256, 256), compression='zlib')
    win = (300, 410, 250, 300)
    got = kc.read_team(f'{tmp}/team.tif', win)
    assert got.shape == (250, 300) and np.array_equal(got, img[300:550, 410:710].astype(np.float32) / 255.0)
    print('read_team ok')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        test_nodecorr(tmp)
        test_resample(tmp)
        test_read_team(tmp)
    print('all tests passed')
