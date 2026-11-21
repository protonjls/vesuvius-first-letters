#!/usr/bin/env python3
"""Synthetic known answers for reappear.py. One flat sheet carries an ink pattern defined in voxel coordinates.
Mesh A samples it on a 16-voxel grid (HerculeX-like), mesh B on a 20-voxel grid with another origin, 2 voxels off
along the normal (GrowPatch-like); B's maps are a window starting at render pixel (40, 60). Expected:
  same sheet, right offset           -> r > 0.95 in the (fwd, fwd) pair, unrelated maps in the other pairs
  --shift 12 (about one blob)        -> r < half
  wrong --offB                       -> r < half
  B on another sheet (30 voxels off) -> no valid pixels (no best pair)
Run: python3 crosscheck/tests/test_reappear.py"""
import os, sys, tempfile
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import reappear as ra  # noqa: E402
import tifffile  # noqa: E402

RNG = np.random.default_rng(7)
BLOBS = RNG.uniform([980, 1980], [1820, 2820], size=(400, 2))  # (x, y) voxel centres


def ink(X, Y, blobs=BLOBS, s=5.0):
    v = np.zeros_like(X, dtype=np.float64)
    for bx, by in blobs:
        v += np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * s * s))
    return np.clip(v / v.max() * 255, 0, 255).astype(np.uint8)


def make_mesh(d, step, ox, oy, z, n):
    r, c = np.mgrid[0:n, 0:n].astype(np.float32)
    os.makedirs(d, exist_ok=True)
    for k, a in zip('xyz', (ox + c * step, oy + r * step, z + 0 * r)):
        tifffile.imwrite(f'{d}/{k}.tif', a.astype(np.float32))
    open(f'{d}/meta.json', 'w').write('{"scale": [%r, %r], "format": "tifxyz"}' % (1 / step, 1 / step))


def make_maps(d, ox, oy, h, w, x0=0, y0=0, seed=0):
    os.makedirs(f'{d}/maps/d0', exist_ok=True)
    Y, X = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float64)
    Image.fromarray(np.full((h, w), 100, np.uint8)).save(f'{d}/ct_d0.png')
    Image.fromarray(ink(X + ox, Y + oy)).save(f'{d}/maps/d0/reader-v3_fwd.png')
    other = np.random.default_rng(seed).uniform([980, 1980], [1820, 2820], size=(400, 2))
    Image.fromarray(ink(X + ox, Y + oy, other)).save(f'{d}/maps/d0/reader-v3_rev.png')


def setup(tmp, dzB=2.0):
    make_mesh(f'{tmp}/A', 16, 1000, 2000, 5000, 50)
    make_maps(f'{tmp}/mA', 1000, 2000, 800, 800, seed=1)
    make_mesh(f'{tmp}/B', 20, 1000 - 13.0, 2000 + 9.0, 5000 + dzB, 45)
    make_maps(f'{tmp}/mB', 1000 - 13.0, 2000 + 9.0, 700, 760, x0=40, y0=60, seed=2)


def go(tmp, **kw):
    kw.setdefault('offB', (40, 60))
    return ra.run(f'{tmp}/A', f'{tmp}/mA', f'{tmp}/B', f'{tmp}/mB', f'{tmp}/out', (8, 40), (8, 40),
                  models=('reader-v3',), sheet=False, **kw)


def test_same_sheet():
    with tempfile.TemporaryDirectory() as t:
        setup(t); i = go(t)
        b = i['best']['reader-v3']
        assert b['r'] > 0.95 and (b['A'], b['B']) == ('fwd', 'fwd'), i['r']
        assert abs(i['r']['reader-v3 Afwd Brev']) < 0.3, i['r']
        assert abs(i['normal_dn_median'] - 2.0) < 0.1, i['normal_dn_median']


def test_shift_control():
    with tempfile.TemporaryDirectory() as t:
        setup(t); r0 = go(t)['r']['reader-v3 Afwd Bfwd']; r1 = go(t, shift=12)['r']['reader-v3 Afwd Bfwd']
        assert r1 < 0.5 * r0, (r0, r1)


def test_wrong_offset():
    with tempfile.TemporaryDirectory() as t:
        setup(t); r0 = go(t)['r']['reader-v3 Afwd Bfwd']; r1 = go(t, offB=(0, 0))['r']['reader-v3 Afwd Bfwd']
        assert r1 < 0.5 * r0, (r0, r1)


def test_other_sheet():
    with tempfile.TemporaryDirectory() as t:
        setup(t, dzB=30.0); i = go(t)
        assert i['valid_pixels'] == 0 and not i['best'], i


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); n += 1; print('ok', k)
    print(f'{n}/{n} passed')
