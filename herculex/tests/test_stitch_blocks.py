#!/usr/bin/env python3
"""Synthetic known answers for stitch_blocks.py. A curved sheet grid (16-voxel node spacing, invalid border) is cut into
2 x 3 overlapping blocks the way HerculeX cuts a source surface (block offsets, shared border nodes). Stitching the
blocks must give back the source grid exactly, with zero overlap disagreement; blocks of different sources must be
refused; a block moved by 0.5 voxels must be reported with an overlap disagreement of 0.5. The CLI round trip writes
a tifxyz that reloads to the same grid.
Run: python3 herculex/tests/test_stitch_blocks.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
import tifffile
KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, KIT)
import stitch_blocks as sb  # noqa: E402


def source_grid(h=200, w=300, step=16.0):
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = c * step / 2000.0                       # a cylinder of radius 2000 voxels
    G = np.stack([3000 + 2000 * np.sin(ang), 4000 + r * step, 5000 + 2000 * np.cos(ang)], -1).astype(np.float32)
    G[:3] = -1; G[:, -2:] = -1                     # invalid border nodes, as in HerculeX pieces
    return G


def cut(G, source=24):
    pieces = []
    k = 0
    for r0 in (0, 96):
        for c0 in (0, 112, 224):
            blk = G[r0:r0 + 128, c0:c0 + 128].copy()
            m = {'source': source, 'stride': 16, 'scale': [0.0625, 0.0625], 'block': [r0, c0], 'piece': source + k * 1000000}
            pieces.append((f'p{k}', m, blk)); k += 1
    return pieces


def test_reassemble_exact():
    G = source_grid()
    C, rep = sb.stitch(cut(G))
    H, W = rep['shape']
    assert (H, W) == (200, 300), rep['shape']
    assert np.array_equal(C, G[:H, :W]), np.abs(C - G[:H, :W]).max()
    assert rep['overlap_max_vox'] == 0.0, rep['overlap_max_vox']
    assert sum(p['overlap_nodes'] > 0 for p in rep['pieces']) == 5


def test_mixed_source_refused():
    p = cut(source_grid())
    p[3][1]['source'] = 25
    try:
        sb.stitch(p)
    except ValueError as e:
        assert 'source' in str(e)
    else:
        raise AssertionError('mixed sources were not refused')


def test_perturbed_block_reported():
    p = cut(source_grid())
    blk = p[4][2]; v = blk[..., 2] > 0
    blk[..., 0][v] += 0.5                          # block 4 moved 0.5 voxels in x
    _, rep = sb.stitch(p)
    r4 = rep['pieces'][4]
    assert abs(r4['overlap_max_vox'] - 0.5) < 1e-4 and abs(r4['overlap_median_vox'] - 0.5) < 1e-4, r4


def test_cli_roundtrip():
    G = source_grid()
    with tempfile.TemporaryDirectory() as t:
        dirs = []
        for name, m, blk in cut(G):
            d = os.path.join(t, name); os.makedirs(d); dirs.append(d)
            for i, c in enumerate('xyz'):
                tifffile.imwrite(os.path.join(d, f'{c}.tif'), blk[..., i])
            json.dump(m, open(os.path.join(d, 'meta.json'), 'w'))
        out = os.path.join(t, 'stitched')
        subprocess.run([sys.executable, os.path.join(KIT, 'stitch_blocks.py'), '--out', out] + dirs, check=True,
                       stdout=subprocess.DEVNULL)
        C = np.stack([tifffile.imread(os.path.join(out, f'{c}.tif')) for c in 'xyz'], -1)
        m = json.load(open(os.path.join(out, 'meta.json')))
        assert np.array_equal(C, G)
        assert m['uuid'] == 'stitch-src24' and len(m['stitched_from']) == 6 and m['block'] == [0, 0]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t(); print('ok', t.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
