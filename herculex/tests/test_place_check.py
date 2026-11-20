#!/usr/bin/env python3
"""Synthetic known answers for place_check.py: a plane grid with 20-voxel node spacing and a copy moved 3 voxels along
its normal and 10 voxels in-plane. Normal distance must read 3, nearest-node distance must read more than 5
(sqrt(3^2 + 10^2) ~ 10.4), and a copy 12 voxels away (next sheet) must fall outside the 5-voxel bar.
Run: python3 herculex/tests/test_place_check.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import place_check as pc  # noqa: E402


def plane(dz=0.0, dx=0.0, n=30, step=20.0):
    r, c = np.mgrid[0:n, 0:n].astype(np.float64)
    return np.stack([1000 + c * step + dx, 2000 + r * step, 5000 + 0 * r + dz], -1)


def test_same_sheet():
    A = plane(); B = plane(dz=3.0, dx=10.0)
    P = A[5:25, 5:25].reshape(-1, 3)
    s, d, _ = pc.normal_dn(B, P)
    assert abs(np.median(s) - 3.0) < 0.1, np.median(s)
    assert np.median(d) > 5.0, np.median(d)


def test_next_sheet():
    A = plane(); B = plane(dz=12.0, dx=10.0)
    s, d, _ = pc.normal_dn(B, A[5:25, 5:25].reshape(-1, 3))
    assert np.median(s) > 5.0


def test_tilted():
    # plane tilted 30 degrees about the row axis: normal distance is still the offset along the normal
    A = plane(); t = np.deg2rad(30); nrm = np.array([-np.sin(t), 0, np.cos(t)])
    rot = lambda G: np.stack([1000 + (G[..., 0] - 1000) * np.cos(t), G[..., 1], 5000 + (G[..., 0] - 1000) * np.sin(t)], -1)
    RA = rot(A); RB = rot(plane(dx=7.0)) + 2.5 * nrm
    s, _, _ = pc.normal_dn(RB, RA[5:25, 5:25].reshape(-1, 3))
    assert abs(np.median(s) - 2.5) < 0.1, np.median(s)


if __name__ == '__main__':
    for f in (test_same_sheet, test_next_sheet, test_tilted):
        f(); print('ok', f.__name__)
    print('3/3 passed')
