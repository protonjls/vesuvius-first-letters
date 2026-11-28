#!/usr/bin/env python3
"""squash.py <in tifxyz dir> <out tifxyz dir> [--amp 0.6] [--period 6.0]

Synthetic angle-z-like column squash of a tifxyz surface, for testing iso_regrid.py on a surface with a known answer.
The columns are re-sampled at new positions p_0 = 0, p_{j+1} = p_j + 1 + amp * sin(j / period) (in input-column units),
by linear interpolation between neighbouring input columns; rows are unchanged. With amp 0.6 the column edges vary
smoothly from about 0.4x to 1.6x of the input step, like the column edges of a spiral-fit winding on its (angle, z)
grid. The surface itself is unchanged (every output node lies on a segment between two input nodes), and meta.json
keeps the input scale, so the distortion measure of the output is large while its geometry is the input's.
A node is invalid (-1) when either input column it interpolates is invalid.
"""
import os, json, argparse
import numpy as np
import tifffile


def squash(X, amp=0.6, period=6.0):
    """(H, W, 3) -> (H, W', 3) column-resampled surface (invalid = -1) and the column positions used."""
    H, W = X.shape[:2]
    n = int(3 * W / max(1 - amp, 0.05)) + 2
    sp = 1.0 + amp * np.sin(np.arange(n) / period)
    pos = np.concatenate([[0.0], np.cumsum(sp)]); pos = pos[pos <= W - 1]
    j0 = np.floor(pos).astype(int); j1 = np.minimum(j0 + 1, W - 1); f = (pos - j0)[None, :, None]
    Q = X[:, j0] * (1 - f) + X[:, j1] * f
    bad = (X[:, j0, 2] <= 0) | ((X[:, j1, 2] <= 0) & (f[..., 0] > 0))
    Q[bad] = -1
    return Q, pos


def main():
    ap = argparse.ArgumentParser(description='synthetic column squash of a tifxyz surface')
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--amp', type=float, default=0.6, help='column spacing amplitude (default 0.6: edges 0.4x-1.6x)')
    ap.add_argument('--period', type=float, default=6.0, help='spacing period in columns / (2 pi) (default 6)')
    a = ap.parse_args()
    X = np.stack([tifffile.imread(f'{a.inp}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    meta = json.load(open(f'{a.inp}/meta.json'))
    Q, pos = squash(X, a.amp, a.period)
    os.makedirs(a.out, exist_ok=True)
    for ch, c in enumerate('xyz'):
        tifffile.imwrite(f'{a.out}/{c}.tif', Q[..., ch].astype(np.float32))
    meta['uuid'] = str(meta.get('uuid', 'mesh')) + '_squash'
    meta['source'] = (f"squash.py of {os.path.basename(os.path.normpath(a.inp))} (columns re-sampled with spacing "
                      f"1 + {a.amp} sin(j / {a.period}); rows unchanged; scale kept)")
    json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
    print(json.dumps(dict(in_shape=list(X.shape[:2]), out_shape=list(Q.shape[:2]),
                          spacing_min=float(np.diff(pos).min()), spacing_max=float(np.diff(pos).max()))))


if __name__ == '__main__':
    main()
