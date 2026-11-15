#!/usr/bin/env python3
"""Control variants of a rendered surface volume, for candidate confirmation (false-positive mitigation).

  variants.py --sv window.zarr --outdir DIR [--depth 24] [--offsets -2,-1,0,1,2] [--off 15] [--seed 0]

The input render has N layers centred on the mesh (layer (N-1)/2 = on the surface). Writes DIR/<name>.npy (D, H, W uint8):
  d<s>      : `depth` layers centred at the surface + s voxels (s in offsets): letters from real ink persist at s = +-1, +-2
  off+<k>, off-<k> : `depth` layers centred k voxels off the surface on either side: a surface-locked signal should vanish
  shuf      : the d0 layers in a fixed random depth permutation: a model reading depth structure should lose real ink
Prints a JSON manifest (names, centres, shapes).
"""
import argparse
import json
import os

import numpy as np


def load(path):
    if path.endswith('.npy'):
        return np.load(path)
    import zarr
    g = zarr.open(path, mode='r')
    a = g['0'] if hasattr(g, 'keys') and '0' in g else g
    return np.asarray(a[:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--depth', type=int, default=24)
    ap.add_argument('--offsets', default='-2,-1,0,1,2')
    ap.add_argument('--off', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    v = load(a.sv)
    N = v.shape[0]
    c = (N - 1) / 2.0
    os.makedirs(a.outdir, exist_ok=True)
    man = {'sv': a.sv, 'N': N, 'shape': list(v.shape), 'variants': {}}

    def cut(centre, name):
        s = int(round(centre - (a.depth - 1) / 2.0))
        if s < 0 or s + a.depth > N:
            man['variants'][name] = {'skipped': f'needs layers {s}..{s + a.depth - 1} of {N}'}
            return None
        x = np.ascontiguousarray(v[s:s + a.depth])
        np.save(os.path.join(a.outdir, name + '.npy'), x)
        man['variants'][name] = {'first_layer': s, 'centre_offset': centre - c, 'shape': list(x.shape)}
        return x

    for s in [int(t) for t in a.offsets.split(',')]:
        cut(c + s, f'd{s:+d}'.replace('+0', '0'))
    cut(c + a.off, f'off+{a.off}')
    cut(c - a.off, f'off-{a.off}')
    d0 = np.load(os.path.join(a.outdir, 'd0.npy'))
    perm = np.random.default_rng(a.seed).permutation(a.depth)
    np.save(os.path.join(a.outdir, 'shuf.npy'), np.ascontiguousarray(d0[perm]))
    man['variants']['shuf'] = {'perm': perm.tolist(), 'shape': list(d0.shape)}
    print(json.dumps(man))


if __name__ == '__main__':
    main()
