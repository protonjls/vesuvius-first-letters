#!/usr/bin/env python3
"""Small helpers for the follow-up kit shell scripts.

  kit_util.py render2npy <render.zarr> <out.npy> <ct_ds2.png>
      VC3D render (OME-Zarr, group '0' = (layers, H, W) uint8) -> .npy for ink-sdk, plus a 2x-downsampled preview of the
      4 central layers.
  kit_util.py prob2png <ink-sdk out dir> <prefix> [<ds2 prefix>]
      ink-sdk probability.npy / probability_reverse.npy -> <prefix>_fwd.png / <prefix>_rev.png (uint8, p*255, full res)
      and optionally 2x-downsampled copies <ds2 prefix>_{fwd,rev}_ds2.png.
  kit_util.py ctd0 <variants dir> <out.png>
      mean of the 4 central layers of <variants dir>/d0.npy (the confirm montage's CT panel and valid mask).
  kit_util.py nearest <tifxyz dir> <x> <y> <z>
      grid node (row col) of the mesh nearest to a 3-D point (volume voxels, x y z order) and its distance.
"""
import sys

import numpy as np


def render2npy(zarr_path, out_npy, ct_png):
    import zarr
    from PIL import Image
    a = np.asarray(zarr.open(zarr_path, mode='r')['0'])
    np.save(out_npy, a)
    c = a.shape[0] // 2
    ct = a[c - 2:c + 2].astype(np.float32).mean(0)
    H, W = (ct.shape[0] // 2) * 2, (ct.shape[1] // 2) * 2
    d = ct[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    Image.fromarray(np.clip(np.rint(d), 0, 255).astype(np.uint8)).save(ct_png)
    print('render', a.shape, a.dtype)


def prob2png(odir, prefix, ds2_prefix=None):
    from PIL import Image
    for m, suf in (('probability', 'fwd'), ('probability_reverse', 'rev')):
        a = np.nan_to_num(np.load(f'{odir}/{m}.npy').astype(np.float32))
        Image.fromarray(np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)).save(f'{prefix}_{suf}.png')
        if ds2_prefix:
            H, W = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
            d = a[:H, :W].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
            Image.fromarray(np.clip(np.rint(d * 255), 0, 255).astype(np.uint8)).save(f'{ds2_prefix}_{suf}_ds2.png')


def ctd0(vdir, out_png):
    from PIL import Image
    v = np.load(f'{vdir}/d0.npy')
    c = v[v.shape[0] // 2 - 2: v.shape[0] // 2 + 2].astype(np.float32).mean(0)
    Image.fromarray(np.clip(np.rint(c), 0, 255).astype(np.uint8)).save(out_png)


def nearest(mesh, x, y, z):
    import tifffile
    P = np.stack([tifffile.imread(f'{mesh}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    ok = P[..., 0] > 0
    d = np.linalg.norm(P - np.array([x, y, z], dtype=np.float64), axis=-1)
    d[~ok] = np.inf
    r, c = np.unravel_index(np.argmin(d), d.shape)
    print(r, c, f'{d[r, c]:.1f}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'render2npy':
        render2npy(*args)
    elif cmd == 'prob2png':
        prob2png(*args)
    elif cmd == 'ctd0':
        ctd0(*args)
    elif cmd == 'nearest':
        nearest(args[0], *map(float, args[1:4]))
    else:
        sys.exit(f'unknown command {cmd}')
