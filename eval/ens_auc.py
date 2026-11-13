#!/usr/bin/env python3
"""Pixel AUC/AP of single ink maps and their ensembles vs labels (PHerc1447 known-text control).

  ens_auc.py --labels inklabels.png --mask mask.png --map v3=path.npy --map rect=... --map v8=... [--stride 2] [--out f.json]

Ensembles (over the masked, strided pixels): product a*b for each pair; z-mean of any subset listed with --z
(e.g. --z v3,v8 --z v3,rect,v8), where z = (map - mean) / sd over the scored pixels. Same table as
results/ensembles_1447.txt. Maps on another grid are resized bilinearly (label_auc.load rules).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_auc import auc_ap, load  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--mask', required=True)
    ap.add_argument('--map', action='append', required=True)
    ap.add_argument('--z', action='append', default=[])
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--out')
    a = ap.parse_args()
    L = load(a.labels) > 127
    K = load(a.mask) > 0
    s = a.stride
    maps = {}
    for spec in a.map:
        name, path = spec.split('=', 1)
        M = load(path)
        if M.shape != L.shape:
            M = np.asarray(Image.fromarray(M, mode='F').resize((L.shape[1], L.shape[0]), Image.BILINEAR))
        maps[name] = M[::s, ::s]
    sel = K[::s, ::s].copy()
    for M in maps.values():
        sel &= np.isfinite(M)
    y = L[::s, ::s][sel].astype(np.int64)
    v = {k: M[sel].astype(np.float64) for k, M in maps.items()}
    z = {k: (x - x.mean()) / (x.std() + 1e-9) for k, x in v.items()}
    res = {}

    def rep(name, sc):
        auc, apv = auc_ap(sc, y)
        res[name] = {'auc': auc, 'ap': apv}
        print(f'{name:30s} AUC {auc:.3f} AP {apv:.3f}')

    for k in v:
        rep(k, v[k])
    for a1, a2 in itertools.combinations(v, 2):
        rep(f'{a1}*{a2} product', v[a1] * v[a2])
    for zs in a.z:
        ks = zs.split(',')
        rep(f'z({"+".join(ks)})', sum(z[k] for k in ks) / len(ks))
    print(f'n {len(y)} pos {y.mean():.3f}')
    if a.out:
        json.dump({'n': int(len(y)), 'pos_frac': float(y.mean()), 'stride': s, 'maps': dict(a.map and [m.split('=', 1) for m in a.map]),
                   'scores': res}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
