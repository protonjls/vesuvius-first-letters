#!/usr/bin/env python3
"""Persistence statistics and montage for a confirm_region.sh result dir.

  confirm_stats.py --dir RES/<tag> [--order rev]

Reads <dir>/maps/<variant>/<model>_{fwd,rev}.png and <dir>/ct_d0.png. Valid pixels: ct_d0 > 0, eroded 16 px.
Per model and order, for each variant: r_d0 = Pearson r with the same model's d0 map (smoothed sigma 4 px), and
contrast = p99 - p50 of the map. Real surface ink: r_d0 high at d+-1/d+-2, low for off+-k and shuf; a model artefact
that ignores depth would keep r_d0 high under shuf. Cross-model r at d0 (independent models agree on real ink).
Writes <dir>/stats.json and <dir>/montage_<order>.png (rows = variants, cols = models, then a product column if two+ models).
"""
import argparse
import itertools
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ORDER_VARS = ['d-2', 'd-1', 'd0', 'd+1', 'd+2', 'off-15', 'off+15', 'shuf']


def load(p):
    return np.asarray(Image.open(p)).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--order', default='rev')
    ap.add_argument('--tile', type=int, default=360)
    a = ap.parse_args()
    D = a.dir
    ct = np.asarray(Image.open(os.path.join(D, 'ct_d0.png')))
    valid = ndi.binary_erosion(ct > 0, iterations=16)
    variants = [v for v in ORDER_VARS if os.path.isdir(os.path.join(D, 'maps', v))]
    variants += sorted(v for v in os.listdir(os.path.join(D, 'maps')) if v not in variants)
    models = sorted({f.rsplit('_', 1)[0] for v in variants for f in os.listdir(os.path.join(D, 'maps', v))})
    maps = {}
    for v in variants:
        for m in models:
            for o in ('fwd', 'rev'):
                p = os.path.join(D, 'maps', v, f'{m}_{o}.png')
                if os.path.exists(p):
                    maps[(v, m, o)] = load(p)
    sm = {k: ndi.gaussian_filter(x, 4) for k, x in maps.items()}
    stats = {'valid_px': int(valid.sum()), 'models': models, 'variants': variants, 'per_map': {}, 'cross_model_d0': {}}
    for (v, m, o), x in maps.items():
        e = {'contrast': float(np.percentile(x[valid], 99) - np.percentile(x[valid], 50)), 'mean': float(x[valid].mean())}
        if ('d0', m, o) in sm:
            e['r_d0'] = float(np.corrcoef(sm[(v, m, o)][valid], sm[('d0', m, o)][valid])[0, 1])
        stats['per_map'][f'{v}/{m}_{o}'] = e
    for o in ('fwd', 'rev'):
        for m1, m2 in itertools.combinations(models, 2):
            if ('d0', m1, o) in sm and ('d0', m2, o) in sm:
                stats['cross_model_d0'][f'{m1}~{m2}_{o}'] = float(
                    np.corrcoef(sm[('d0', m1, o)][valid], sm[('d0', m2, o)][valid])[0, 1])
    json.dump(stats, open(os.path.join(D, 'stats.json'), 'w'), indent=1)
    # table
    o = a.order
    print(f'order {o}: r_d0 / contrast per variant')
    print('variant  ' + ''.join(f'{m[:14]:>22s}' for m in models))
    for v in variants:
        row = []
        for m in models:
            e = stats['per_map'].get(f'{v}/{m}_{o}')
            row.append(f"{e.get('r_d0', float('nan')):>12.3f} {e['contrast']:>8.3f}" if e else ' ' * 22)
        print(f'{v:8s} ' + ''.join(row))
    for k, r in stats['cross_model_d0'].items():
        print(f'cross-model d0 {k}: r {r:.3f}')
    # montage
    t = a.tile
    cols = models + (['product'] if len(models) > 1 else [])
    sheet = Image.new('L', (len(cols) * (t + 4) + 70, len(variants) * (t + 4) + 16), 30)
    dr = ImageDraw.Draw(sheet)
    for j, m in enumerate(cols):
        dr.text((70 + j * (t + 4) + 2, 2), m[:20], fill=255)
    for i, v in enumerate(variants):
        dr.text((2, 16 + i * (t + 4) + t // 2), v, fill=255)
        prod = None
        for j, m in enumerate(cols):
            if m == 'product':
                x = prod
            else:
                x = maps.get((v, m, o))
                if x is not None:
                    prod = x if prod is None else prod * x
            if x is None:
                continue
            if m == 'product':
                x = x / max(np.percentile(x[valid], 99.5), 1e-6)
            im = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8)).resize((t, t), Image.BILINEAR)
            sheet.paste(im, (70 + j * (t + 4), 16 + i * (t + 4)))
    sheet.save(os.path.join(D, f'montage_{o}.png'))


if __name__ == '__main__':
    main()
