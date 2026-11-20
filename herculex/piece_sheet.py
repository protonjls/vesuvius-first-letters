#!/usr/bin/env python3
"""Review sheet and instrument check for one piece_confirm.sh result.

  piece_sheet.py <confirm out_dir> --out sheet.png [--ct-face piece-<n>.png] [--ink-face piece-<n>-taught.png]
                 [--stride 16] [--box x0,y0,x1,y1] [--ds 2]

Row 1: [HerculeX ink image, aligned | CT d0 | each model at d0] in the chosen depth order.
Row 2: the first model on the control variants d-2, d+2, off-15, off+15, shuf (same order and stretch).
HerculeX writes, per piece, a CT face image (<faces>/<unit>/piece-<n>.png) and an ink image (piece-<n>-taught.png).
Both are the scale-1 render padded by 2 grid cells on every side. The script crops the padding and tries the 4 flips:
  --ct-face : Pearson r (4x-downsampled, smoothed, valid pixels) against our CT d0 -> geometry check (same surface,
              same pixel grid); its best flip is used for the ink image too.
  --ink-face: r against the first model's d0 map in both orders -> instrument check (our render + model reproduce
              HerculeX's ink on the same surface); the order with the higher r is used for the sheet.
Without --ink-face the order with the larger first-model contrast (p99 - p50) is used.
--box (render pixels, e.g. place_check.py cand_render_box_xyxy) is drawn on every panel; cross-model r is also given
inside it. Prints a JSON report (flip, r tables, order, cross-model r, contrast per model).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
CTRL = ['d-2', 'd+2', 'off-15', 'off+15', 'shuf']


def load(p):
    return np.asarray(Image.open(p).convert('L')).astype(np.float32)


def small(a, k=4):
    a = ndi.gaussian_filter(a, 2.0)
    H, W = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))


def corr(a, b, m):
    a, b = a[m], b[m]
    if a.size < 10 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def stretch(a, m):
    v = a[m] if m.any() else a.ravel()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.7)
    return np.clip((a - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--out', required=True); ap.add_argument('--ct-face'); ap.add_argument('--ink-face')
    ap.add_argument('--stride', type=int, default=16); ap.add_argument('--box'); ap.add_argument('--ds', type=int, default=2)
    a = ap.parse_args()
    d0 = os.path.join(a.dir, 'maps', 'd0')
    models = sorted({f.rsplit('_', 1)[0] for f in os.listdir(d0) if f.endswith(('_fwd.png', '_rev.png'))})
    models.sort(key=lambda m: (m != 'reader-v3', m))
    ct = load(os.path.join(a.dir, 'ct_d0.png')); valid = ct > 0
    M = {(m, o): load(os.path.join(d0, f'{m}_{o}.png')) for m in models for o in ('fwd', 'rev')}
    H, W = ct.shape
    rep = {'models': models}
    face = None; flip = None; mg = 2 * a.stride
    sv = small(valid.astype(np.float32)) > 0.99

    def flips(F):
        return {'none': F, 'rows': F[::-1], 'cols': F[:, ::-1], 'both': F[::-1, ::-1]}

    def crop(G):
        g = np.zeros((H, W), np.float32); c = G[mg:mg + H, mg:mg + W]; g[:c.shape[0], :c.shape[1]] = c
        return g
    if a.ct_face:
        tab = {fn: round(corr(small(crop(G)), small(ct), sv), 3) for fn, G in flips(load(a.ct_face)).items()}
        flip = max(tab, key=lambda k: -1 if np.isnan(tab[k]) else tab[k])
        rep.update({'ct_face': os.path.basename(a.ct_face), 'ct_face_r': tab, 'flip': flip, 'ct_face_best_r': tab[flip]})
    if a.ink_face:
        tab = {}
        for fn, G in flips(load(a.ink_face)).items():
            if flip and fn != flip:
                continue
            for o in ('fwd', 'rev'):
                tab[f'{fn}/{o}'] = round(corr(small(crop(G)), small(M[(models[0], o)]), sv), 3)
        best = max(tab, key=lambda k: -1 if np.isnan(tab[k]) else tab[k])
        flip, order = best.split('/')
        face = crop(flips(load(a.ink_face))[flip])
        rep.update({'ink_face': os.path.basename(a.ink_face), 'ink_face_r': tab, 'ink_best': best, 'ink_best_r': tab[best]})
    else:
        con = {o: float(np.percentile(M[(models[0], o)][valid], 99) - np.percentile(M[(models[0], o)][valid], 50)) for o in ('fwd', 'rev')}
        order = max(con, key=con.get)
    rep['order'] = order
    rep['contrast_p99_p50'] = {m: round(float(np.percentile(M[(m, order)][valid], 99) - np.percentile(M[(m, order)][valid], 50)), 1) for m in models}
    bx = None
    if a.box:
        x0, y0, x1, y1 = map(int, a.box.split(',')); bx = (x0, y0, x1, y1)
    if len(models) > 1:
        rep['xmodel_r'] = {f'{models[0]}~{m}': round(corr(small(M[(models[0], order)]), small(M[(m, order)]), sv), 3) for m in models[1:]}
        if bx:
            bm = np.zeros_like(valid); bm[max(0, y0):y1, max(0, x0):x1] = True; sb = small((valid & bm).astype(np.float32)) > 0.99
            rep['xmodel_r_box'] = {f'{models[0]}~{m}': round(corr(small(M[(models[0], order)]), small(M[(m, order)]), sb), 3) for m in models[1:]}
    k = a.ds

    def panel(arr, title, is8=False):
        im = Image.fromarray(arr if is8 else stretch(arr, valid)).resize((W // k, H // k), Image.BILINEAR).convert('RGB')
        dr = ImageDraw.Draw(im)
        if bx:
            dr.rectangle([bx[0] // k, bx[1] // k, bx[2] // k, bx[3] // k], outline=(255, 0, 0), width=2)
        dr.rectangle([0, 0, 8 * len(title) + 6, 14], fill=(0, 0, 0)); dr.text((3, 2), title, fill=(255, 255, 0))
        return im
    row1 = []
    if face is not None:
        row1.append(panel(face, f'HerculeX ink ({flip})'))
    row1.append(panel(ct, 'CT d0'))
    row1 += [panel(M[(m, order)], f'{m} d0 {order}') for m in models]
    row2 = []
    for v in CTRL:
        p = os.path.join(a.dir, 'maps', v, f'{models[0]}_{order}.png')
        if os.path.exists(p):
            row2.append(panel(load(p), f'{models[0]} {v} {order}'))
    rows = [row1] + ([row2] if row2 else [])
    pw, ph = W // k, H // k; gap = 6
    ncol = max(len(r) for r in rows)
    S = Image.new('RGB', (ncol * (pw + gap), len(rows) * (ph + gap)), (25, 25, 25))
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            S.paste(im, (j * (pw + gap), i * (ph + gap)))
    S.save(a.out)
    rep['sheet'] = a.out
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
