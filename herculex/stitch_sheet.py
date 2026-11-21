#!/usr/bin/env python3
"""Review sheet for a confirm_region.sh window on a stitched HerculeX surface, with the block boundaries drawn.

  stitch_sheet.py <confirm out_dir> <stitched dir> --x0 X0 --y0 Y0 --out sheet.jpg [--box x0,y0,x1,y1] [--order rev|fwd|auto]
                  [--ds 2]

Row 1: CT d0 | each model at d0. Row 2: the first model on off+15 and shuf (whichever were run). Same depth order and,
per model, one contrast stretch for all its panels. Block boundaries come from <stitched dir>/stitch_report.json (piece
block offsets x the render stride, 1 / meta.json scale) and are drawn as dashed lines: a text line that runs across a
boundary shows that the stitched surface carries it from one HerculeX piece into the next. --box (render pixels of
the full stitched render, e.g. place_check.py cand_render_box_xyxy) is drawn as a red box. --x0/--y0 = the window
origin passed to confirm_region.sh. --order auto (default) uses stats.json: the order whose first-model off+15 r_d0
is lower (better separated from the sheet). Prints a JSON report: order, boundaries in window pixels, control r_d0
per model from stats.json.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def load(f):
    return np.asarray(Image.open(f).convert('L')).astype(np.float32)


def stretch(a, lo, hi):
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('stitched')
    ap.add_argument('--x0', type=int, required=True); ap.add_argument('--y0', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--box'); ap.add_argument('--order', default='auto')
    ap.add_argument('--ds', type=int, default=2)
    a = ap.parse_args()
    stats = json.load(open(os.path.join(a.dir, 'stats.json')))
    models = stats.get('models') or sorted({k.split('/')[1].rsplit('_', 1)[0] for k in stats['per_map']})
    pm = stats['per_map']
    order = a.order
    if order == 'auto':
        cand = [(pm.get(f'off+15/{models[0]}_{o}', {}).get('r_d0', 1.0), o) for o in ('rev', 'fwd')]
        order = min(cand)[1]
    rep = {'order': order, 'controls_r_d0': {}}
    for m in models:
        rep['controls_r_d0'][m] = {v: round(pm[f'{v}/{m}_{order}']['r_d0'], 3)
                                   for v in ('off+15', 'off-15', 'shuf', 'd-1', 'd+1', 'd-2', 'd+2')
                                   if f'{v}/{m}_{order}' in pm}
    meta = json.load(open(os.path.join(a.stitched, 'meta.json')))
    stride = int(round(1.0 / float(meta['scale'][0])))
    sr = json.load(open(os.path.join(a.stitched, 'stitch_report.json')))
    ys = sorted({p['block'][0] * stride - a.y0 for p in sr['pieces'] if p['block'][0] > 0})
    xs = sorted({p['block'][1] * stride - a.x0 for p in sr['pieces'] if p['block'][1] > 0})
    ct = load(os.path.join(a.dir, 'ct_d0.png'))
    H, W = ct.shape
    rep['boundaries_window_px'] = {'y': [y for y in ys if 0 < y < H], 'x': [x for x in xs if 0 < x < W]}
    tiles = [('CT d0', stretch(ct, *np.percentile(ct[ct > 0], [1, 99])) if (ct > 0).any() else ct.astype(np.uint8))]
    ctrl = []
    for i, m in enumerate(models):
        d0 = load(os.path.join(a.dir, 'maps', 'd0', f'{m}_{order}.png'))
        lo, hi = np.percentile(d0, 50), np.percentile(d0, 99.7)
        tiles.append((f'{m} d0 {order}', stretch(d0, lo, hi)))
        if i == 0:
            for v in ('off+15', 'shuf'):
                f = os.path.join(a.dir, 'maps', v, f'{m}_{order}.png')
                if os.path.exists(f):
                    ctrl.append((f'{m} {v}', stretch(load(f), lo, hi)))
    ds = max(1, a.ds)
    h, w = H // ds, W // ds
    ncol = max(len(tiles), len(ctrl), 1)
    sheet = Image.new('RGB', (ncol * w, (2 if ctrl else 1) * (h + 18)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    box = [int(v) for v in a.box.split(',')] if a.box else None
    for row, items in enumerate([tiles, ctrl]):
        for j, (lab, img) in enumerate(items):
            ox, oy = j * w, row * (h + 18)
            sheet.paste(Image.fromarray(img).resize((w, h)).convert('RGB'), (ox, oy + 18))
            dr.text((ox + 4, oy + 3), lab, fill=(255, 255, 255))
            for y in rep['boundaries_window_px']['y']:
                for x in range(0, w, 12):
                    dr.line([(ox + x, oy + 18 + y // ds), (ox + min(x + 6, w), oy + 18 + y // ds)], fill=(0, 200, 255))
            for x in rep['boundaries_window_px']['x']:
                for y in range(0, h, 12):
                    dr.line([(ox + x // ds, oy + 18 + y), (ox + x // ds, oy + 18 + min(y + 6, h))], fill=(0, 200, 255))
            if box:
                bx0, by0, bx1, by1 = [(box[0] - a.x0) // ds, (box[1] - a.y0) // ds, (box[2] - a.x0) // ds, (box[3] - a.y0) // ds]
                dr.rectangle([ox + bx0, oy + 18 + by0, ox + bx1, oy + 18 + by1], outline=(255, 0, 0), width=2)
    sheet.save(a.out, quality=90)
    rep['sheet'] = a.out
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
