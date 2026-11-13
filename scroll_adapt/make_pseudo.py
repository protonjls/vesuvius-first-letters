#!/usr/bin/env python3
"""Consensus pseudo-labels for scroll-adaptive fine-tuning (scroll-adapt).

Two ink probability maps of one surface (same frame, float in [0,1], e.g. reader-v3 and rectina-m4 in the reading
depth order) -> ink label and supervision mask:
  ink    = sqrt(a*b) >= --ink-thr  (both models confident),  small components (< --min-area px) dropped
  blank  = max(a,b)  <= --blank-thr (neither model sees ink)
  ignore = everything else, plus a --ring px band around every ink component and the invalid (no-papyrus) area.
Outputs uint8 PNGs (0/255) and .npy (0/1): <out>_inklabels.*, <out>_supervision_mask.* (1 where ink or blank), and a JSON
summary with coverages. Thresholds are on the label-smoothed probability scale of these models (blank papyrus
~0.27, confident ink >0.6); see results/pseudo_quality_1447.txt.
"""
import argparse, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None


def load_img(p):
    if p.endswith('.npy'):
        return np.squeeze(np.load(p))
    if p.endswith(('.tif', '.tiff')):
        import tifffile
        return np.squeeze(tifffile.imread(p))
    return np.asarray(Image.open(p).convert('L'))


def load_map(p):
    a = load_img(p)
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map-a', required=True); ap.add_argument('--map-b', required=True)
    ap.add_argument('--valid', help='image whose nonzero pixels mark papyrus (e.g. a central render layer)')
    ap.add_argument('--exclude', action='append', default=[], help='y0:y1,x0:x1 box (map px) forced to ignore; repeatable')
    ap.add_argument('--ink-thr', type=float, default=0.6); ap.add_argument('--blank-thr', type=float, default=0.3)
    ap.add_argument('--min-area', type=int, default=30); ap.add_argument('--ring', type=int, default=3)
    ap.add_argument('--out', required=True, help='output prefix')
    a = ap.parse_args()
    A, B = load_map(a.map_a), load_map(a.map_b)
    assert A.shape == B.shape, (A.shape, B.shape)
    valid = np.ones(A.shape, bool)
    if a.valid:
        v = load_img(a.valid)
        if v.ndim == 3: v = v[..., 0] if v.shape[-1] <= 4 else v[v.shape[0] // 2]
        assert v.shape == A.shape, (v.shape, A.shape)
        valid = ndi.binary_erosion(v > 0, iterations=8)        # keep away from mesh borders
    c = np.sqrt(A * B); mx = np.maximum(A, B)
    ink = (c >= a.ink_thr) & valid
    lab, n = ndi.label(ink)
    if n:
        area = ndi.sum(ink, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = area >= a.min_area
        ink = keep[lab]
    blank = (mx <= a.blank_thr) & valid
    if a.ring > 0:
        blank &= ~ndi.binary_dilation(ink, iterations=a.ring)
    for box in a.exclude:
        ys, xs = box.split(','); y0, y1 = map(int, ys.split(':')); x0, x1 = map(int, xs.split(':'))
        ink[y0:y1, x0:x1] = False; blank[y0:y1, x0:x1] = False
    sup = ink | blank
    np.save(a.out + '_inklabels.npy', ink.astype(np.uint8)); np.save(a.out + '_supervision_mask.npy', sup.astype(np.uint8))
    Image.fromarray((ink * 255).astype(np.uint8)).save(a.out + '_inklabels.png')
    Image.fromarray((sup * 255).astype(np.uint8)).save(a.out + '_supervision_mask.png')
    nv = max(int(valid.sum()), 1)
    s = dict(shape=list(A.shape), valid_px=nv, ink_frac=float(ink.sum() / nv), blank_frac=float(blank.sum() / nv),
             ignore_frac=float(1 - sup.sum() / nv), ink_thr=a.ink_thr, blank_thr=a.blank_thr, min_area=a.min_area,
             ring=a.ring, map_a=a.map_a, map_b=a.map_b, exclude=a.exclude)
    json.dump(s, open(a.out + '_pseudo.json', 'w'), indent=1)
    print(json.dumps(s))


if __name__ == '__main__':
    main()
