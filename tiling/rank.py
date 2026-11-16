#!/usr/bin/env python3
"""Rank 4 cm^2 windows of swept ink maps by text-likeness, for visual review (not a detector of record).

  rank_text.py <sweep.tar | sweep_dir> [--out-json f.json] [--sheet f.png] [--top 24] [--um 8.64]

Input layout (tools/jobs/sweep_band.sh): <band>/<wNNN>/{ct.png, <model>_{fwd,rev}.png, meta.json}; maps are
uint8 = round(p*255) in the raw render frame (rows ~ scroll axis z, cols ~ along the winding), so text rows run
along the columns (horizontal).
Per map (each model x order), at 1/DS resolution:
  valid = ct > 0, eroded; z = (p - median) / (1.4826 MAD) over valid pixels of the whole map;
  r = G(sig_s) z - G(sig_b) z (sigmas in mm: --sig-s, --sig-b) (masked/normalised Gaussians): letter-scale excess over the local background.
Per window (side = 20 mm, stride side/4, valid fraction >= MINVALID):
  mass  = mean(clip(r - 1, 0)) over valid pixels           (amount of letter-scale ink excess)
  rows  = max normalised autocorrelation of the row profile mean_x(clip(r,0)) at lags 3-8 mm (line spacing)
  cols  = max normalised autocorrelation of the column profile in the window's inked rows at lags 1.5-4.5 mm
  score = mass * (1 + rows + cols)
Outputs a JSON list of windows sorted by score and an optional contact sheet of the top windows (ink | CT).
"""
import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
DS = 4


def load_png(data):
    return np.asarray(Image.open(io.BytesIO(data)))


def block_mean(a, k):
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))


def iter_windings(src):
    """yield (band, winding, {name: bytes}) grouped by winding dir."""
    groups = {}
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for f in files:
                if f.endswith('.png') or f == 'meta.json':
                    rel = os.path.relpath(os.path.join(root, f), src).split(os.sep)
                    if len(rel) >= 3:
                        groups.setdefault((rel[-3], rel[-2]), {})[f] = open(os.path.join(root, f), 'rb').read()
    else:
        with tarfile.open(src) as t:   # a tar still being appended to may end in a partial member: stop there
            try:
                for m in t:
                    if not m.isfile():
                        continue
                    parts = m.name.split('/')
                    if len(parts) >= 3:
                        data = t.extractfile(m).read()
                        if len(data) == m.size:
                            groups.setdefault((parts[-3], parts[-2]), {})[parts[-1]] = data
            except (tarfile.TarError, OSError, EOFError) as e:
                print(f'{src}: stopped at a partial member ({e})', file=sys.stderr)
    for (band, w) in sorted(groups):
        yield band, w, groups[(band, w)]


def mgauss(x, m, s):
    num = ndi.gaussian_filter(np.where(m, x, 0.0), s, mode='constant')
    den = ndi.gaussian_filter(m.astype(np.float64), s, mode='constant')
    return num / np.maximum(den, 1e-6)


def autocorr_peak(prof, lo, hi):
    prof = prof - prof.mean()
    d = float((prof * prof).sum())
    if d <= 0 or hi >= len(prof):
        hi = min(hi, len(prof) - 1)
    if d <= 0 or hi <= lo:
        return 0.0
    ac = [float((prof[:-L] * prof[L:]).sum()) / d for L in range(lo, hi + 1)]
    return max(0.0, max(ac))


def set_ds(k):
    global DS
    DS = k


WIN_MM = 20.0


def score_map(p, valid, um):
    pxmm = 1000.0 / (um * DS)                      # downsampled pixels per mm
    v = valid
    if v.sum() < 1000:
        return None, None
    med = np.median(p[v]); mad = np.median(np.abs(p[v] - med)) * 1.4826 + 1e-3
    z = (p - med) / mad
    r = mgauss(z, v, SIG_S * pxmm) - mgauss(z, v, SIG_B * pxmm)
    r[~v] = 0
    side = int(round(WIN_MM * pxmm)); step = max(1, side // 4)
    H, W = p.shape
    out = []
    ys = list(range(0, max(1, H - side + 1), step)) or [0]
    xs = list(range(0, max(1, W - side + 1), step)) or [0]
    for y in ys:
        for x in xs:
            vv = v[y:y + side, x:x + side]
            if vv.mean() < MINVALID:
                continue
            rr = r[y:y + side, x:x + side]
            mass = float(np.clip(rr - 1, 0, None)[vv].mean())
            pos = np.clip(rr, 0, None)
            rowp = pos.sum(1) / np.maximum(vv.sum(1), 1)
            rows = autocorr_peak(rowp, int(3 * pxmm), int(8 * pxmm))
            ink_rows = rowp > np.percentile(rowp, 70)
            colp = pos[ink_rows].sum(0) / np.maximum(vv[ink_rows].sum(0), 1)
            cols = autocorr_peak(colp, int(1.5 * pxmm), int(4.5 * pxmm))
            out.append(dict(y=y * DS, x=x * DS, side=side * DS, mass=mass, rows=rows, cols=cols,
                            score=mass * (1 + rows + cols), valid=float(vv.mean())))
    return out, r


def main():
    global MINVALID, SIG_S, SIG_B, WIN_MM
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out-json')
    ap.add_argument('--sheet')
    ap.add_argument('--ink-only', action='store_true', help='sheet tiles show the ink map only (no CT panel)')
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--top', type=int, default=24)
    ap.add_argument('--um', type=float, default=None, help='override pixel size (default meta.json pixel_um)')
    ap.add_argument('--min-valid', type=float, default=0.6)
    ap.add_argument('--win-mm', type=float, default=20.0, help='window side in mm (default 20)')
    ap.add_argument('--sig-s', type=float, default=0.25, help='small Gaussian sigma, mm')
    ap.add_argument('--sig-b', type=float, default=1.0, help='background Gaussian sigma, mm')
    ap.add_argument('--models', default='', help='comma list of model names to use (default all)')
    ap.add_argument('--out-npz', help='every window (band, winding, model, order, y, x, side, score) as arrays')
    a = ap.parse_args()
    MINVALID = a.min_valid; SIG_S = a.sig_s; SIG_B = a.sig_b; WIN_MM = a.win_mm
    wins, crops = [], {}
    for band, w, files in iter_windings(a.src):
        if 'ct.png' not in files or 'meta.json' not in files:
            print(f'{band}/{w}: no ct.png', file=sys.stderr); continue
        meta = json.loads(files['meta.json']) if 'meta.json' in files else {}
        pds = int(meta.get('png_ds', 1))                 # PNGs stored at 1/png_ds of the render resolution
        um = (a.um or meta.get('pixel_um', 8.64)) * pds  # PNG pixel size
        set_ds(max(1, 4 // pds))                         # analyse at ~4x render pixels whatever png_ds is
        ct = load_png(files['ct.png'])
        validf = ct > 0
        valid = block_mean(validf.astype(np.float32), DS) > 0.99
        valid = ndi.binary_erosion(valid, iterations=8)
        ctd = block_mean(ct.astype(np.float32), DS)
        for f, data in sorted(files.items()):
            if f in ('ct.png', 'meta.json') or not f.endswith('.png'):
                continue
            model, order = f[:-4].rsplit('_', 1)
            if a.models and model not in a.models.split(','):
                continue
            m = load_png(data).astype(np.float32) / 255.0
            if m.shape != ct.shape:
                print(f'{band}/{w}/{f}: shape {m.shape} != ct {ct.shape}', file=sys.stderr); continue
            md = block_mean(m, DS)
            res, r = score_map(md, valid, um)
            if res is None:
                continue
            for d in res:
                d.update(band=band, winding=w, model=model, order=order, png_ds=pds)
            wins += res
            crops[(band, w, model, order)] = (md, ctd)
        print(f'{band}/{w}: {len(wins)} windows so far', file=sys.stderr)
    wins.sort(key=lambda d: -d['score'])
    if a.out_npz:
        np.savez_compressed(a.out_npz, **{k: np.array([d[k] for d in wins]) for k in
                                          ('band', 'winding', 'model', 'order', 'y', 'x', 'side', 'score')})
    # per-(winding, model, order) best window, for ranking windings
    best = {}
    for d in wins:
        k = (d['band'], d['winding'], d['model'], d['order'])
        if k not in best:
            best[k] = d
    ranked = sorted(best.values(), key=lambda d: -d['score'])
    # band-level reading-side asymmetry: per winding, best rev score minus best fwd score (over models)
    bw = {}
    for d in ranked:
        k = (d['band'], d['winding'], d['order'])
        bw[k] = max(bw.get(k, 0.0), d['score'])
    wset = sorted({(b, w) for b, w, _ in bw})
    delta = np.array([bw.get((b, w, 'rev'), 0.0) - bw.get((b, w, 'fwd'), 0.0) for b, w in wset]) if wset else np.zeros(0)
    band_stats = {'windings': len(wset), 'median_rev_minus_fwd': float(np.median(delta)) if len(delta) else None,
                  'frac_rev_gt_fwd': float(np.mean(delta > 0)) if len(delta) else None,
                  'top40_rev': int(sum(d['order'] == 'rev' for d in ranked[:40]))}
    print('band_stats', json.dumps(band_stats), file=sys.stderr)
    if a.out_json:
        json.dump({'src': a.src, 'n_windows': len(wins), 'band_stats': band_stats, 'best_per_map': ranked, 'top_windows': wins[:500]},
                  open(a.out_json, 'w'), indent=0)
    for i, d in enumerate(ranked[:a.top]):
        print(f"{i:3d} {d['band']} {d['winding']} {d['model']:<12} {d['order']} y{d['y']} x{d['x']} "
              f"score {d['score']:.4f} mass {d['mass']:.4f} rows {d['rows']:.2f} cols {d['cols']:.2f}")
    if a.sheet:
        tiles = []
        for d in ranked[:a.top]:
            md, ctd = crops[(d['band'], d['winding'], d['model'], d['order'])]
            y, x, s = d['y'] // DS, d['x'] // DS, d['side'] // DS
            im = np.clip(md[y:y + s, x:x + s] * 255, 0, 255).astype(np.uint8)
            c = ctd[y:y + s, x:x + s]
            lo, hi = np.percentile(c, [1, 99]) if c.size else (0, 1)
            c = np.clip((c - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
            if a.ink_only:
                t = Image.new('L', (s, s + 14), 0)
                t.paste(Image.fromarray(im), (0, 14))
            else:
                t = Image.new('L', (2 * s + 4, s + 14), 0)
                t.paste(Image.fromarray(im), (0, 14)); t.paste(Image.fromarray(c), (s + 4, 14))
            ImageDraw.Draw(t).text((2, 1), f"{d['winding']} {d['model'][:8]} {d['order']} {d['score']:.3f}", fill=255)
            tiles.append(t)
        if tiles:
            cw = max(t.size[0] for t in tiles); chh = max(t.size[1] for t in tiles); nc = a.cols
            nr = (len(tiles) + nc - 1) // nc
            sheet = Image.new('L', (nc * (cw + 4), nr * (chh + 4)), 40)
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % nc) * (cw + 4), (i // nc) * (chh + 4)))
            sheet.save(a.sheet)


if __name__ == '__main__':
    main()
