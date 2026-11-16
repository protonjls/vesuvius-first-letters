#!/usr/bin/env python3
"""Seed points on every predicted sheet crossing of one axial slice window (tiling step 1).

  seeds.py --pred URL --z Z --x0 X0 --x1 X1 --y0 Y0 --y1 Y1 --out seeds.tsv [--png overlay.png]
               [--tile 200] [--thr 128] [--maxrun 25] [--minfrac 0.03] [--mincoh 0.2]

Reads the level-0 slice z of a surface-prediction zarr (e.g. m7 L0 th0.2; plain HTTP GETs of the zarr v2 level-0 chunks covering the window, blosc via numcodecs).
For each tile x tile block: dominant sheet normal n = principal eigenvector of the block-summed structure tensor of the
smoothed binary prediction (B = pred >= thr); a line through the block centre along n (length 1.2 tile) is sampled
(nearest pixel); every run of B along it no longer than --maxrun px gives one seed at the run centre. Blocks with
positive fraction < minfrac or coherence < mincoh are skipped. No umbilicus needed: works where sheets run in any
direction (crushed/folded regions). Many seeds lie on the same sheet; the consumer (tile_grow.sh) skips seeds already
covered by an earlier grown patch.
Output TSV columns: idx tile_r tile_c x y z runlen coherence (x y z in volume voxels).
"""
import argparse, json, os, urllib.request
import numpy as np
import numcodecs
import cv2


def zarray(url):
    return json.loads(urllib.request.urlopen(url + '/0/.zarray', timeout=60).read())


def read_window(url, z, y0, y1, x0, x1):
    meta = zarray(url); cz, cy, cx = meta['chunks']; sep = meta.get('dimension_separator', '.')
    comp = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    out = np.zeros((y1 - y0, x1 - x0), np.uint8); iz = z // cz
    for iy in range(y0 // cy, (y1 - 1) // cy + 1):
        for ix in range(x0 // cx, (x1 - 1) // cx + 1):
            key = sep.join(map(str, (iz, iy, ix)))
            try:
                raw = urllib.request.urlopen(f'{url}/0/{key}', timeout=120).read()
            except Exception:
                continue  # missing chunk = fill value 0
            buf = comp.decode(raw) if comp else raw
            ch = np.frombuffer(buf, np.uint8).reshape(cz, cy, cx)[z - iz * cz]
            ya, yb = max(y0, iy * cy), min(y1, (iy + 1) * cy); xa, xb = max(x0, ix * cx), min(x1, (ix + 1) * cx)
            out[ya - y0:yb - y0, xa - x0:xb - x0] = ch[ya - iy * cy:yb - iy * cy, xa - ix * cx:xb - ix * cx]
    return out


def tile_seeds(B, Bs, r0, c0, T, maxrun, minfrac, mincoh):
    blk = B[r0:r0 + T, c0:c0 + T]
    if blk.size == 0 or blk.mean() < minfrac:
        return [], 0.0
    sub = Bs[r0:r0 + T, c0:c0 + T]
    gy, gx = np.gradient(sub)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    tr = jxx + jyy
    if tr <= 0:
        return [], 0.0
    disc = np.sqrt(max((jxx - jyy) ** 2 / 4 + jxy ** 2, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    coh = float((l1 - l2) / (l1 + l2))
    if coh < mincoh:
        return [], coh
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # direction of max gradient energy = sheet normal (x, y)
    nx, ny = np.cos(ang), np.sin(ang)
    H, W = B.shape
    cr, cc = r0 + T / 2.0, c0 + T / 2.0
    L = int(0.6 * T)
    ts = np.arange(-L, L + 1)
    rr = np.rint(cr + ts * ny).astype(int); cc_ = np.rint(cc + ts * nx).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc_ >= 0) & (cc_ < W)
    v = np.zeros(len(ts), bool); v[ok] = B[rr[ok], cc_[ok]]
    out = []
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            n = j - i + 1
            if n <= maxrun and ok[i] and ok[j]:
                m = (i + j) // 2
                out.append((int(rr[m]), int(cc_[m]), n))
            i = j + 1
        else:
            i += 1
    return out, coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    for k in ('z', 'x0', 'x1', 'y0', 'y1'):
        ap.add_argument('--' + k, type=int, required=True)
    ap.add_argument('--tile', type=int, default=200); ap.add_argument('--thr', type=int, default=128)
    ap.add_argument('--maxrun', type=int, default=25); ap.add_argument('--minfrac', type=float, default=0.03)
    ap.add_argument('--mincoh', type=float, default=0.2)
    ap.add_argument('--out', required=True); ap.add_argument('--png', default='')
    ap.add_argument('--npy', default='', help='optional cache of the fetched slice window (read if present)')
    a = ap.parse_args()
    if a.npy and os.path.exists(a.npy):
        S = np.load(a.npy)
    else:
        S = read_window(a.pred, a.z, a.y0, a.y1, a.x0, a.x1)
        if a.npy:
            np.save(a.npy, S)
    B = S >= a.thr
    Bs = cv2.GaussianBlur(B.astype(np.float32), (0, 0), 3)
    T = a.tile
    rows = []
    stats = []
    for ti, r0 in enumerate(range(0, B.shape[0], T)):
        for tj, c0 in enumerate(range(0, B.shape[1], T)):
            sd, coh = tile_seeds(B, Bs, r0, c0, T, a.maxrun, a.minfrac, a.mincoh)
            stats.append(coh)
            for (r, c, n) in sd:
                rows.append((ti, tj, a.x0 + c, a.y0 + r, a.z, n, coh))
    with open(a.out, 'w') as f:
        f.write('idx\ttile_r\ttile_c\tx\ty\tz\trunlen\tcoherence\n')
        for i, (ti, tj, x, y, z, n, coh) in enumerate(rows):
            f.write(f'{i}\t{ti}\t{tj}\t{x}\t{y}\t{z}\t{n}\t{coh:.3f}\n')
    print(f'slice {S.shape} pos frac {B.mean():.3f}; tiles {len(stats)}; seeds {len(rows)}')
    if a.png:
        im = cv2.cvtColor((B * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) // 2
        for (ti, tj, x, y, z, n, coh) in rows:
            cv2.circle(im, (x - a.x0, y - a.y0), 3, (0, 0, 255), -1)
        cv2.imwrite(a.png, im)


if __name__ == '__main__':
    main()
