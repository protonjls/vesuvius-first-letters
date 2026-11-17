#!/usr/bin/env python3
"""Contact sheets of EVERY patch of a tiling run, for a full review that does not depend on rank.py windows.

  sheets.py <maps_dir | sweep.tar> <out_dir> [--key key.tsv] [--size 480] [--per 8] [--cols 2] [--seed 0]
            [--model reader-v3]

Input as rank.py: <...>/<band>/<wNNNN>/{<model>_fwd.png, <model>_rev.png, meta.json} (ink_patches.sh output dir, or a
tar of it). Each patch gives one tile pair, labelled with an anonymised code (P001..) and the sides A (fwd) / B (rev),
each map box-downsampled to --size px on its long side, with a 1 cm bar from meta.json (pixel_um * png_ds). Tiles are
placed in a seeded random order, --per patches per sheet in --cols columns, so a reviewer cannot tell patch ids or
tiling order. Writes <out_dir>/sheet_NN.jpg and the code -> patch key (--key, default <out_dir>/key.tsv). For a blind
review, write the key outside the directory the reviewer reads.

Why: rank.py scores only 20 mm windows that are >= 60% on-sheet. On the PHerc0306B known-answer window (199 patches)
its scored windows reach 41% of the on-sheet surface and its top 36 reach 5% (review_coverage.py). A blind review of
these sheets found both claim-covering patches (see README).
"""
import argparse, io, json, os, random
from PIL import Image, ImageDraw
from rank import iter_windings


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--key', default=None); ap.add_argument('--size', type=int, default=480)
    ap.add_argument('--per', type=int, default=8); ap.add_argument('--cols', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--model', default='reader-v3')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fw, rv = f'{a.model}_fwd.png', f'{a.model}_rev.png'
    pats = [(b, w, f) for b, w, f in iter_windings(a.src) if fw in f and rv in f]
    order = list(range(len(pats))); random.Random(a.seed).shuffle(order)
    S, gap, lab = a.size, 6, 22
    tw, th = 2 * S + gap, S + lab
    rows = -(-a.per // a.cols)
    key = []
    for si in range(0, len(order), a.per):
        sheet = Image.new('L', (a.cols * tw + (a.cols - 1) * 3 * gap, rows * (th + gap)), 40)
        d = ImageDraw.Draw(sheet)
        for k, pi in enumerate(order[si:si + a.per]):
            band, w, files = pats[pi]
            code = f'P{si + k + 1:03d}'
            meta = json.loads(files['meta.json']) if 'meta.json' in files else {}
            um = float(meta.get('pixel_um', 9.362)) * float(meta.get('png_ds', 2))
            r, c = divmod(k, a.cols); x0 = c * (tw + 3 * gap); y0 = r * (th + gap)
            for j, (f, side) in enumerate(((fw, 'A'), (rv, 'B'))):
                im = Image.open(io.BytesIO(files[f])).convert('L')
                s = S / max(im.size)
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BOX)
                xo = x0 + j * (S + gap)
                sheet.paste(im, (xo, y0 + lab))
                bar = 10000.0 / um * s
                d.rectangle([xo + 8, y0 + lab + S - 14, xo + 8 + bar, y0 + lab + S - 9], fill=255)
                d.text((xo + 4, y0 + 4), f'{code} {side}', fill=255)
            key.append((code, band, w, si // a.per + 1, k + 1))
        sheet.save(f'{a.out}/sheet_{si // a.per + 1:02d}.jpg', quality=88)
    kp = a.key or f'{a.out}/key.tsv'
    with open(kp, 'w') as f:
        f.write('code\tband\tpatch\tsheet\tslot\n' + ''.join('\t'.join(map(str, x)) + '\n' for x in sorted(key)))
    print(f'{len(pats)} patches -> {-(-len(pats) // a.per)} sheets in {a.out}; key {kp} (A = fwd, B = rev)')


if __name__ == '__main__':
    main()
