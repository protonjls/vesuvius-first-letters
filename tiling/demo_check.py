#!/usr/bin/env python3
"""Compare a patch's ink maps, resampled into the PHerc0306B winners' claim-mesh frame, with a reference reading view.

  demo_check.py <fwd_in_winners.png> <rev_in_winners.png> <reference_reading.jpg> <out_dir> [--shift 150]

The resampled maps (followup/mesh_resample.py output: raw frame, ink bright, 0 = no patch surface) are turned into the
reference's reading view (flipped top-to-bottom, inverted, autocontrast 0.5%), saved as <out_dir>/{fwd,rev}_reading.png,
and correlated (Pearson r) with the reference over pixels valid in both. Control: the reference shifted by +-shift px
along the line (150 px = 15 mesh cells = about 2.6 mm). Prints and writes <out_dir>/demo_check.json.
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fwd'); ap.add_argument('rev'); ap.add_argument('ref'); ap.add_argument('out')
    ap.add_argument('--shift', type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref_im = Image.open(a.ref).convert('L')
    ref = np.asarray(ref_im).astype(float)
    out = {'reference': a.ref, 'shift_px': a.shift}
    for name, path in (('fwd', a.fwd), ('rev', a.rev)):
        im = Image.open(path).convert('L')
        if im.size != ref_im.size:
            raise SystemExit(f'size mismatch {im.size} vs reference {ref_im.size}')
        valid = np.flipud(np.asarray(im)) > 0
        rd = ImageOps.autocontrast(ImageOps.invert(ImageOps.flip(im)), cutoff=0.5)
        rd.save(f'{a.out}/{name}_reading.png')
        m = np.asarray(rd).astype(float)
        v = valid & (ref > 0) & (ref < 255)
        r = float(np.corrcoef(ref[v], m[v])[0, 1])
        ctrl = [float(np.corrcoef(np.roll(ref, s, axis=1)[v], m[v])[0, 1]) for s in (-a.shift, a.shift)]
        out[name] = {'r': r, 'r_shifted': ctrl, 'n_px': int(v.sum())}
    print(json.dumps(out))
    json.dump(out, open(f'{a.out}/demo_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
