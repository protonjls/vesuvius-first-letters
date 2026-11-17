#!/usr/bin/env python3
"""Synthetic known-answer tests for the tiling tools (seeds.py, patch.py, rank.py). CPU, ~1 min, no data download.

  python3 tiling/tests/test_tiling.py            # prints PASS/FAIL per test; exit code 1 if any test fails

Needs numpy, scipy, opencv, tifffile, numcodecs, Pillow. Each test states the answer it knows:
  seeds_parallel  sheets at a known angle and spacing: every seed lies on a sheet, consecutive seeds of a tile are one
                  sheet spacing apart along the true normal, ~one seed per sheet the probe line crosses, no seed in an
                  empty tile.
  seeds_rings     concentric sheets (curved, any direction): seeds on sheets, spacing = ring spacing.
  mask_cli        planar mesh, prediction render with two disconnected on-sheet regions: patch.py keeps exactly the
                  region holding the seed (node count and crop known), covers the seeds on that region within --dedup
                  and no seed off it (off-plane by 20 vox, or on the other region).
  next_part       patch.py next skips done/covered seeds and honours --part.
  coverage        a plane and copies of half of it 3 and 8 voxels off: coverage.py reports 22/41 columns covered at
                  dn 3 vox (half + one edge cell), and nothing at 8 vox (tol 5).
  rank_text       ink maps with (a) rows of letter-like glyphs, (b) the same glyph count as a random blob field,
                  (c) vertical streaks, (d) the glyph rows rotated 90 degrees: rank.py puts (a) first.
  sheets          10 synthetic patches: sheets.py writes 2 sheets (8 per sheet), a key with 10 distinct codes that map
                  one-to-one onto the patches, a code order that differs from the patch-id order (shuffled), and no key in
                  the sheet directory when --key points elsewhere.
  review_coverage plane patch A (41x41 nodes) fully inside one scored window, patch B on the same plane 2 vox off
                  overlapping A's right half and extending 20 columns past it, with no window: 41 of 61 columns of the
                  union surface are reachable (0.672); the top-1 window gives the same, a top-0 list gives 0.
"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
sys.path.insert(0, KIT)
import seeds as S  # noqa: E402

RESULTS = []


def report(name, ok, msg):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}")


def run_tiles(B, T=200):
    import cv2
    Bs = cv2.GaussianBlur(B.astype(np.float32), (0, 0), 3)
    out = {}
    for r0 in range(0, B.shape[0], T):
        for c0 in range(0, B.shape[1], T):
            sd, coh = S.tile_seeds(B, Bs, r0, c0, T, 25, 0.03, 0.2)
            out[(r0, c0)] = sd
    return out


def test_seeds_parallel():
    th = np.deg2rad(30.0); n = np.array([np.cos(th), np.sin(th)])      # normal (x, y)
    yy, xx = np.mgrid[0:600, 0:600]
    d = xx * n[0] + yy * n[1]
    B = (np.mod(d, 24.0) < 5.0)
    B[400:600, 400:600] = False                                         # one empty tile
    res = run_tiles(B)
    errs = []
    on = all(B[r, c] for sd in res.values() for (r, c, _) in sd)
    if not on:
        errs.append('seed off sheet')
    if res[(400, 400)]:
        errs.append('seeds in empty tile')
    counts = []
    for k, sd in res.items():
        if k == (400, 400) or len(sd) < 2:
            continue
        P = np.array([(c, r) for (r, c, _) in sd], float)
        steps = np.diff(P, axis=0)
        along = np.abs(steps @ n)
        across = np.abs(steps @ np.array([-n[1], n[0]]))
        if np.median(np.abs(along - 24.0)) > 1.5 or np.median(across) > 1.5:
            errs.append(f'tile {k}: step along {np.median(along):.1f} across {np.median(across):.1f}')
        full = (k[0] + 200 <= 400 or k[1] + 200 <= 400)                 # tiles not touching the empty one
        if full:
            counts.append(len(sd))
    if counts and not (8 <= min(counts) and max(counts) <= 11):
        errs.append(f'seeds per tile {min(counts)}-{max(counts)} (expect ~240/24 = 10)')
    report('seeds_parallel', not errs, '; '.join(errs) or f'{sum(len(v) for v in res.values())} seeds, per full tile {min(counts)}-{max(counts)}')


def test_seeds_rings():
    yy, xx = np.mgrid[0:800, 0:800]
    rad = np.hypot(xx - 400.0, yy - 400.0)
    B = (np.mod(rad, 30.0) < 6.0) & (rad > 40)
    res = run_tiles(B)
    allsd = [(r, c) for sd in res.values() for (r, c, _) in sd]
    errs = []
    if not all(B[r, c] for r, c in allsd):
        errs.append('seed off sheet')
    dr = []
    for sd in res.values():
        R = sorted(np.hypot(c - 400.0, r - 400.0) for (r, c, _) in sd)
        dr += list(np.diff(R))
    dr = np.array(dr)
    good = np.mean(np.abs(dr - 30.0) < 4.0) if len(dr) else 0.0
    if good < 0.8:
        errs.append(f'only {good:.2f} of consecutive seed radius steps are one ring spacing')
    report('seeds_rings', not errs, '; '.join(errs) or f'{len(allsd)} seeds, {good:.2f} of radial steps = ring spacing')


def write_mesh(d, P):
    import tifffile
    os.makedirs(d, exist_ok=True)
    for i, ch in enumerate('xyz'):
        tifffile.imwrite(f'{d}/{ch}.tif', P[..., i].astype(np.float32))
    json.dump({'format': 'tifxyz', 'scale': [0.05, 0.05], 'type': 'seg'}, open(f'{d}/meta.json', 'w'))


def test_mask_cli():
    import tifffile
    tmp = tempfile.mkdtemp(prefix='tiling_test_')
    H = W = 61
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([1000 + 20.0 * jj, 2000 + 20.0 * ii, np.full((H, W), 5000.0)], -1)
    write_mesh(f'{tmp}/grow', P)
    R = np.zeros((7, H * 20, W * 20), np.uint8)
    R[2:5, :, :710] = 255                     # region A: node cols 0..35 (block of col 35 = px 690-710)
    R[2:5, 100:400, 900:1200] = 255           # region B, disconnected
    np.save(f'{tmp}/m7.npy', R)
    node = lambda i, j: P[i, j]
    sd = [node(30, 10),                       # 0: the seed, region A
          node(30, 5),                        # 1: region A -> covered
          node(10, 50),                       # 2: region B -> not covered
          node(30, 10) + [0, 0, 20],          # 3: 20 vox off the plane -> not covered
          node(30, 10) + [0, 0, 3]]           # 4: 3 vox off -> covered (dedup 5)
    with open(f'{tmp}/seeds.tsv', 'w') as f:
        f.write('idx\ttile_r\ttile_c\tx\ty\tz\trunlen\tcoherence\n')
        for k, p in enumerate(sd):
            f.write(f'{k}\t0\t{k}\t{p[0]:.1f}\t{p[1]:.1f}\t{p[2]:.1f}\t5\t0.9\n')
    cmd = [sys.executable, f'{KIT}/patch.py', 'mask', f'{tmp}/grow', f'{tmp}/m7.npy', f'{tmp}/seeds.tsv', '0',
           f'{tmp}/out', f'{tmp}/covered.txt', '--um', '9.0']
    r = subprocess.run(cmd, capture_output=True, text=True)
    errs = []
    if r.returncode != 0:
        report('mask_cli', False, r.stderr[-400:]); return
    st = json.loads(r.stdout.strip().splitlines()[-1])
    cov = {int(x) for x in open(f'{tmp}/covered.txt').read().split()}
    if st.get('action') != 'kept':
        errs.append(f"action {st.get('action')}")
    if st.get('onsheet_nodes') != H * 36:
        errs.append(f"onsheet_nodes {st.get('onsheet_nodes')} != {H * 36}")
    if cov != {0, 1, 4}:
        errs.append(f'covered {sorted(cov)} != [0, 1, 4]')
    if os.path.exists(f'{tmp}/out/x.tif'):
        X = tifffile.imread(f'{tmp}/out/x.tif')
        if X.shape != (H, 36) or (X == -1).any():
            errs.append(f'kept grid {X.shape}, invalid {(X == -1).sum()} (expect ({H}, 36), 0)')
    else:
        errs.append('no output mesh')
    report('mask_cli', not errs, '; '.join(errs) or f"kept {st['onsheet_nodes']} nodes, {st['area_cm2']} cm2, covered {sorted(cov)}")
    return tmp


def test_next_part(tmp):
    rows = [(i, i // 3, i % 9) for i in range(27)]           # tile_c 0..8
    with open(f'{tmp}/seeds9.tsv', 'w') as f:
        f.write('idx\ttile_r\ttile_c\tx\ty\tz\trunlen\tcoherence\n')
        for i, tr, tc in rows:
            f.write(f'{i}\t{tr}\t{tc}\t{i}\t{i}\t{i}\t5\t0.9\n')
    open(f'{tmp}/done9.txt', 'w').write('0\n')
    open(f'{tmp}/cov9.txt', 'w').write('1\n')

    def nxt(*extra):
        r = subprocess.run([sys.executable, f'{KIT}/patch.py', 'next', f'{tmp}/seeds9.tsv', f'{tmp}/done9.txt',
                            f'{tmp}/cov9.txt', *extra], capture_output=True, text=True)
        return r.stdout.split()
    a = nxt(); b = nxt('--part', '2/3')
    errs = []
    if not a or a[0] != '2':
        errs.append(f'next {a} (expect seed 2: 0 done, 1 covered)')
    if not b or int(b[0]) % 9 < 6:
        errs.append(f'part 2/3 gave {b} (expect a seed with tile_c 6-8)')
    report('next_part', not errs, '; '.join(errs) or f'next {a[0]}, part 2/3 -> {b[0]}')


def glyph(img, x, y, kind, h, w_px, val):
    import cv2
    a = h // 2
    if kind == 0:
        cv2.ellipse(img, (x, y), (int(a * 0.7), a), 0, 0, 360, val, w_px)
    elif kind == 1:
        cv2.line(img, (x, y - a), (x, y + a), val, w_px)
    elif kind == 2:
        cv2.line(img, (x - a // 2, y + a), (x, y - a), val, w_px); cv2.line(img, (x, y - a), (x + a // 2, y + a), val, w_px)
    else:
        cv2.ellipse(img, (x, y), (int(a * 0.7), a), 0, 45, 315, val, w_px)


def test_rank_text():
    import cv2
    from PIL import Image
    rng = np.random.default_rng(0)
    um, pds = 9.0, 2
    pxmm = 1000.0 / (um * pds)                 # PNG px per mm
    Hh = Ww = 2400
    h = int(2.5 * pxmm); wpx = max(2, int(0.5 * pxmm))
    sp_l, sp_r = int(3.5 * pxmm), int(6.0 * pxmm)
    base = lambda: np.clip(rng.normal(0.08, 0.04, (Hh, Ww)), 0, 1).astype(np.float32)
    text = base(); pos = []
    for yc in range(400, Hh - 300, sp_r):
        for xc in range(300, Ww - 300, sp_l):
            pos.append((xc, yc)); glyph(text, xc, yc, int(rng.integers(4)), h, wpx, 0.8)
    blobs = base()
    for _ in pos:
        xc, yc = rng.integers(200, Ww - 200, 2)
        cv2.circle(blobs, (int(xc), int(yc)), int(rng.uniform(0.75, 1.5) * pxmm), 0.8, -1)
    streak = base()
    for xc in rng.integers(200, Ww - 200, 6):
        cv2.line(streak, (int(xc), 0), (int(xc) + 100, Hh - 1), 0.8, wpx * 2)
    rot = np.ascontiguousarray(np.rot90(text))
    maps = {'w001': text, 'w002': blobs, 'w003': streak, 'w004': rot}
    tmp = tempfile.mkdtemp(prefix='tiling_rank_')
    for w, m in maps.items():
        d = f'{tmp}/syn/{w}'; os.makedirs(d)
        Image.fromarray(np.full((Hh, Ww), 128, np.uint8)).save(f'{d}/ct.png')
        Image.fromarray(np.clip(cv2.GaussianBlur(m, (0, 0), 2) * 255, 0, 255).astype(np.uint8)).save(f'{d}/model_fwd.png')
        json.dump({'pixel_um': um, 'png_ds': pds}, open(f'{d}/meta.json', 'w'))
    r = subprocess.run([sys.executable, f'{KIT}/rank.py', tmp, '--out-json', f'{tmp}/rank.json'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        report('rank_text', False, r.stderr[-400:]); return
    J = json.load(open(f'{tmp}/rank.json'))
    order = [d['winding'] for d in J['best_per_map']]
    sc = {d['winding']: round(d['score'], 3) for d in J['best_per_map']}
    report('rank_text', order[:1] == ['w001'], f'order {order} (w001 text rows, w002 blobs, w003 streaks, w004 rotated rows), scores {sc}')


def test_coverage():
    """Reference plane A (41x41 nodes, 20 vox cells); patch B = the same plane shifted 3 vox along the normal over
    A's columns 0-20 only, and a second patch C shifted 8 vox: coverage.py must give dn 3 vox and 22 of 41 columns
    covered for B (columns 0-20, plus column 21, one cell past B's edge, within the documented 1.5-cell in-plane
    allowance) at tol 5, and nothing for C."""
    import tifffile
    tmp = tempfile.mkdtemp(prefix='tiling_cov_')
    jj, ii = np.meshgrid(np.arange(41), np.arange(41))
    A = np.stack([1000 + 20.0 * jj, 2000 + 20.0 * ii, np.full((41, 41), 5000.0)], -1)
    write_mesh(f'{tmp}/A', A)
    jb, ib = np.meshgrid(np.arange(21), np.arange(41))
    Bm = np.stack([1000 + 20.0 * jb, 2000 + 20.0 * ib, np.full((41, 21), 5003.0)], -1)
    write_mesh(f'{tmp}/B', Bm)
    Cm = Bm.copy(); Cm[..., 2] = 5008.0
    write_mesh(f'{tmp}/C', Cm)
    out = {}
    for k in 'BC':
        r = subprocess.run([sys.executable, f'{KIT}/coverage.py', f'{tmp}/A', f'{tmp}/{k}', '--tol', '5'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            report('coverage', False, r.stderr[-400:]); return
        out[k] = json.loads(r.stdout.strip().splitlines()[-1])
    fb, mb, fc = out['B']['frac_within_tol'], out['B']['median_dn_within_tol'], out['C']['frac_within_tol']
    ok = abs(fb - 22 / 41) < 1e-9 and mb is not None and abs(mb - 3.0) < 1e-6 and fc == 0.0
    report('coverage', ok, f'B (3 vox off, cols 0-20): frac {fb:.4f} (expect {22 / 41:.4f}) median dn {mb}; C (8 vox off): frac {fc:.3f}')


def test_sheets():
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix='tiling_sheets_')
    for i in range(10):
        d = f'{tmp}/maps/syn/w{100 + i:04d}'; os.makedirs(d)
        for o in ('fwd', 'rev'):
            Image.fromarray(np.full((300, 400), 20 * i, np.uint8)).save(f'{d}/reader-v3_{o}.png')
        json.dump({'pixel_um': 9.0, 'png_ds': 2}, open(f'{d}/meta.json', 'w'))
    r = subprocess.run([sys.executable, f'{KIT}/sheets.py', f'{tmp}/maps', f'{tmp}/out', '--key', f'{tmp}/key.tsv',
                        '--seed', '3'], capture_output=True, text=True)
    if r.returncode != 0:
        report('sheets', False, r.stderr[-400:]); return
    jpgs = sorted(f for f in os.listdir(f'{tmp}/out') if f.endswith('.jpg'))
    rows = [l.rstrip('\n').split('\t') for l in open(f'{tmp}/key.tsv')][1:]
    codes, pats = [x[0] for x in rows], [x[2] for x in rows]
    ok = (jpgs == ['sheet_01.jpg', 'sheet_02.jpg'] and len(set(codes)) == 10 and sorted(pats) == [f'w{100 + i:04d}' for i in range(10)]
          and [c for _, c in sorted(zip(pats, codes))] != sorted(codes) and not os.path.exists(f'{tmp}/out/key.tsv'))
    report('sheets', ok, f'sheets {jpgs}, codes {len(set(codes))} distinct, patches {len(set(pats))}')


def test_review_coverage():
    tmp = tempfile.mkdtemp(prefix='tiling_revcov_')
    jj, ii = np.meshgrid(np.arange(41), np.arange(41))
    A = np.stack([1000 + 20.0 * jj, 2000 + 20.0 * ii, np.full((41, 41), 5000.0)], -1)
    B = np.stack([1400 + 20.0 * jj, 2000 + 20.0 * ii, np.full((41, 41), 5002.0)], -1)
    write_mesh(f'{tmp}/patches/w0001', A); write_mesh(f'{tmp}/patches/w0002', B)
    np.savez(f'{tmp}/rank.npz', winding=np.array(['w0001']), y=np.array([0]), x=np.array([0]), side=np.array([410]),
             score=np.array([1.0]), band=np.array(['syn']), model=np.array(['m']), order=np.array(['fwd']))
    out = {}
    for top in (1, 0):
        r = subprocess.run([sys.executable, f'{KIT}/review_coverage.py', f'{tmp}/rank.npz', f'{tmp}/patches', '--top', str(top)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            report('review_coverage', False, r.stderr[-400:]); return
        out[top] = eval(r.stdout.strip().splitlines()[-1])
    fs, f1, f0 = out[1]['frac_scored'], out[1]['frac_top1'], out[0]['frac_top0']
    ok = abs(fs - 41 / 61) < 0.01 and abs(f1 - fs) < 1e-9 and f0 == 0.0 and out[1]['patches_scored'] == 1
    report('review_coverage', ok, f'scored {fs:.3f} (expect {41 / 61:.3f}), top1 {f1:.3f}, top0 {f0:.3f}')


if __name__ == '__main__':
    test_seeds_parallel()
    test_seeds_rings()
    t = test_mask_cli()
    if t:
        test_next_part(t)
    test_rank_text()
    test_coverage()
    test_sheets()
    test_review_coverage()
    print(f'{sum(RESULTS)}/{len(RESULTS)} passed')
    sys.exit(0 if all(RESULTS) else 1)
