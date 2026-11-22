#!/usr/bin/env python3
"""Known-answer comparison for rescan/ (see README.md): our canonical 2.4 um maps against a published team ink map.

Two renders of one sheet on the 2.4 um rescan:
  B = the sheet's own 2.4 um mesh (mesh B), window = the team map window (the team map is on mesh B's render grid);
  C = the sheet's 9.362 um mesh (mesh A) mapped into the rescan by the inverse official transform (render_infer.sh with
      <transform.json>:inv); C's pixels are resampled into the team window through a node fit between the two meshes
      (nodecorr.py): node_A = px_C * scale_A / sA, node_B = fit(node_A), team px = node_B / scale_B.
  kat_compare.py window  --corr corr.json --affine T.json --mesh-a A --mesh-b B --win R0 C0 H W [--margin 200]
        prints the C render window "x0 y0 width height" that covers the team window (plus a margin)
  kat_compare.py compare --b DIR_B --c DIR_C --team team.tif --win R0 C0 H W --corr corr.json --affine T.json
        --mesh-a A --mesh-b B --c-origin X0 Y0 --out DIR
        DIR_B / DIR_C = render_infer.sh outputs (maps/d0_{fwd,rev}.png, ct.png). Writes kat.json + sheet.png
        (team | B | C resampled | CT of B). r = Pearson r after a Gaussian blur (sigma 4 px) over pixels valid in the
        renders. Known-answer bars used in the README: B (best order) r >= 0.8 vs the team map; C r >= 0.6 vs B."""
import argparse, json, sys
import numpy as np


def blur(a, s=4.0):
    import cv2
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)


def pearson(a, b, m):
    x, y = blur(a)[m], blur(b)[m]
    return float(np.corrcoef(x, y)[0, 1]) if m.sum() > 100 and x.std() > 0 and y.std() > 0 else float('nan')


def mesh_scale(d):
    s = json.load(open(f'{d}/meta.json'))['scale']
    return float(s[0]) if isinstance(s, (list, tuple)) else float(s)


def s_affine(path):
    p = path[:-4] if path.endswith(':inv') else path
    M = np.array(json.load(open(p))['transformation_matrix'], float)[:3, :3]
    return float(np.cbrt(abs(np.linalg.det(np.linalg.inv(M)))))


def px_c_to_team(corr, sA, scale_a, scale_b):
    """2x3 matrix: C render px (row, col, 1) -> team px (row, col)."""
    W = np.array(json.load(open(corr))['fit_nodeB_from_nodeA (rows: [rowA,colA,1] -> [rowB,colB])'], float)  # 3x2
    k = scale_a / sA
    return np.array([[W[0, 0] * k, W[1, 0] * k, W[2, 0]], [W[0, 1] * k, W[1, 1] * k, W[2, 1]]]) / scale_b


def window(a):
    A2 = px_c_to_team(a.corr, s_affine(a.affine), mesh_scale(a.mesh_a), mesh_scale(a.mesh_b))
    Ai = np.linalg.inv(np.vstack([A2, [0, 0, 1]]))
    R0, C0, H, W = a.win
    corners = np.array([[R0, C0, 1], [R0, C0 + W, 1], [R0 + H, C0, 1], [R0 + H, C0 + W, 1]], float) @ Ai[:2].T
    r0, c0 = np.floor(corners.min(0)) - a.margin; r1, c1 = np.ceil(corners.max(0)) + a.margin
    print(int(c0), int(r0), int(c1 - c0), int(r1 - r0))


def resample_into_team(img, A2, win, c_origin):
    """Sample a C-frame image (rows/cols relative to the C crop origin (x0, y0)) on the team window grid."""
    import cv2
    R0, C0, H, W = win
    Ai = np.linalg.inv(np.vstack([A2, [0, 0, 1]]))
    rr, cc = np.mgrid[R0:R0 + H, C0:C0 + W].astype(np.float64)
    pr = Ai[0, 0] * rr + Ai[0, 1] * cc + Ai[0, 2] - c_origin[1]
    pc = Ai[1, 0] * rr + Ai[1, 1] * cc + Ai[1, 2] - c_origin[0]
    return cv2.remap(img.astype(np.float32), pc.astype(np.float32), pr.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def read_team(path, win):
    """Window R0 C0 H W of a team ink map (uint8 TIFF; full-resolution level of its first series) as float in [0, 1].
    Decodes the whole level (about 750 MB for the PHerc0139 demo map) so it needs no zarr (tifffile's zarr store
    requires zarr >= 3)."""
    import tifffile
    R0, C0, H, W = win
    with tifffile.TiffFile(path) as tf:
        img = tf.series[0].levels[0].asarray()
    assert img.ndim == 2 and R0 >= 0 and C0 >= 0 and R0 + H <= img.shape[0] and C0 + W <= img.shape[1], (img.shape, win)
    return img[R0:R0 + H, C0:C0 + W].astype(np.float32) / 255.0


def compare(a):
    from PIL import Image
    load = lambda p: np.array(Image.open(p)).astype(np.float32)
    team = read_team(a.team, a.win)
    H, W = a.win[2], a.win[3]
    valid_b = load(f'{a.b}/ct.png') > 0
    out = {'window_rc_hw': a.win}
    B = {o: load(f'{a.b}/maps/d0_{o}.png') / 255 for o in ('fwd', 'rev')}
    for o in B:
        assert B[o].shape == (H, W), (B[o].shape, (H, W))
        out[f'B_r_vs_team_{o}'] = pearson(B[o], team, valid_b)
    ob = max(B, key=lambda o: -9 if np.isnan(out[f'B_r_vs_team_{o}']) else out[f'B_r_vs_team_{o}'])
    out['B_order'] = ob; out['B_r'] = out[f'B_r_vs_team_{ob}']
    sA = s_affine(a.affine); A2 = px_c_to_team(a.corr, sA, mesh_scale(a.mesh_a), mesh_scale(a.mesh_b))
    out['sA'] = sA; out['pxC_to_team_rc'] = A2.tolist()
    valid_c = resample_into_team((load(f'{a.c}/ct.png') > 0).astype(np.float32), A2, a.win, a.c_origin) > 0.99
    m = valid_b & valid_c; out['C_valid_frac'] = float(m.mean())
    C = {o: resample_into_team(load(f'{a.c}/maps/d0_{o}.png') / 255, A2, a.win, a.c_origin) for o in ('fwd', 'rev')}
    for o in C:
        out[f'C_r_vs_B_{o}'] = pearson(C[o], B[ob], m); out[f'C_r_vs_team_{o}'] = pearson(C[o], team, m)
    oc = max(C, key=lambda o: -9 if np.isnan(out[f'C_r_vs_B_{o}']) else out[f'C_r_vs_B_{o}'])
    out['C_order'] = oc; out['C_r'] = out[f'C_r_vs_B_{oc}']
    out['ct_r_B_vs_C'] = pearson(load(f'{a.b}/ct.png'), resample_into_team(load(f'{a.c}/ct.png'), A2, a.win, a.c_origin), m)
    out['known_answer'] = {'B_r>=0.8': bool(out['B_r'] >= 0.8), 'C_r>=0.6': bool(out['C_r'] >= 0.6)}
    json.dump(out, open(f'{a.out}/kat.json', 'w'), indent=1); print(json.dumps(out, indent=1))
    ds = 3
    tiles = [team, B[ob], C[oc], load(f'{a.b}/ct.png') / 255.0]
    sheet = np.concatenate([np.pad(np.clip(t[::ds, ::ds], 0, 1), ((0, 0), (0, 8)), constant_values=1) for t in tiles], 1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(f'{a.out}/sheet.png')


def main(argv=None):
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('window', 'compare'):
        p = sp.add_parser(name)
        for k in ('--corr', '--affine', '--mesh-a', '--mesh-b'): p.add_argument(k, required=True)
        p.add_argument('--win', type=int, nargs=4, required=True)
    sp.choices['window'].add_argument('--margin', type=int, default=200)
    c = sp.choices['compare']
    for k in ('--b', '--c', '--team', '--out'): c.add_argument(k, required=True)
    c.add_argument('--c-origin', type=int, nargs=2, required=True, help='C render crop x0 y0')
    a = ap.parse_args(argv)
    if a.cmd == 'window':
        window(a)
    else:
        import os; os.makedirs(a.out, exist_ok=True); compare(a)


if __name__ == '__main__':
    main(sys.argv[1:])
