#!/usr/bin/env python3
"""Stitch the blocks of one HerculeX source surface back into one tifxyz surface.

  stitch_blocks.py --out <dir> <piece dir> [<piece dir> ...]

HerculeX cuts each large source surface into pieces of at most about 2 cm (piece ids source + k * 1000000). The pieces
of one source share one node grid: meta.json 'block' = [row0, col0] is the piece's node offset in that grid and
'overlap' gives the border nodes it shares with its neighbours. A row of text longer than a piece is therefore split
across pieces. This script places every piece at its block offset in one grid. Where pieces overlap, the first valid
node is kept and the disagreement with each later piece is reported (median and max |dxyz| in voxels over nodes valid
in both). For blocks of one source this is 0 by construction; a nonzero value means the pieces do not come from one
grid and the stitch should not be trusted. Invalid nodes are -1 (HerculeX convention).

All pieces must have the same 'source', 'stride' and 'scale'. Writes x.tif y.tif z.tif (float32) and meta.json (the first
piece's meta with bbox and node count recomputed, uuid 'stitch-src<source>', 'stitched_from' = pieces and blocks) plus
stitch_report.json (grid shape, per piece: block, shape, valid and added nodes, overlap statistics).
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile


def load_piece(d):
    m = json.load(open(os.path.join(d, 'meta.json')))
    G = np.stack([tifffile.imread(os.path.join(d, f'{c}.tif')) for c in 'xyz'], -1).astype(np.float32)
    return m, G


def stitch(pieces):
    """pieces: list of (name, meta, grid (h, w, 3)). Returns (stitched grid, report dict)."""
    for key in ('source', 'stride', 'scale'):
        vals = {json.dumps(m.get(key)) for _, m, _ in pieces}
        if len(vals) != 1:
            raise ValueError(f'pieces differ in {key}: {sorted(vals)}')
    H = max(m['block'][0] + G.shape[0] for _, m, G in pieces)
    W = max(m['block'][1] + G.shape[1] for _, m, G in pieces)
    C = np.full((H, W, 3), -1, np.float32)
    rep = {'shape': [H, W], 'source': pieces[0][1].get('source'), 'pieces': []}
    for name, m, G in pieces:
        r0, c0 = m['block']
        h, w = G.shape[:2]
        sub = C[r0:r0 + h, c0:c0 + w]
        vg = G[..., 2] > 0
        vc = sub[..., 2] > 0
        both = vg & vc
        dd = np.linalg.norm(sub[both] - G[both], axis=-1) if both.any() else np.zeros(0)
        put = vg & ~vc
        sub[put] = G[put]
        rep['pieces'].append({'dir': name, 'piece': m.get('piece'), 'block': [int(r0), int(c0)], 'shape': [h, w],
                              'valid': int(vg.sum()), 'added': int(put.sum()), 'overlap_nodes': int(both.sum()),
                              'overlap_max_vox': float(dd.max()) if dd.size else None,
                              'overlap_median_vox': float(np.median(dd)) if dd.size else None})
    rep['valid_nodes'] = int((C[..., 2] > 0).sum())
    ov = [p['overlap_max_vox'] for p in rep['pieces'] if p['overlap_max_vox'] is not None]
    rep['overlap_max_vox'] = max(ov) if ov else None
    return C, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', required=True)
    ap.add_argument('pieces', nargs='+')
    a = ap.parse_args()
    items = [(d, *load_piece(d)) for d in a.pieces]
    try:
        C, rep = stitch(items)
    except ValueError as e:
        sys.exit(str(e))
    os.makedirs(a.out, exist_ok=True)
    for i, c in enumerate('xyz'):
        tifffile.imwrite(os.path.join(a.out, f'{c}.tif'), C[..., i])
    v = C[..., 2] > 0
    P = C[v]
    meta = dict(items[0][1])
    meta.update({'uuid': f"stitch-src{meta['source']}", 'piece': f"stitch-src{meta['source']}", 'block': [0, 0],
                 'nodes': int(v.sum()), 'bbox': [P.min(0).tolist(), P.max(0).tolist()],
                 'stitched_from': [{'piece': p['piece'], 'block': p['block']} for p in rep['pieces']]})
    for k in ('area_mm2', 'width_mm', 'height_mm', 'cells', 'isometry', 'locked_share', 'overlap'):
        meta.pop(k, None)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    json.dump(rep, open(os.path.join(a.out, 'stitch_report.json'), 'w'), indent=1)
    print(f"stitched {len(items)} pieces of source {rep['source']} -> grid {rep['shape'][0]}x{rep['shape'][1]}, "
          f"{rep['valid_nodes']} valid nodes; max overlap disagreement {rep['overlap_max_vox']} vox")
    for p in rep['pieces']:
        print(f"  piece {p['piece']} block {p['block']} shape {p['shape']} added {p['added']} "
              f"overlap {p['overlap_nodes']} nodes, median/max {p['overlap_median_vox']}/{p['overlap_max_vox']} vox")


if __name__ == '__main__':
    main()
