#!/usr/bin/env python3
"""arm_check.py <arm> <input mesh> <iso output mesh> <reference mesh> <out.json> --out-med M --out-p95 P --dn-med D [--in-p95-min Q]

One arm of demo_0211.sh: reads the input and output distortion (iso_regrid.py's iso_info.json in the output dir), measures
the normal distance of the output nodes to the reference surface (normal_dn.py), applies the arm's bars and writes one
JSON. Bars (all strict as written in the README): output distortion median < M and p95 < P, dn median <= D and, when
given, input distortion p95 >= Q (the input must really be distorted for the arm to test anything). Prints one line
ending in PASS or FAIL; exit status 0 either way (the demo decides).
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normal_dn import load, dn, summary  # noqa: E402
from iso_regrid import distortion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ('arm', 'inp', 'iso', 'ref', 'out'):
        ap.add_argument(k)
    ap.add_argument('--out-med', type=float, required=True); ap.add_argument('--out-p95', type=float, required=True)
    ap.add_argument('--dn-med', type=float, required=True); ap.add_argument('--in-p95-min', type=float, default=None)
    a = ap.parse_args()
    info = json.load(open(f'{a.iso}/iso_info.json'))
    out_meta = json.load(open(f'{a.iso}/meta.json'))
    G = load(a.iso)
    om, op, ne = distortion(G, 1 / out_meta['scale'][0])            # recomputed from the written files
    s, d, _ = dn(load(a.ref), G[G[..., 2] > 0])
    r = dict(arm=a.arm, input=os.path.basename(os.path.normpath(a.inp)), reference=os.path.basename(os.path.normpath(a.ref)),
             in_distortion_median=info['in_distortion'][0], in_distortion_p95=info['in_distortion'][1],
             out_distortion_median=om, out_distortion_p95=op, out_edges=ne,
             flipped_triangles=info['flipped_triangles'], triangles=info['triangles'], out_shape=info['out_shape'],
             out_valid=info['out_valid'], **{'dn_out_to_ref_' + k: v for k, v in summary(s, d).items()})
    bars = {f'out_distortion_median < {a.out_med}': om < a.out_med,
            f'out_distortion_p95 < {a.out_p95}': op < a.out_p95,
            f'dn_median <= {a.dn_med}': r['dn_out_to_ref_dn_median'] <= a.dn_med}
    if a.in_p95_min is not None:
        bars[f'in_distortion_p95 >= {a.in_p95_min}'] = r['in_distortion_p95'] >= a.in_p95_min
    r['bars'] = bars; r['pass'] = all(bars.values())
    json.dump(r, open(a.out, 'w'), indent=1)
    print(f"{a.arm}: input distortion {r['in_distortion_median']:.3f}/{r['in_distortion_p95']:.3f} -> output "
          f"{om:.3f}/{op:.3f} (median/p95); dn output->{r['reference']} median {r['dn_out_to_ref_dn_median']:.2f} "
          f"p90 {r['dn_out_to_ref_dn_p90']:.2f}; flipped {r['flipped_triangles']}/{r['triangles']}; "
          f"out {r['out_shape']} -> {'PASS' if r['pass'] else 'FAIL'}"
          + ('' if r['pass'] else ' (' + ', '.join(k for k, v in bars.items() if not v) + ')'))


if __name__ == '__main__':
    main()
