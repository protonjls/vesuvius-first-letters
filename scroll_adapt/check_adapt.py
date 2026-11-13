#!/usr/bin/env python3
"""Bar for the scroll-adapt release (fixed 2026-11-13, before any fine-tune result existed).

  check_adapt.py --base NAME --adapted NAME score_w058.json score_w060.json [--teacher NAME]

Each JSON is eval/label_auc.py output on a held-out labelled winding (PHerc1447 w058/w060 refined labels).
PASS iff on EVERY file: AUC(adapted) - AUC(base) >= 0.01 and FPR@R50(adapted) <= FPR@R50(base) + 0.005.
The teacher (pseudo-label source, e.g. the v3 x rect product) is reported, not gated.
"""
import argparse, json, sys

ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--adapted', required=True)
ap.add_argument('--teacher'); ap.add_argument('scores', nargs='+'); a = ap.parse_args()
ok = True
for f in a.scores:
    r = json.load(open(f)); b, d = r[a.base], r[a.adapted]
    dauc = d['auc'] - b['auc']; dfpr = d.get('fpr_at_r50', float('nan')) - b.get('fpr_at_r50', float('nan'))
    good = dauc >= 0.01 and dfpr <= 0.005
    ok &= good
    t = f" teacher AUC {r[a.teacher]['auc']:.3f}" if a.teacher and a.teacher in r else ''
    print(f"{f}: base AUC {b['auc']:.3f} adapted {d['auc']:.3f} (d {dauc:+.3f}); FPR@R50 d {dfpr:+.3f}{t} -> {'ok' if good else 'FAIL'}")
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
