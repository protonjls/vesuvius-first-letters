#!/usr/bin/env python3
"""Pixel AUC / AP of ink maps against an ink label inside a certainty mask (same pixel grid).

  label_auc.py --labels inklabels.png --mask mask.png --map NAME=path [--map ...] [--out out.json] [--stride 2]

label > 127 = ink; mask != 0 = scored; maps may be .npy (float) or .png/.tif (uint8). Pixels where the map is
not finite are dropped. AUC is the rank AUC (ties 0.5); AP is average precision; FPR@R50 is the fraction of
labelled-blank pixels at or above the median score of labelled-ink pixels (false ink at 50% ink recall).
"""
import argparse
import json

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(p):
    if p.endswith(".npy"):
        return np.squeeze(np.load(p)).astype(np.float32)
    if p.endswith((".tif", ".tiff")):
        import tifffile
        return np.squeeze(tifffile.imread(p)).astype(np.float32)
    return np.asarray(Image.open(p).convert("L"), np.float32)


def auc_ap(score, y):
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], y[order]
    # AP
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    ap = float((prec * t).sum() / max(1, t.sum()))
    # rank AUC with ties
    asc = np.argsort(score, kind="mergesort")
    sv = score[asc]
    uniq, start, counts = np.unique(sv, return_index=True, return_counts=True)
    ranks = np.empty(len(sv))
    ranks[asc] = np.repeat(start + (counts + 1) / 2.0, counts)
    npos, nneg = y.sum(), len(y) - y.sum()
    auc = float((ranks[y.astype(bool)].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--map", action="append", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    L = load(a.labels) > 127
    K = load(a.mask) > 0
    res = {}
    for spec in a.map:
        name, path = spec.split("=", 1)
        M = load(path)
        resized = None
        if M.shape != L.shape:  # e.g. a map made on a resampled (9.6 um) grid: same area, bilinear back
            resized = list(M.shape)
            M = np.asarray(Image.fromarray(M, mode="F").resize((L.shape[1], L.shape[0]), Image.BILINEAR))
        s = a.stride
        sel = (K & np.isfinite(M))[::s, ::s]
        sc, y = M[::s, ::s][sel], L[::s, ::s][sel].astype(np.int64)
        auc, apv = auc_ap(sc, y)
        # false ink: fraction of labelled-blank pixels above the threshold that recovers half of the labelled ink
        thr50 = float(np.quantile(sc[y == 1], 0.5))
        fpr50 = float((sc[y == 0] >= thr50).mean())
        res[name] = {"path": path, "resized_from": resized, "auc": auc, "ap": apv, "fpr_at_r50": fpr50,
                     "n": int(len(y)), "pos_frac": float(y.mean())}
        print(f"{name:40s} AUC {auc:.3f}  AP {apv:.3f}  FPR@R50 {fpr50:.3f}  n {len(y)}  pos {y.mean():.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
