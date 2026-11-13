#!/usr/bin/env python3
"""Positive-control score: how strongly an ink map marks known letters.

  pc_score.py --boxes letter_boxes.json --map NAME=path.npy [--map NAME=path.npy ...] [--valid valid.npy] [--out out.json]

Each map is a 2D float array in the RAW render frame (rows = mesh rows, cols = mesh cols), the frame the
letter boxes use ({"frame": "raw", "letters": [{"box": [x0, y0, x1, y1]}]}, x = column).
Per map, over valid pixels (map finite and > 0 unless --valid is given):
  z          = (map - mean) / sd
  letter_z   = mean z inside each letter box (boxes include some background, so this understates stroke contrast)
  auc_in_out = P(z at a random in-box pixel > z at a random out-of-box valid pixel) (rank AUC, ties 0.5)
  hit_frac   = fraction of letters with letter_z > 0.5
A map with no letter information gives auc ~0.5, letter_z ~0. The same boxes are used for every map.
"""
import argparse
import json
import sys

import numpy as np


def rank_auc(pos, neg):
    pos = np.asarray(pos, np.float64).ravel()
    neg = np.asarray(neg, np.float64).ravel()
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), np.float64)
    sv = allv[order]
    # average ranks for ties
    i = 0
    n = len(sv)
    r = np.arange(1, n + 1, dtype=np.float64)
    # vectorised tie averaging
    uniq, start, counts = np.unique(sv, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--map", action="append", required=True, help="NAME=path.npy|.tif")
    ap.add_argument("--valid", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-shape", default=None, help="H,W of the raw render; maps of another shape are resized")
    ap.add_argument("--subsample", type=int, default=4, help="pixel stride for the AUC (speed)")
    a = ap.parse_args()
    B = json.load(open(a.boxes))
    assert B.get("frame", "raw") == "raw", "boxes must be in the raw render frame"
    letters = B["letters"]
    res = {"boxes": a.boxes, "n_letters": len(letters), "maps": {}}
    for spec in a.map:
        name, path = spec.split("=", 1)
        if path.endswith(".npy"):
            M = np.load(path).astype(np.float32)
        elif path.endswith(".png"):
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            M = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
        else:
            import tifffile
            M = tifffile.imread(path).astype(np.float32)
        M = np.squeeze(M)
        resized = None
        if a.ref_shape:
            H, W = [int(v) for v in a.ref_shape.split(",")]
            if M.shape != (H, W):
                from PIL import Image
                resized = list(M.shape)
                M = np.asarray(Image.fromarray(M, mode="F").resize((W, H), Image.BILINEAR))
        valid = np.isfinite(M) & (M > 0) if a.valid is None else np.load(a.valid).astype(bool)
        mu, sd = float(M[valid].mean()), float(M[valid].std() + 1e-9)
        Z = (M - mu) / sd
        inbox = np.zeros(M.shape, bool)
        lz = []
        for L in letters:
            x0, y0, x1, y1 = [int(round(v)) for v in L["box"]]
            x0, x1 = max(0, x0), min(M.shape[1], x1)
            y0, y1 = max(0, y0), min(M.shape[0], y1)
            sub = valid[y0:y1, x0:x1]
            lz.append(float(Z[y0:y1, x0:x1][sub].mean()) if sub.any() else float("nan"))
            inbox[y0:y1, x0:x1] = True
        s = a.subsample
        pos = Z[::s, ::s][(inbox & valid)[::s, ::s]]
        neg = Z[::s, ::s][(~inbox & valid)[::s, ::s]]
        lz_arr = np.array(lz)
        res["maps"][name] = {
            "path": path, "shape": list(M.shape), "resized_from": resized, "valid_px": int(valid.sum()),
            "auc_in_out": rank_auc(pos, neg),
            "letter_z_mean": float(np.nanmean(lz_arr)), "letter_z_median": float(np.nanmedian(lz_arr)),
            "hit_frac": float(np.nanmean(lz_arr > 0.5)), "letter_z": lz,
        }
        r = res["maps"][name]
        print(f"{name:40s} auc {r['auc_in_out']:.3f}  letter_z mean {r['letter_z_mean']:+.2f} median {r['letter_z_median']:+.2f}  hit {r['hit_frac']:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
