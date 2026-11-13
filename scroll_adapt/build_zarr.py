#!/usr/bin/env python3
"""Training inputs for koine_machines flat-mode fine-tuning (docs/koine_finetune_notes.md s1).

  build_zarr.py vol   <layers_dir|stack.tif|in.zarr> <out.zarr> [--flip] [--keep N]
      layers_dir = 00.tif..NN.tif (one uint8 layer per normal offset). Writes a zarr v2 group with array "0",
      uint8 [Z,H,W], chunks (Z,512,512). --flip reverses Z (use when the reading side is the "reverse" order, so
      training, which reads Z ascending, sees the reading side). --keep N keeps the central N layers.
  build_zarr.py label <ink.npy> <sup.npy> <segments_root> <seg> --z Z
      Writes <segments_root>/<seg>/<seg>_inklabels.zarr and _supervision_mask.zarr: uint8 0/1 [Z,H,W], the 2D map
      in every Z slice (never villa's create_label_zarrs.py: it fills only slice 32, docs/koine_finetune_notes.md s1).
"""
import argparse, glob, os
import numpy as np
import zarr
from numcodecs import Blosc

COMP = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def write(path, arr):
    g = zarr.open_group(path, mode="w")
    g.create_dataset("0", data=arr, chunks=(arr.shape[0], 512, 512), compressor=COMP, dtype="u1")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("vol"); v.add_argument("src"); v.add_argument("out"); v.add_argument("--flip", action="store_true")
    v.add_argument("--keep", type=int)
    l = sp.add_parser("label"); l.add_argument("ink"); l.add_argument("sup"); l.add_argument("root"); l.add_argument("seg")
    l.add_argument("--z", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "vol":
        if os.path.isdir(a.src) and not a.src.rstrip("/").endswith(".zarr"):
            import tifffile
            files = sorted(glob.glob(os.path.join(a.src, "*.tif")))
            arr = np.stack([tifffile.imread(f) for f in files]).astype(np.uint8)
        elif a.src.rstrip("/").endswith(".zarr"):
            z = zarr.open(a.src, mode="r"); z = z["0"] if hasattr(z, "keys") else z
            arr = np.asarray(z[:]).astype(np.uint8)
        else:
            import tifffile
            arr = tifffile.imread(a.src).astype(np.uint8)
        if a.keep:
            c = arr.shape[0] // 2; arr = arr[c - a.keep // 2: c - a.keep // 2 + a.keep]
        if a.flip:
            arr = arr[::-1]
        arr = np.ascontiguousarray(arr)
        write(a.out, arr)
        print(f"vol {a.src} -> {a.out} shape {arr.shape} flip {a.flip}")
    else:
        ink = np.load(a.ink) > 0; sup = np.load(a.sup) > 0
        assert ink.shape == sup.shape
        d = os.path.join(a.root, a.seg); os.makedirs(d, exist_ok=True)
        for name, m in (("inklabels", ink), ("supervision_mask", sup)):
            write(os.path.join(d, f"{a.seg}_{name}.zarr"), np.broadcast_to(m[None].astype(np.uint8), (a.z,) + m.shape))
        print(f"labels {a.seg}: Z {a.z} HxW {ink.shape} ink {ink.mean():.4f} sup {sup.mean():.4f}")


if __name__ == "__main__":
    main()
