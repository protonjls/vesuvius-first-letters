# herculex: second model and false-positive controls on a HerculeX piece (CPU only)

[HerculeX](https://github.com/adambetka2/HerculeX) (MIT, Adam Betka) unwraps a whole scan into pieces
(tifxyz surfaces) and inks each piece with reader-v3. Its whole-scan mosaics are a fast way to look for First Letters
candidates. A HerculeX ink image shows one model on one face, contrast-stretched. Before you trust forms in it, three
checks help: does an independent model show the same forms on the same surface? Do the forms persist a voxel or two
into the sheet, and vanish 15 voxels off it and under depth shuffling? Is the piece on the same sheet as a reference
surface, such as a claim mesh or a regrown patch? This kit runs those checks on a single piece.

| script | what it does |
|---|---|
| `piece_confirm.sh` | Renders the whole piece with VC3D `vc_render_tifxyz` at scale 1 (render pixel = grid cell × stride, stride 16 for HerculeX pieces) and runs [`../followup/confirm_region.sh`](../followup/) on it. That script builds the control variants (window centred at d−2..d+2, 15 voxels off the sheet on either side, depth-shuffled), runs reader-v3 and rectina-m4 in both depth orders on every variant, and writes persistence statistics and a montage. |
| `piece_sheet.py` | Review sheet: [HerculeX ink | CT | each model at d0] plus the first model on the controls. Aligns the HerculeX images to the render: it crops their 2-cell padding and tries 4 flips. It reports r of the HerculeX CT face against our CT (geometry check) and of the HerculeX ink against our reader-v3 in both orders (instrument check). |
| `place_check.py` | Normal distance from the nodes of a reference mesh, inside a row/col box, to a candidate mesh: overlap share, median/p10/p90, share ≤ 5 voxels, and the candidate cells and render box the overlap maps to. Also prints the nearest-node distance, to show why that measure misleads (below). |
| `demo_306b.sh` | Known-answer demo on the awarded PHerc0306B line (below). |
| `tests/test_place_check.py` | Synthetic known answers for `place_check.py`: same sheet, next sheet, tilted plane. |

**Why normal distance.** tifxyz nodes are often 16–20 voxels apart. Two meshes of the same sheet therefore read about
half a node spacing apart by nearest-node distance. We measured a median of 8.5–8.9 voxels on mesh pairs whose normal
distance was 0.3–0.7. Sheet pitch is about 10–15 voxels in many scrolls, so nearest-node distance cannot tell "same
sheet" from "next sheet".

## Known-answer demo: PHerc0306B
`demo_306b.sh` runs on one HerculeX piece shipped in [`../results/herculex-306b/`](../results/herculex-306b/):
- **The piece:** PHerc0306B scan 20250521133212, band rows 30–42, piece 7000024, with its CT face and 'taught' ink images.
- **How it was made:** HerculeX PR#1 `0d73c4c` with our band/shard patches, on CPU.
- **Why this piece:** it holds part of the awarded first line.

The demo runs `piece_confirm.sh` on the whole piece, then `place_check.py` against the winners' claim mesh (fetched at
commit `feabff6c`, CC BY-NC 4.0), then `piece_sheet.py`. Results from a clean checkout (CPU, 16 cores, 31 min):

![HerculeX ink, CT, reader-v3, rectina-m4 (top); reader-v3 at d−2, d+2, off−15, off+15, shuf (bottom)](../results/herculex-306b/sheet.jpg)

Top row of the sheet, as rendered (not the reading orientation): HerculeX's ink, the CT, reader-v3 and rectina-m4.
The awarded line is at the top in all three ink maps. Bottom row: reader-v3 at d−2 and d+2, where the line persists,
then 15 voxels below and above the sheet and depth-shuffled, where it is gone. The red box is where the winners' claim
box falls on the piece.

- **Same surface.** 52% of the winners' claim-box nodes (rows 82–137, cols 121–341 of their `V5Rc` mesh) lie within
  25 voxels of the piece. Over that overlap the normal distance is median 2.75 voxels (p10 0.46, p90 7.13), and 77% of
  nodes are within 5 voxels. Nearest-node distance reads 7.54 for the same nodes.
- **Our render reproduces HerculeX's ink.** HerculeX's CT face image matches our CT render without a flip (r 0.651;
  the other flips give ≤ 0.05). HerculeX's ink matches our reader-v3 at r 0.842 in **reverse** depth order and 0.04 in
  forward order. Run both orders on HerculeX pieces; `piece_sheet.py` picks the matching one.
- **Second model and controls.** rectina-m4 agrees with reader-v3 at r 0.899 (reverse order, whole piece). Pearson r
  of each variant's map with the d0 map (reverse order, whole piece 2304×2064 px, `controls_stats.json`):

| variant | reader-v3 r | rectina-m4 r |
|---|---|---|
| window centre d−2 / d−1 | 0.954 / 0.984 | 0.952 / 0.982 |
| window centre d+1 / d+2 | 0.976 / 0.910 | 0.974 / 0.914 |
| 15 voxels off the sheet, −15 / +15 | 0.186 / 0.026 | 0.218 / 0.025 |
| depth-shuffled | 0.037 | 0.008 |

The montage in `confirm/` uses forward order (`ORDER`); `stats.json` holds both orders.

A caution from our use on other pieces: the 'taught' image is contrast-stretched and saturated. It makes forms look
crisper than the raw model maps do. Judge letter forms from the raw maps on the sheet.

## Your own piece
```bash
# a HerculeX piece: <HERCULEX_OUTPUTS>/unwrapping/level/<scroll>/<scan>/pieces/<unit>/piece-<n>/
# its images:       <HERCULEX_OUTPUTS>/unwrapping/level/<scroll>/<scan>/faces/level-0/<unit>/piece-<n>{,-taught}.png
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... \
  bash herculex/piece_confirm.sh out/p <piece dir> <CT zarr URL> <pixel um>
python3 herculex/piece_sheet.py out/p --out out/p/sheet.png --ct-face <faces>/piece-<n>.png --ink-face <faces>/piece-<n>-taught.png
python3 herculex/place_check.py <reference tifxyz> <piece dir> --rows r0:r1 --cols c0:c1
```
Use `VSEL="d0"` for a quick two-model check, then run all variants on the pieces that pass. On 16 CPU cores a piece of
about 2000×2000 px takes 3–6 min per model per variant, after a render of under a minute. Environment: as
[`../followup/`](../followup/) (VC3D, ink-sdk, koine Python with numpy, scipy, PIL and tifffile).

## Stitching the blocks of a source surface
HerculeX cuts each large source surface into pieces of at most about 2 cm, the "blocks" of that source (piece ids
`source + k × 1000000`). The blocks of one source share one node grid: `meta.json` `block` = [row0, col0] is the
piece's node offset in it, and neighbouring blocks overlap by a few border nodes. A row of text longer than a block is
split across pieces, so a single piece can hold only part of a line.

| script | what it does |
|---|---|
| `stitch_blocks.py` | Places every piece of one source at its block offset in one grid and writes one tifxyz (x/y/z.tif, `meta.json`, `stitch_report.json`). Where pieces overlap it keeps the first valid node and reports the disagreement with each later piece (median and max \|dxyz\| in voxels). For blocks of one source this is 0 by construction; a nonzero value means the pieces do not share a grid. It refuses pieces whose `source`, `stride` or `scale` differ. |
| `stitch_sheet.py` | Review sheet for a `../followup/confirm_region.sh` window on the stitched surface: CT, each model at d0, the first model off the sheet and depth-shuffled. Block boundaries are dashed and the reference box is red. `--order auto` uses the depth order with the larger d0 contrast, the same rule as `piece_sheet.py`. |
| `stitch_demo_306b.sh` | Known-answer demo on the awarded PHerc0306B line (below). |
| `tests/test_stitch_blocks.py` | Synthetic known answers: blocks cut from one grid reassemble exactly; pieces of different sources are refused; a block moved by a known amount is reported at that size; command-line round trip. |

```bash
python3 herculex/stitch_blocks.py --out out/src24 <pieces>/piece-24 <pieces>/piece-1000024 ...   # all blocks of one source
```

**Known answer: PHerc0306B, source 24.** The demo stitches all 9 written blocks of HerculeX source surface 24, from
the same HerculeX run as the piece above, shipped in [`../results/herculex-306b-stitch/blocks/`](../results/herculex-306b-stitch/blocks/).
It place-checks the stitched surface against the winners' claim box, renders the claim box plus a 128 px margin, and
runs reader-v3 and rectina-m4 with controls on it. Results from a clean checkout (CPU, 16 cores; the demo took 1.5 h and 2.8 h on two nodes):

- **Stitch.** 9 blocks give one 225 × 531 node grid (82,325 valid nodes). Every overlap agrees to 0.0 voxels.
- **Same surface.** The stitched surface reaches 98.5% of the winners' claim-box nodes, against 52% for the single
  piece 7000024. Over that overlap the normal distance is median 3.04 voxels (p10 0.55, p90 8.9), and 73% of nodes are
  within 5 voxels. This is about as close as the single piece over its smaller overlap (2.75 voxels, 77%). We set a
  bar of median ≤ 3.0 and ≥ 75% within 5 voxels before the run, and the stitched surface narrowly misses it.
- **Text across a block boundary.** In reverse depth order both models show the awarded line, and it runs across the
  boundary between blocks 6000024 and 7000024 (dashed line at window x 1008). Control r against d0 (window
  4752 × 1744 px, reverse order): 15 voxels off the sheet 0.062 (reader-v3) and 0.045 (rectina-m4); depth-shuffled 0.030 and 0.054. Reader-v3 and rectina-m4
  agree at r 0.877. Forward order shows no line (contrast p99−p50 79 against 123 in reverse).

![CT, reader-v3, rectina-m4 (top); reader-v3 15 voxels off the sheet and depth-shuffled (bottom); dashed: block boundaries; red: winners' claim box](../results/herculex-306b-stitch/sheet.jpg)

## Reproduce
```bash
python3 herculex/tests/test_place_check.py                                              # 3 synthetic tests
python3 herculex/tests/test_stitch_blocks.py                                            # 4 synthetic tests
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash herculex/demo_306b.sh out/hx306b   # CPU, ~30 min
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash herculex/stitch_demo_306b.sh out/st306b   # CPU, 1.5–3 h
```
The stitch demo outputs in `results/herculex-306b-stitch/` (`summary.txt`, `stitch.txt`, `stitch_report.json`,
`place_check.json`, `window.txt`, `controls_stats.json`, `sheet.json`, `sheet.jpg`) are from a clean checkout of these scripts (paths shown as `<checkout>` and `<out>`).
The demo outputs in `results/herculex-306b/` (`summary.txt`, `controls_stats.json`, `place_check.json`,
`sheet.json`, `sheet.jpg`) are from a clean checkout of these scripts.

Data: the piece and its images derive from the PHerc0306B CT (Vesuvius Challenge open data, CC BY-NC 4.0) and are
shared under the same license. HerculeX: MIT.
