# vesuvius-first-letters

Open tools and measured results for ink detection on Herculaneum scroll CT
([Vesuvius Challenge](https://scrollprize.org)). Each result here was scored on real known text: PHerc1447 windings
with human ink labels, and the PHerc0343 First Letters region with its published letter boxes. The scripts that
produced each number are in this repository.

## Contents (v0.8, 2026-11)
| directory | what it is | headline result |
|---|---|---|
| [`scroll_adapt/`](scroll_adapt/) | Kit for fine-tuning reader-v3 on one scroll from consensus machine pseudo-labels, with no human labels (villa `koine_machines`). Includes the label-zarr layout, weights-only init, an exact no-op control and a fixed pass bar. | **Negative on PHerc1447.** Two runs gave held-out AUC changes of +0.005/−0.020 and +0.003/−0.020 (w058/w060) against base reader-v3. Self-training on agreed pixels did not help. |
| [`tta/`](tta/) | Test-time augmentation of reader-v3 measured on known text: koine's `--tta-mirror` and depth-window averaging. | Mirror TTA: +0.014 / +0.008 / +0.009 AUC (1447 w058, w060, 343 letter boxes) for about 1.5x the inference time. Depth averaging: ≤0.002. |
| [`eval/`](eval/) | Scorers used above: pixel AUC/AP/false-positive rate against ink labels, letter-box AUC, and single-model and ensemble tables. | Baselines on 1447 w058/w060 (reverse order): reader-v3 0.849/0.774, rectina-m4 0.857/0.782, v8-in 0.862/0.810, v3 × v8-in product 0.880/0.826. See `results/ensembles_1447_*.txt`. |
| [`followup/`](followup/) | CPU kit for following up a spot: grow a local surface from a seed with VC3D GrowPatch on the public m7 predictions, render it, run reader-v3/rectina-m4 in both depth orders, and run false-positive controls (depth offsets, off-sheet, depth shuffle). Also resamples a map between two meshes of one sheet. | **Known answer: PHerc0306B.** From the winners' published seed, public reader-v3 alone shows their awarded first line at the same positions as their claim image. The two surfaces lie a median 0.5 voxels apart. |
| [`tiling/`](tiling/) | CPU kit that covers every sheet crossing a slice window with VC3D GrowPatch patches. Seeds come from the public surface prediction, with no umbilicus and no hand picks. Each patch is masked to its on-sheet part and covered seeds are skipped. The kit then renders the patches, runs reader-v3, ranks 4 cm² windows and lays out every patch on anonymised contact sheets for a full review. Includes coverage tools and synthetic tests. | **Known answer: PHerc0306B.** In an 1800×1800-voxel window (827 seeds, 228 grows), two automatic patches lie on 82% of the winners' claim surface, at a median of 0.26 voxels. Reader-v3 on one of them shows the awarded line (r 0.86 against the v0.2 reference). Its window ranks 4th of 896, and a blind full-sheet review graded it strong; ranked windows alone reach only 41% of the tiled surface. **Negative on crushed PHerc0211:** patches reach the award arms but sit about 1.3 voxels off the hand-refined surface, and the letters do not resolve. |
| [`herculex/`](herculex/) | CPU kit for checking forms in a [HerculeX](https://github.com/adambetka2/HerculeX) unwrap on the piece itself. It renders a HerculeX piece with VC3D and runs a second model (rectina-m4) next to reader-v3, in both depth orders. It adds false-positive controls (d±1/±2, 15 voxels off the sheet, depth shuffle) and a review sheet that lines up HerculeX's own CT and ink images with the render. A normal-distance place check tests whether two meshes are on the same sheet. `stitch_blocks.py` joins the blocks of one HerculeX source surface into one mesh, so a text line split across pieces can be read whole. | **Known answer: PHerc0306B.** A HerculeX piece lies a median 2.75 voxels from the winners' claim surface; nearest-node distance reads 7.5. Our render and reader-v3 reproduce HerculeX's ink on it (r 0.84, reverse order only). Rectina-m4 agrees (r 0.90). The awarded line persists at d±2 (r ≥ 0.91) and is gone off the sheet (r ≤ 0.22) and under shuffling (r ≤ 0.04). The 9 blocks of that piece's source stitch with 0.0-voxel overlap disagreement. They reach 98.5% of the claim box (one piece: 52%), and the line runs across a block boundary in both models. |
| [`crosscheck/`](crosscheck/) | Mesh-to-mesh check that ink forms are a property of the papyrus, not of one mesh. Each pixel of mesh A is matched to the same surface point on mesh B (nearest node refined on B's tangent plane, normal-distance gate). B's ink map is resampled into A's frame and the Pearson r is reported for every model and depth-order pair, with a shifted negative control. No hand alignment and no annotation. Includes synthetic tests. | **Known answer: PHerc0306B.** A HerculeX piece and an independent GrowPatch mesh of the awarded line lie a median 3.0 voxels apart. Reader-v3 maps correlate at r 0.685 (rectina-m4 0.736) in one order pair. With B shifted by half a letter, r falls to 0.12/0.11. |
| [`rescan/`](rescan/) | Runs the Vesuvius Challenge team's canonical 2.4 µm ink model (`canonical_2um`, via ink-sdk) on a 2.4 µm rescan along any tifxyz mesh. The mesh can come from the scroll's 9.36 µm volume: it is mapped into the rescan by the team's official transform. Both depth orders are run. Includes a node correspondence between two meshes of one sheet, a comparison against a published map, and synthetic tests. | **Known answer: PHerc0139.** On the segment's own 2.4 µm mesh our map matches the team's published canonical map at r 0.940. The segment's 9.36 µm mesh, mapped through the official transform, gives the same map (r 0.931 against it, 0.929 against the team map). Only the reverse depth order matches. |
| [`isoregrid/`](isoregrid/) | CPU tool that re-grids a tifxyz surface isometrically: a mass-spring flatten of the nodes, then barycentric re-sampling on a regular grid. Spiral-fit windings (villa `fit_spiral`) are regular in (angle, z), so their column edges vary from about 16 to 44 voxels in 3D (10th to 90th percentile on PHerc0211), and renders squash the text. The tool reports the input and output distortion \|edge3D/step − 1\|, a number for the low-distortion isometric flattening that a First Letters package needs. Includes a synthetic column squash, a normal-distance check and synthetic tests. | **Known answer: PHerc0211.** The awardees' v9 mesh stays isometric (distortion median/p95 0.006/0.038) and on its surface (normal distance median 0.18 voxels). A synthetic squash goes from p95 0.594 back to 0.033. A raw spiral-fit winding of the awarded region goes from 0.109/1.256 to 0.006/0.051. In the project's own ink pipeline (not reproduced by the demo), the re-gridded winding's reader-v3 + rectina-m4 map matches the map on v9 at r 0.912, against 0.741 for the raw winding. That is map agreement, not a letter count: a blind count read no letters on the re-gridded winding's maps, nor on a HerculeX surface's maps of the same row. |
| [`docs/`](docs/) | `koine_finetune_notes.md`: how koine_machines fine-tuning works, with code citations (dataset layout, label zarrs, checkpoint loading, loss masking, inference parity). `environment.md`: how to build the environment. | |
| [`results/`](results/) | Raw outputs of every run reported above: scores, configs and run info. | |

## Who this helps
- Teams fine-tuning reader-v3 or other koine models on new scrolls. `docs/koine_finetune_notes.md` and
  `scroll_adapt/` avoid traps that fail silently. One example: villa's label-zarr helper puts labels only in slice 32,
  so on a 24-layer volume every pixel is ignored. The no-op control proves a fine-tune pipeline adds no drift before
  you trust its gains.
- Teams running reader-v3 surveys: one flag (`--tta-mirror`) gave a small, consistent AUC gain on three held-out
  known-text tests.
- Teams with a candidate spot from a sweep: `followup/` grows a local surface around it, maps it with public
  models and runs the false-positive controls, on CPU only. It was checked on the awarded PHerc0306B line.
- Teams covering more of a scroll than spiral-fit windings give: `tiling/` grows local surfaces for every sheet of a
  window automatically and says how much of a reference surface they hit. The PHerc0306B known-answer run shows that it
  finds the awarded surface and line. The PHerc0211 run shows its limit in crushed regions.
- Teams searching HerculeX whole-scan mosaics: `herculex/` checks a candidate on the piece itself with a second model and
  controls, finds the depth order that matches HerculeX's ink, and measures whether a regrow or claim mesh is on the same
  sheet by normal distance rather than nearest-node distance. It also joins the blocks of a source surface, so a text
  line that HerculeX split across pieces can be checked as one line.
- Anyone with two meshes of one place (two unwrapping methods, two seeds, or a regrow of a small piece):
  `crosscheck/` measures whether the same ink forms reappear at the same places, with a shifted negative control.
- Anyone with a 9.36 µm segment of a scroll that also has a 2.4 µm rescan: `rescan/` shows what the team's canonical
  2.4 µm model sees along that exact surface, without retracing the sheet in the rescan. It also gives 2.4 µm-model maps
  in the 9 µm surface's own frame, for anyone distilling them into 9 µm models.
- Anyone rendering ink on spiral-fit windings or other non-isometric tifxyz grids: `isoregrid/` re-grids the surface so
  the grid does not stretch letters, and reports the distortion of the flattening a First Letters package needs.
- Anyone considering self-training from public-model consensus: here is a measured negative result, with the
  pseudo-label quality that produced it.

## Reproduce
Build the environment (`docs/environment.md`), then:
```bash
scroll_adapt/fetch_1447.sh $DATA
PY=... V3=reader-v3.pth RECT=ckpt_020000.pth DATA=$DATA OUT=out/adapt bash scroll_adapt/run_adapt.sh
PY=... CKPT=reader-v3.pth DATA=$DATA OUT=out/tta bash tta/tta_eval.sh
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash followup/demo_306b.sh out/demo306b   # CPU, about 3 h
python3 tiling/tests/test_tiling.py                                                       # CPU, about 1 min
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash tiling/demo_306b.sh out/tiling306b     # CPU, about 40 min
python3 herculex/tests/test_place_check.py                                                # CPU, seconds
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash herculex/demo_306b.sh out/hx306b       # CPU, about 30 min
python3 herculex/tests/test_stitch_blocks.py                                              # CPU, seconds
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash herculex/stitch_demo_306b.sh out/st306b # CPU, 1.5–3 h
python3 crosscheck/tests/test_reappear.py                                                 # CPU, seconds
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash crosscheck/demo_306b.sh out/cc306b     # CPU, about 10 min
python3 rescan/tests/test_rescan.py                                                       # CPU, seconds
VC3D=... INKSDK=... KPY=... VILLA_SRC=... MODELS_DIR=... bash rescan/demo_0139.sh out/rescan0139    # GPU, a few minutes
python3 isoregrid/tests/test_iso_regrid.py                                                # CPU, seconds
PY=python3 bash isoregrid/demo_0211.sh out/iso0211                                        # CPU, about 2-5 min
```
Each run takes about 30 min on one MIG 2g.20gb H100 slice.

## Credits and licenses
Code: MIT (this repository). It builds on [ScrollPrize/villa](https://github.com/ScrollPrize/villa)
(`koine_machines`, VC3D), reader-v3 by DomRusso2, rectina-m4 and v8-in as packaged by
[LimeGS/ink-sdk](https://github.com/LimeGS/ink-sdk), and the PHerc1447 surfaces and refined labels of
[YoussefMoNader/ink-8um-pherc1447-surfaces](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)
(CC BY-NC 4.0, not redistributed here). The PHerc0343 test uses the mesh and letter boxes of
[Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters). The `herculex/` demo uses a piece made with
[HerculeX](https://github.com/adambetka2/HerculeX) (MIT, Adam Betka) and the PHerc0306B claim mesh of
[SirProdigle/pherc0306b-first-letters](https://github.com/SirProdigle/pherc0306b-first-letters); the `crosscheck/` demo reuses that piece. The `rescan/` kit runs the Vesuvius Challenge team's
canonical 2.4 µm model as packaged by ink-sdk; its demo compares with the team's published PHerc0139 canonical map and
uses their official transform. The `isoregrid/` demo uses the PHerc0211 First Letters awardees' hand-refined mesh
(open-data segment `20261028201525-crop1711-lines123-v9`) and ships one villa `fit_spiral` winding of that region. Scroll data: Vesuvius Challenge open data.

This repository was built by an AI agent (Claude, Anthropic) working for the repository owner. Treat every result as
a proposal to verify: the scripts and raw outputs are here so you can.
