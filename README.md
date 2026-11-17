# vesuvius-first-letters

Open tools and measured results for ink detection on Herculaneum scroll CT
([Vesuvius Challenge](https://scrollprize.org)). Each result here was scored on real known text: PHerc1447 windings
with human ink labels, and the PHerc0343 First Letters region with its published letter boxes. The scripts that
produced each number are in this repository.

## Contents (v0.3.1, 2026-11)
| directory | what it is | headline result |
|---|---|---|
| [`scroll_adapt/`](scroll_adapt/) | Kit for fine-tuning reader-v3 on one scroll from consensus machine pseudo-labels, with no human labels (villa `koine_machines`). Includes the label-zarr layout, weights-only init, an exact no-op control and a fixed pass bar. | **Negative on PHerc1447.** Two runs gave held-out AUC changes of +0.005/−0.020 and +0.003/−0.020 (w058/w060) against base reader-v3. Self-training on agreed pixels did not help. |
| [`tta/`](tta/) | Test-time augmentation of reader-v3 measured on known text: koine's `--tta-mirror` and depth-window averaging. | Mirror TTA: +0.014 / +0.008 / +0.009 AUC (1447 w058, w060, 343 letter boxes) for about 1.5x the inference time. Depth averaging: ≤0.002. |
| [`eval/`](eval/) | Scorers used above: pixel AUC/AP/false-positive rate against ink labels, letter-box AUC, and single-model and ensemble tables. | Baselines on 1447 w058/w060 (reverse order): reader-v3 0.849/0.774, rectina-m4 0.857/0.782, v8-in 0.862/0.810, v3 × v8-in product 0.880/0.826. See `results/ensembles_1447_*.txt`. |
| [`followup/`](followup/) | CPU kit for following up a spot: grow a local surface from a seed with VC3D GrowPatch on the public m7 predictions, render it, run reader-v3/rectina-m4 in both depth orders, and run false-positive controls (depth offsets, off-sheet, depth shuffle). Also resamples a map between two meshes of one sheet. | **Known answer: PHerc0306B.** From the winners' published seed, public reader-v3 alone shows their awarded first line at the same positions as their claim image. The two surfaces lie a median 0.5 voxels apart. |
| [`tiling/`](tiling/) | CPU kit that covers every sheet crossing a slice window with VC3D GrowPatch patches. Seeds come from the public surface prediction, with no umbilicus and no hand picks. Each patch is masked to its on-sheet part and covered seeds are skipped. The kit then renders the patches, runs reader-v3, ranks 4 cm² windows and lays out every patch on anonymised contact sheets for a full review. Includes coverage tools and synthetic tests. | **Known answer: PHerc0306B.** In an 1800×1800-voxel window (827 seeds, 228 grows), two automatic patches lie on 82% of the winners' claim surface, at a median of 0.26 voxels. Reader-v3 on one of them shows the awarded line (r 0.86 against the v0.2 reference). Its window ranks 4th of 896, and a blind full-sheet review graded it strong; ranked windows alone reach only 41% of the tiled surface. **Negative on crushed PHerc0211:** patches reach the award arms but sit about 1.3 voxels off the hand-refined surface, and the letters do not resolve. |
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
```
Each run takes about 30 min on one MIG 2g.20gb H100 slice.

## Credits and licenses
Code: MIT (this repository). It builds on [ScrollPrize/villa](https://github.com/ScrollPrize/villa)
(`koine_machines`, VC3D), reader-v3 by DomRusso2, rectina-m4 and v8-in as packaged by
[LimeGS/ink-sdk](https://github.com/LimeGS/ink-sdk), and the PHerc1447 surfaces and refined labels of
[YoussefMoNader/ink-8um-pherc1447-surfaces](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)
(CC BY-NC 4.0, not redistributed here). The PHerc0343 test uses the mesh and letter boxes of
[Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters). Scroll data: Vesuvius
Challenge open data.

This repository was built by an AI agent (Claude, Anthropic) working for the repository owner. Treat every result as
a proposal to verify: the scripts and raw outputs are here so you can.
