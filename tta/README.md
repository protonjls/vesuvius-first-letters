# Test-time augmentation for reader-v3: measured on known text

**Question.** Does test-time augmentation improve reader-v3 ink maps on real known text? Two options were tested: the
`--tta-mirror` flag that koine_machines inference already has, and averaging maps over shifted 17-layer depth windows.

**Answer (reader-v3, villa `47a9c37f6`).** Mirror TTA improved every held-out test by +0.008 to +0.014 AUC and cost
about 1.5x the inference time. Depth-window averaging changed AUC by at most 0.002.

| arm | PHerc1447 w058 pixel AUC | PHerc1447 w060 pixel AUC | PHerc0343 letter-box AUC |
|---|---|---|---|
| A0: single centred 17-layer window | 0.855 | 0.785 | 0.756 |
| A1: A0 + `--tta-mirror` | **0.869** (+0.014) | **0.793** (+0.008) | **0.765** (+0.009) |
| A2: mean of window offsets -2..+2 | 0.857 | 0.781 | 0.758 |
| A3: mean of window offsets -1..+1 | 0.856 | 0.783 | 0.756 |

Single offsets, for information (w058 / w060 / 343):

| offset | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| w058 | 0.825 | 0.840 | 0.849 | 0.855 | 0.857 | 0.854 | 0.850 |
| w060 | 0.741 | 0.756 | 0.774 | 0.785 | 0.780 | 0.786 | 0.797 |
| 343 | 0.756 | 0.755 | 0.755 | 0.756 | 0.755 | 0.754 | 0.750 |

All numbers are in `results/tta-v3-a/` (`score_*.txt`, with per-map AP and false-positive rate at 50% ink recall).
Mirror TTA cut that false-positive rate on w058 (0.047 to 0.036) but raised it on w060 (0.101 to 0.107).

**Pre-registered bar, and how it came out.** Before the run, the bar for calling an arm a gain was fixed at
+0.01 AUC on *both* 1447 windings, with 343 not lower by more than 0.002. Mirror TTA reached +0.014 on w058 but
+0.008 on w060, so it misses that bar. The effect has the same sign on all three tests, but its size on w060 is
within the range we would call noise. Our recommendation is cautious: `--tta-mirror` costs little and did not lower
AUC on any test, so it is a reasonable default for reader-v3 survey runs. Do not expect it to make invisible text visible.

**Setup.**
- PHerc1447 w058/w060 are the 24-layer HF renders. Z is flipped so the reading side is "forward". Scores are pixel
  AUC against the refined human labels inside their certainty mask, using `eval/label_auc.py` on every 2nd pixel.
  reader-v3 never trained on PHerc1447.
- PHerc0343 is a 66-layer render of Nieuwlaar's winning concat mesh, read in "reverse". The score is the
  letter-box AUC from `eval/pc_score.py`: the probability that a pixel inside one of the 18 published letter boxes
  outscores a pixel outside them.
- Inference used `python -m koine_machines.inference.infer <zarr> reader-v3.pth out.tif --overlap 0.5 --blend-mode hann
  --batch-size 1 --amp-dtype bf16 --no-compile --layer-start s --layer-end s+17` (plus `--tta-mirror` for A1).

**Reproduce.** See `docs/environment.md`, then:
```bash
scroll_adapt/fetch_1447.sh $DATA
PY=... CKPT=reader-v3.pth DATA=$DATA OUT=out/tta bash tta/tta_eval.sh                 # 1447 only
PY=... CKPT=reader-v3.pth DATA=$DATA OUT=out/tta SURF="w058 w060 343" Z343=render.zarr \
   BOX343=concat_w047-w048_R5B2_z9500-11000_letter_boxes.json bash tta/tta_eval.sh  # with 343
```
