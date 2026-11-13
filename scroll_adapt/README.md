# scroll_adapt: fine-tune reader-v3 on one scroll from machine pseudo-labels, with no human labels

**Idea.** Public ink models are trained on a few scrolls. The 2026 PHerc. 306B First Letters win fine-tuned a model
on that scroll's own machine pseudo-labels. scroll_adapt is a small, checked kit for doing the same with
[reader-v3](https://github.com/DomRusso2) in villa's `koine_machines` trainer:

1. Run two public models (reader-v3 and rectina-m4) on *unlabelled* windings of the target scroll.
2. Keep only the pixels both models agree on. Ink = `sqrt(v3*rect) >= 0.6`, minus components under 30 px. Blank =
   `max(v3, rect) <= 0.3`. Everything else, including a 3 px ring around each ink component, is excluded from
   the loss through koine's supervision mask (`make_pseudo.py`).
3. Fine-tune reader-v3 from its own weights with its exact training contract (`make_config.py`; the contract is
   copied from the checkpoint's embedded config).
4. Score base and fine-tuned models on *held-out labelled* windings of the same scroll, on the same zarr with the same
   flags.

**What the kit gets right (each item is a trap we hit or checked).** All of them are documented in
`docs/koine_finetune_notes.md` with code citations.
- Label zarrs carry the 2D map in every Z slice. Villa's `create_label_zarrs.py` writes only slice 32, which a
  21–24-layer volume never reaches, so every pixel ends up silently ignored.
- `weights_only: true`. reader-v3 ships without optimizer state, so a full restore raises.
- The model contract is copied verbatim: patch 17×384×384, robust-MAD normalisation, Z jitter ±2, BCE label
  smoothing 0.5. Smoothing keeps the output scale that downstream tools assume.
- **No-op control:** a run with learning rate 0 for 2 iterations must reproduce base reader-v3 *exactly*. It did on
  PHerc1447 w058 (uint8 maps, max |diff| 0), so any change after fine-tuning comes from training, not from the
  pipeline.
- Base and fine-tuned models run on the same (Z-flipped) zarr with the same inference flags. With an even layer
  count, a flip shifts the centred 17-layer window by one layer: in our runs, base reader-v3 scored 0.855 on the
  flipped zarr against 0.849 through ink-sdk's `reverse` mode on w058.

## Result on PHerc1447 (negative)
Training data: pseudo-labels on w062 (2.0% ink, 63.7% blank, 34.2% ignored). Test data: w058 and w060 with the refined
human labels from the HF dataset. Neither model saw those labels; reader-v3 was never trained on PHerc1447.

| model | w058 AUC | w060 AUC | w058 FPR@R50 | w060 FPR@R50 |
|---|---|---|---|---|
| base reader-v3 | 0.855 | 0.785 | 0.047 | 0.101 |
| teacher: sqrt(v3 × rect) | 0.869 | 0.797 | 0.035 | 0.092 |
| fine-tuned, LR 1e-3, 2000 it (final, fixed before results) | 0.860 | 0.764 | 0.042 | 0.098 |
| fine-tuned, LR 3e-4, 3000 it (final) | 0.858 | 0.765 | 0.044 | 0.095 |

The bar was fixed before any result: +0.01 AUC on both windings, and false-positive rate at 50% recall no more than
0.005 higher. **Neither run passes.** The fine-tuned model does not beat its own teachers, and on w060 it falls
below base. Intermediate checkpoints were no better (first run, w060: 0.746 / 0.722 / 0.759 at 500 / 1000 / 1500
iterations; second run: 0.761 / 0.763 at 1000 / 2000), so training on consensus pseudo-labels pulled the model *away* from the human labels. The
per-batch loss (label-smoothed BCE + Dice) stayed in about 0.50–0.57 throughout, close to its smoothing floor.

Pseudo-label quality on the labelled windings, with the same thresholds: ink precision 0.88/0.90, ink recall
0.39/0.20, blank purity 0.92/0.88 (`results/pseudo_quality_1447.txt`). The labels are mostly right but cover only the
easiest ink, the pixels both models already find. That is a plausible reason the student learns nothing new. It is
consistent with community reports that uncurated pseudo-labels tend to collapse, and with a leave-one-out fine-tune
on *real* 1447 labels that did not beat the base model either.

Only two configurations were tried, and we stopped there on purpose: every additional variant scored on the same
two test windings would be tuning on the test set. Untested and still open: more unlabelled windings (here only one,
w062), a third teacher such as v8-in (the best single model on 1447, see `results/ensembles_1447_*.txt`), and other
scrolls.

## Run it
See `docs/environment.md` for the Python environment and weights. Then:
```bash
scroll_adapt/fetch_1447.sh $DATA                         # HF renders + labels, about 1.6 GB
PY=/path/to/ink-detection/.venv/bin/python V3=reader-v3.pth RECT=rectina-m4/ckpt_020000.pth \
DATA=$DATA OUT=out/adapt LR=0.001 ITERS=2000 bash scroll_adapt/run_adapt.sh
cat out/adapt/info.txt        # must contain "NOOP max|diff| 0"
cat out/adapt/check_adapt.txt # PASS/FAIL against the bar above
```
For another scroll, render its windings as 24-layer uint8 layer stacks (`<DATA>/<winding>/layers/00..23.tif`) and set
`TRAIN` to the unlabelled windings. Set `TEST` to windings with labels; without labelled windings there is no way to
know whether the fine-tune helped. Set `FLIP=0` if the reading side is already "forward".

Files: `run_adapt.sh` (pipeline), `make_pseudo.py` (consensus labels), `build_zarr.py` (volume and label zarrs in
koine's layout), `make_config.py` (fine-tune config), `check_adapt.py` (bar), `fetch_1447.sh` (data).
