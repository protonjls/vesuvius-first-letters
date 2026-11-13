#!/bin/bash
# scroll-adapt: fine-tune reader-v3 on consensus pseudo-labels of unlabelled windings of ONE scroll (no human labels),
# then score it on held-out labelled windings of the same scroll. Reference experiment: PHerc1447, train on w062,
# test on w058/w060 against the refined labels of the HF dataset (scroll_adapt/fetch_1447.sh).
#
#   PY=... V3=... RECT=... DATA=... OUT=... bash scroll_adapt/run_adapt.sh
#
# Required env:
#   V3     reader-v3 checkpoint (reader-v3.pth, sha256 ea4054...84d8a as pinned by ink-sdk)
#   RECT   rectina-m4 checkpoint (ckpt_020000.pth); second teacher for the consensus
#   DATA   directory produced by fetch_1447.sh (<DATA>/<winding>/layers/00..23.tif, labels/{inklabels,mask}.png)
#   OUT    results directory (scores, logs, pseudo-label summary, configs)
# Optional env:
#   PY     python with koine_machines (villa branch merge-ink-pipelines @47a9c37f6, ink-detection/) + tifffile,
#          zarr 2, numcodecs, scipy, pillow                                          (default: python)
#   WRAP   command prefix for every python call, e.g. "apptainer exec --nv img.sif"  (default: none)
#   WORK   scratch dir for zarrs, maps and checkpoints (several GB)  (default: $SLURM_TMPDIR or a mktemp dir)
#   TRAIN  unlabelled windings used for pseudo-labels (default: w062); TEST  labelled held-out windings (w058 w060)
#   LR (0.001) ITERS (2000) BATCH (4) SAVE (500) WORKERS (3) INK_THR (0.6) BLANK_THR (0.3)
#   FLIP   1 (default) = reverse Z of the layer stacks: on these renders the reading side is the "reverse" order, so
#          every model then runs "forward" on the same zarr and base vs fine-tuned are like-for-like.
# Steps: zarrs -> teacher maps (v3, rect) on TRAIN -> make_pseudo (ink: sqrt(v3*rect) >= INK_THR; blank: max <= BLANK_THR;
# rest ignored) -> label zarrs -> NO-OP control (lr 0, 2 iterations: its map must equal base v3 exactly) -> fine-tune ->
# infer every saved checkpoint on TEST -> eval/label_auc.py -> scroll_adapt/check_adapt.py on the final checkpoint.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
V3=${V3:?set V3=reader-v3.pth}; RECT=${RECT:?set RECT=rectina-m4 ckpt}; DATA=${DATA:?set DATA}; OUT=${OUT:?set OUT}
PY=${PY:-python}; WRAP=${WRAP:-}
WORK=${WORK:-${SLURM_TMPDIR:-$(mktemp -d)}}
TRAIN=${TRAIN:-w062}; TEST=${TEST:-"w058 w060"}
LR=${LR:-0.001}; ITERS=${ITERS:-2000}; BATCH=${BATCH:-4}; SAVE=${SAVE:-500}; WORKERS=${WORKERS:-3}
INK_THR=${INK_THR:-0.6}; BLANK_THR=${BLANK_THR:-0.3}; FLIP=${FLIP:-1}
T=$WORK/scroll_adapt; mkdir -p $T/data/volumes $T/data/segments $T/maps $T/pl $OUT
run() { $WRAP "$@"; }
FL=""; [ "$FLIP" = 1 ] && FL=--flip
{ hostname; nvidia-smi -L 2>/dev/null; echo "REPO $REPO $(git -C $REPO rev-parse --short HEAD 2>/dev/null)";
  echo "V3 $V3"; echo "RECT $RECT"; echo "TRAIN $TRAIN TEST $TEST LR $LR ITERS $ITERS BATCH $BATCH SAVE $SAVE FLIP $FLIP"; } | tee $OUT/info.txt
infer() {  # infer <ckpt> <zarr> <out.tif> <log>
  run $PY -m koine_machines.inference.infer $2 $1 $3 --overlap 0.5 --blend-mode hann --direction forward \
       --batch-size 1 --amp-dtype bf16 --no-compile > $4 2>&1
  local rc=$?
  grep -iE "missing_keys|unexpected" $4 | head -2 | sed "s|^|  [$(basename $3)] |"
  [ -s $3 ] || { echo "infer failed: $3 (rc $rc)"; tail -5 $4; return 1; }
}
# 1. volumes
for w in $TRAIN $TEST; do
  [ -s $DATA/$w/layers/23.tif ] || { echo "missing layers for $w under $DATA (run fetch_1447.sh)"; exit 2; }
  run $PY $REPO/scroll_adapt/build_zarr.py vol $DATA/$w/layers $T/data/volumes/$w.zarr $FL | tee -a $OUT/info.txt
done
NZ=$(run $PY -c "import zarr; print(zarr.open('$T/data/volumes/${TRAIN%% *}.zarr', mode='r')['0'].shape[0])")
# 2. teacher maps on the training windings
for w in $TRAIN; do
  infer $V3 $T/data/volumes/$w.zarr $T/maps/${w}_v3.tif $OUT/infer_${w}_v3.log || exit 4
  infer $RECT $T/data/volumes/$w.zarr $T/maps/${w}_rect.tif $OUT/infer_${w}_rect.log || exit 4
  # 3. pseudo-labels
  run $PY $REPO/scroll_adapt/make_pseudo.py --map-a $T/maps/${w}_v3.tif --map-b $T/maps/${w}_rect.tif \
       --valid $DATA/$w/layers/12.tif --ink-thr $INK_THR --blank-thr $BLANK_THR --out $T/pl/$w | tee $OUT/pseudo_$w.json
  cp $T/pl/${w}_inklabels.png $T/pl/${w}_supervision_mask.png $OUT/
  run $PY $REPO/scroll_adapt/build_zarr.py label $T/pl/${w}_inklabels.npy $T/pl/${w}_supervision_mask.npy \
       $T/data/segments $w --z $NZ | tee -a $OUT/info.txt
done
SEGS=""; for w in $TRAIN; do SEGS="$SEGS --seg $w"; done
# 4. no-op control: lr 0, 2 iterations -> must reproduce base reader-v3 exactly on the first test winding
W0=${TEST%% *}
run $PY $REPO/scroll_adapt/make_config.py --out $T/noop.json --out-dir $T/noop --root $T/data $SEGS --checkpoint $V3 \
     --lr 0 --iters 2 --save-every 2 --warmup 1 --batch $BATCH --workers $WORKERS > /dev/null
run $PY -m koine_machines.training.train $T/noop.json > $OUT/train_noop.log 2>&1; echo "noop train rc=$?" | tee -a $OUT/info.txt
NCK=$(ls $T/noop/*.pth 2>/dev/null | grep -v best | sort | tail -1)
[ -n "$NCK" ] || { echo "no-op produced no checkpoint"; tail -30 $OUT/train_noop.log; exit 5; }
for w in $TEST; do
  infer $V3 $T/data/volumes/$w.zarr $T/maps/${w}_v3.tif $OUT/infer_${w}_v3.log || exit 4
  infer $RECT $T/data/volumes/$w.zarr $T/maps/${w}_rect.tif $OUT/infer_${w}_rect.log || exit 4
done
infer $NCK $T/data/volumes/$W0.zarr $T/maps/${W0}_noop.tif $OUT/infer_${W0}_noop.log || exit 4
run $PY -c "
import tifffile, numpy as np
a=tifffile.imread('$T/maps/${W0}_v3.tif').astype(int); b=tifffile.imread('$T/maps/${W0}_noop.tif').astype(int)
d=np.abs(a-b); print('NOOP max|diff| %d  frac>1 %.6f  (uint8 maps)'%(d.max(), (d>1).mean()))" | tee -a $OUT/info.txt
# 5. fine-tune
run $PY $REPO/scroll_adapt/make_config.py --out $OUT/train_cfg.json --out-dir $T/ft --root $T/data $SEGS --checkpoint $V3 \
     --lr $LR --iters $ITERS --save-every $SAVE --batch $BATCH --workers $WORKERS > /dev/null
start=$(date +%s)
run $PY -m koine_machines.training.train $OUT/train_cfg.json > $OUT/train.log 2>&1; echo "train rc=$? $(( $(date +%s)-start )) s" | tee -a $OUT/info.txt
# 6. infer every saved checkpoint on the held-out windings; final = highest step (fixed before results)
for ck in $(ls $T/ft/ckpt_*.pth 2>/dev/null | sort); do
  s=$(basename $ck .pth | sed 's/ckpt_//')
  for w in $TEST; do infer $ck $T/data/volumes/$w.zarr $T/maps/${w}_ft$s.tif $OUT/infer_${w}_ft$s.log || continue; done
done
FINAL=$(ls $T/ft/ckpt_*.pth 2>/dev/null | sort | tail -1)
FS=$(basename ${FINAL:-none} .pth | sed 's/ckpt_//'); echo "final checkpoint: ${FINAL:-none} (kept in $T/ft)" | tee -a $OUT/info.txt
# 7. scores
SC=""
for w in $TEST; do
  run $PY -c "
import tifffile, numpy as np
a=tifffile.imread('$T/maps/${w}_v3.tif').astype(np.float32); b=tifffile.imread('$T/maps/${w}_rect.tif').astype(np.float32)
np.save('$T/maps/${w}_prod.npy', np.sqrt(a*b))"
  args="--map v3=$T/maps/${w}_v3.tif --map rect=$T/maps/${w}_rect.tif --map v3xrect=$T/maps/${w}_prod.npy"
  for f in $T/maps/${w}_ft*.tif; do [ -s $f ] && args="$args --map $(basename $f .tif | sed "s/${w}_//")=$f"; done
  [ -n "$FINAL" ] && [ -s $T/maps/${w}_ft$FS.tif ] && args="$args --map final=$T/maps/${w}_ft$FS.tif"
  run $PY $REPO/eval/label_auc.py --labels $DATA/$w/labels/inklabels.png --mask $DATA/$w/labels/mask.png \
       $args --out $OUT/score_$w.json | tee $OUT/score_$w.txt
  SC="$SC $OUT/score_$w.json"
done
run $PY $REPO/scroll_adapt/check_adapt.py --base v3 --adapted final --teacher v3xrect $SC | tee $OUT/check_adapt.txt
echo done
