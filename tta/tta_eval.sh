#!/bin/bash
# Test-time augmentation for koine_machines ink models (reader-v3 by default), scored on known text.
#
#   PY=... CKPT=reader-v3.pth DATA=... OUT=... bash tta/tta_eval.sh
#
# Surfaces: PHerc1447 w058/w060 (24-layer HF renders from scroll_adapt/fetch_1447.sh; Z flipped so the reading side is
# "forward"; pixel AUC vs the refined labels). Optional: PHerc0343 letter-box AUC on a 66-layer render of Nieuwlaar's
# winning concat mesh (set Z343=<render.zarr> and BOX343=<letter_boxes.json>, see tta/README.md).
# Arms (fixed 2026-11-13 before results): A0 = single centred 17-layer window (offset 0); A1 = A0 + --tta-mirror;
# A2 = mean of offsets -2..+2; A3 = mean of offsets -1..+1. Single offsets are reported for information.
# Offset o moves the 17-layer window by o layers (koine --layer-start/--layer-end, half-open).
# Env: PY (python, default "python"), WRAP (command prefix, e.g. "apptainer exec --nv img.sif"), CKPT (required),
#      DATA (required for 1447), OUT (required), WORK (default $SLURM_TMPDIR or mktemp), OFFS ("-3 -2 -1 0 1 2 3"),
#      SURF ("w058 w060", add "343" with Z343/BOX343 set).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT=${CKPT:?set CKPT}; OUT=${OUT:?set OUT}; DATA=${DATA:-}; PY=${PY:-python}; WRAP=${WRAP:-}
WORK=${WORK:-${SLURM_TMPDIR:-$(mktemp -d)}}
OFFS=${OFFS:-"-3 -2 -1 0 1 2 3"}; SURF=${SURF:-"w058 w060"}
T=$WORK/tta; mkdir -p $T/vol $T/maps $OUT
run() { $WRAP "$@"; }
{ hostname; nvidia-smi -L 2>/dev/null; echo "REPO $REPO $(git -C $REPO rev-parse --short HEAD 2>/dev/null)"; echo "CKPT $CKPT OFFS $OFFS SURF $SURF"; } | tee $OUT/info.txt
infer() {  # infer <zarr> <out.tif> <log> <direction> <layer-start> <layer-end> [extra]
  local z=$1 o=$2 l=$3 d=$4 s=$5 e=$6; shift 6
  run $PY -m koine_machines.inference.infer $z $CKPT $o --overlap 0.5 --blend-mode hann --direction $d \
       --batch-size 1 --amp-dtype bf16 --no-compile --layer-start $s --layer-end $e "$@" > $l 2>&1
  grep -m1 "Selected source layer indices" $l | sed "s|^.*indices=|  $(basename $o) layers |"
  [ -s $o ] || { echo "infer failed: $o"; tail -5 $l; return 1; }
}
for S in $SURF; do
  if [ $S = 343 ]; then
    Z=${Z343:?set Z343 for the 343 surface}; D=reverse
  else
    Z=$T/vol/$S.zarr; D=forward
    [ -d $Z ] || run $PY $REPO/scroll_adapt/build_zarr.py vol $DATA/$S/layers $Z --flip | tee -a $OUT/info.txt
  fi
  NZ=$(run $PY -c "import zarr; print(zarr.open('$Z', mode='r')['0'].shape[0])")
  C=$(( NZ / 2 - 8 ))   # start of the centred 17-layer window (koine: Z//2 - 8)
  for o in $OFFS; do
    s=$(( C + o )); e=$(( s + 17 ))
    [ $s -ge 0 ] && [ $e -le $NZ ] || { echo "skip offset $o for $S"; continue; }
    infer $Z $T/maps/${S}_o$o.tif $OUT/infer_${S}_o$o.log $D $s $e || continue
  done
  infer $Z $T/maps/${S}_mirror.tif $OUT/infer_${S}_mirror.log $D $C $(( C + 17 )) --tta-mirror
  run $PY -c "
import tifffile, numpy as np, os
m='$T/maps/${S}'
for name, offs in (('dmean2', range(-2, 3)), ('dmean1', range(-1, 2))):
    fs = [f'{m}_o{o}.tif' for o in offs]
    if all(os.path.exists(f) for f in fs):
        np.save(f'{m}_{name}.npy', np.mean([tifffile.imread(f).astype(np.float32) for f in fs], axis=0))
        print(name, 'from', len(fs), 'maps')
"
  args=""
  for f in $T/maps/${S}_o*.tif $T/maps/${S}_mirror.tif $T/maps/${S}_dmean*.npy; do
    [ -s $f ] && args="$args --map $(basename $f | sed -e "s/^${S}_//" -e 's/\.[a-z]*$//')=$f"
  done
  if [ $S = 343 ]; then
    run $PY $REPO/eval/pc_score.py --boxes ${BOX343:?set BOX343} $args --out $OUT/score_343.json | tee $OUT/score_343.txt
  else
    run $PY $REPO/eval/label_auc.py --labels $DATA/$S/labels/inklabels.png --mask $DATA/$S/labels/mask.png \
         $args --out $OUT/score_$S.json | tee $OUT/score_$S.txt
  fi
done
echo done
