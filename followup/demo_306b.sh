#!/bin/bash
# Known-answer demo on a published, already-awarded region: the PHerc0306B First Letters line (awarded October 2026,
# github.com/SirProdigle/pherc0306b-first-letters). From the winners' published GrowPatch seed, the kit grows its own
# surface, runs public reader-v3 on it, resamples the map into the winners' claim-mesh frame, and runs the
# false-positive controls on a window over the line.
#
#   demo_306b.sh <out_dir>          (env as grow_region.sh: VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, ...)
#
# Optional env: DEMO_PROFILES (models on the grown patch, default ink_9um@reader-v3), CONFIRM=0 (skip the controls),
#               CONFIRM_PROFILES (default "ink_9um@reader-v3 ink_9um@rectina-m4").
# Output: <out_dir>/grow/ (grow_region.sh outputs), winners_mesh/ (their claim mesh, CC BY-NC 4.0, fetched),
#   reader-v3_fwd_in_winners_frame.png (+ _reading.png: flipped top-to-bottom = their reading orientation, ink dark),
#   confirm/ (confirm_region.sh outputs, order fwd), summary.txt.
# CPU timings (16 cores): grow ~27 min, render ~1 min, reader-v3 ~45 min, controls ~1-2 h.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
KPY=${KPY:-python}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B
S=$B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2
CT=$B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
PUM=8.64
# 1. grow + render + reader-v3 (seed, step 20 and 140 generations as in the winners' docs/METHOD.md)
PROFILES=${DEMO_PROFILES:-ink_9um@reader-v3} STEP=20 bash $KIT/grow_region.sh $OUT/grow 3559 3224 9348 140 $S.zarr $S.normal-grids $CT $PUM
[ -f $OUT/grow/reader-v3_fwd_ds2.png ] || { echo "no reader-v3 map"; exit 5; }
# 2. the winners' claim mesh (pinned commit) -> our map in their frame, rows 70:150 cols 100:360 around their 4 cm2 box
W=https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc
mkdir -p $OUT/winners_mesh
for f in x.tif y.tif z.tif meta.json; do curl -sfL --max-time 120 -o $OUT/winners_mesh/$f $W/$f || echo "fetch $f failed"; done
if [ -s $OUT/winners_mesh/z.tif ]; then
  run $KPY $KIT/mesh_resample.py $OUT/winners_mesh $OUT/grow/mesh_grow $OUT/grow/reader-v3_fwd_ds2.png \
      $OUT/reader-v3_fwd_in_winners_frame.png --rows 70:150 --cols 100:360 --ppcA 10 --ppcB 10 | tee $OUT/resample.txt
  run $KPY -c "
from PIL import Image, ImageOps
im = Image.open('$OUT/reader-v3_fwd_in_winners_frame.png')
ImageOps.autocontrast(ImageOps.invert(ImageOps.flip(im)), cutoff=0.5).save('$OUT/reader-v3_fwd_in_winners_frame_reading.png')"
fi
# 3. controls on a 2400 px window centred on the grown-mesh node nearest a point of the line (3312, 3427, 9697)
if [ "${CONFIRM:-1}" = 1 ]; then
  read R C D <<< "$(run $KPY $KIT/kit_util.py nearest $OUT/grow/mesh_grow 3312 3427 9697)"
  echo "line anchor: grid node $R $C (distance $D vox)" | tee $OUT/anchor.txt
  X0=$(( C * 20 - 1200 )); Y0=$(( R * 20 - 1200 )); [ $X0 -lt 0 ] && X0=0; [ $Y0 -lt 0 ] && Y0=0
  PROFILES=${CONFIRM_PROFILES:-"ink_9um@reader-v3 ink_9um@rectina-m4"} ORDER=fwd \
    bash $KIT/confirm_region.sh $OUT/confirm $OUT/grow/mesh_grow $CT $PUM $X0 $Y0 2400
fi
{ echo "demo_306b summary"; cat $OUT/grow/run.log; cat $OUT/resample.txt 2>/dev/null; cat $OUT/anchor.txt 2>/dev/null
  grep -A 12 "^order" $OUT/confirm/run.log 2>/dev/null; } > $OUT/summary.txt
echo "done: $OUT/summary.txt"
