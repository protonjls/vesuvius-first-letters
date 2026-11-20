#!/bin/bash
# Second model and false-positive controls ON a HerculeX piece: the same surface HerculeX inked, rendered by VC3D.
# A HerculeX ink image shows one model (reader-v3) on one face. This script renders the whole piece at scale 1 and
# runs ../followup/confirm_region.sh on it: control variants (centred at d-2..d+2, 15 voxels off the surface on
# either side, depth-shuffled), every model in PROFILES in both depth orders, persistence statistics and a montage.
#
#   piece_confirm.sh <out_dir> <piece dir> <ct_url> <pixel_um>
#
#   piece dir: a HerculeX piece in tifxyz form (x.tif y.tif z.tif meta.json), e.g.
#     <HERCULEX_OUTPUTS>/unwrapping/level/<scroll>/<scan>/pieces/<unit>/piece-<n>/
#   Render pixel (y, x) = piece grid (y/stride, x/stride), stride = 1/meta.json scale (16 for HerculeX pieces).
# Env: as ../followup/confirm_region.sh (VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, WORK, NCPU, CACHE_GB,
#   PROFILES, N, ORDER, VSEL). KPY needs numpy + tifffile.
# Output: as confirm_region.sh, plus <out_dir>/piece.txt (piece dir, grid shape, stride, render size).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; P=${2:?piece dir}; URL=${3:?ct_url}; PUM=${4:?pixel_um}
KPY=${KPY:-python}; WRAP=${WRAP:-}
P=$(cd $P && pwd); mkdir -p $OUT
read ROWS COLS STRIDE <<< "$($WRAP $KPY $KIT/place_check.py --size $P)"
[ -n "${STRIDE:-}" ] || { echo "cannot read piece size of $P"; exit 2; }
W=$(( COLS * STRIDE )); H=$(( ROWS * STRIDE ))
echo "piece $P grid ${ROWS}x${COLS} stride $STRIDE render ${W}x${H}" | tee $OUT/piece.txt
CROP_H=$H bash $KIT/../followup/confirm_region.sh $OUT $P $URL $PUM 0 0 $W
