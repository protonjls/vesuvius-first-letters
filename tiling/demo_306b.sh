#!/bin/bash
# Tiling demo on PHerc0306B (8.64 um volume 20250521133212), around the slice where its awarded First Letters text lies.
#
#   demo_306b.sh <out_dir>
#
# 1. seeds.py on the public m7 surface prediction, slice z=9348, window (3559 +- HALF, 3224 +- HALF)
#    (3559,3224,9348 = the seed published by the PHerc0306B winners);
# 2. tile_grow.sh: GrowPatch patches (GENS generations, step 20) from those seeds until every seed is covered or
#    MAXGROWS grows have run;
# 3. ink_patches.sh: CT render (24 layers) + reader-v3 in both depth orders for every kept patch;
# 4. rank.py: 20 mm windows ranked by text-likeness -> <out_dir>/rank.json, rank.png (top-36 contact sheet).
# Env: as tile_grow.sh and ink_patches.sh (VC3D, PY/KPY, INKSDK, VILLA_SRC, MODELS_DIR, WRAP, WORK, NCPU);
#   HALF (half window side, vox; 300), GENS (100), MAXGROWS (30), PROFILES (ink_9um@reader-v3).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
HALF=${HALF:-300}; GENS=${GENS:-100}; export MAXGROWS=${MAXGROWS:-30}
PY=${PY:-${KPY:-python3}}; WRAP=${WRAP:-}
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B
S=$B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2
CT=$B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
X=3559; Y=3224; Z=9348
t0=$(date +%s)
[ -s $OUT/seeds.tsv ] || $WRAP $PY $KIT/seeds.py --pred $S.zarr --z $Z --x0 $((X - HALF)) --x1 $((X + HALF)) \
    --y0 $((Y - HALF)) --y1 $((Y + HALF)) --out $OUT/seeds.tsv --png $OUT/seeds.png || { echo "seeds failed"; exit 2; }
echo "seeds $(($(wc -l < $OUT/seeds.tsv) - 1)) $(( $(date +%s) - t0 )) s"
PY=$PY bash $KIT/tile_grow.sh $OUT/tile $OUT/seeds.tsv $GENS $S.zarr $S.normal-grids 8.64 || { echo "tile_grow failed"; exit 3; }
t1=$(date +%s)
KPY=${KPY:-$PY} bash $KIT/ink_patches.sh $OUT/tile $CT 8.64 demo306b || { echo "ink_patches failed"; exit 4; }
t2=$(date +%s)
$WRAP $PY $KIT/rank.py $OUT/tile/maps --out-json $OUT/rank.json --sheet $OUT/rank.png --top 36 > $OUT/rank.txt \
  || { echo "rank failed"; exit 5; }
echo "grow $((t1 - t0)) s, ink $((t2 - t1)) s, total $(( $(date +%s) - t0 )) s; kept patches $(ls $OUT/tile/patches | wc -l)"
head -12 $OUT/rank.txt
