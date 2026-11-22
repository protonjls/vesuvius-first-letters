#!/bin/bash
# Known-answer demo on public PHerc0139 data (segment 20250108000000-w025, which carries the Vesuvius Challenge team's own
# canonical 2.4 um ink map). Two checks:
#   B: the segment's 2.399 um mesh rendered on the 2.399 um rescan 20260102150214 + canonical_2um must reproduce the team's
#      published canonical map over one 3000 x 3000 px window (bar r >= 0.8);
#   C: the segment's 9.362 um mesh (on volume 20250728140407) mapped into the rescan by the inverse official transform
#      20260102150214-to-20250728140407-r1 must give the same map (bar r >= 0.6 vs B), i.e. a 9 um mesh can carry the
#      2.4 um model.
#   demo_0139.sh <out_dir>      env as render_infer.sh (VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, ...); GPU needed
#                               for canonical_2um (about 30 min on one MIG 2g.20gb H100 slice incl. streaming).
# Downloads about 110 MB (meshes, transform, team map) plus the streamed CT chunks.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
KPY=${KPY:-python}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
S3=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139
SEG=$S3/segments/20250108000000-w025_2025010863
CT24=$S3/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr
D=$OUT/data; mkdir -p $D/m24 $D/m9 $D/tf
get() { [ -s "$2" ] || curl -sfL --retry 3 -o "$2" "$1" || { echo "download failed: $1"; exit 2; }; }
for f in meta.json x.tif y.tif z.tif; do
  get $SEG/mesh/20250108000000-on-20260102150214-2.399um.tifxyz/$f $D/m24/$f
  get $SEG/mesh/20250108000000-on-20250728140407-9.362um.tifxyz/$f $D/m9/$f
done
get $S3/transforms/20260102150214-to-20250728140407-r1/fit/transform.json $D/tf/transform.json
get "$SEG/ink-detection/PHerc0139-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif" $D/team.tif
WIN="4000 11500 3000 3000"     # team map rows, cols, height, width (holds clear letter forms)
run $KPY $KIT/nodecorr.py --a $D/m9 --b $D/m24 --affine $D/tf/transform.json:inv --out $OUT/corr.json > $OUT/nodecorr.log \
  || { echo "nodecorr failed"; exit 3; }
CW=$(run $KPY $KIT/kat_compare.py window --corr $OUT/corr.json --affine $D/tf/transform.json --mesh-a $D/m9 --mesh-b $D/m24 --win $WIN)
echo "C window (x0 y0 w h): $CW" | tee $OUT/demo.log
set -- $WIN; R0=$1; C0=$2; H=$3; W=$4
bash $KIT/render_infer.sh $OUT/B $D/m24 $CT24 2.399 - $C0 $R0 $W $H || { echo "B failed"; exit 4; }
set -- $CW
bash $KIT/render_infer.sh $OUT/C $D/m9 $CT24 2.399 $D/tf/transform.json:inv $1 $2 $3 $4 || { echo "C failed"; exit 4; }
run $KPY $KIT/kat_compare.py compare --b $OUT/B --c $OUT/C --team $D/team.tif --win $WIN --corr $OUT/corr.json \
    --affine $D/tf/transform.json --mesh-a $D/m9 --mesh-b $D/m24 --c-origin $1 $2 --out $OUT/kat | tee -a $OUT/demo.log
echo "done: $OUT/kat/kat.json $OUT/kat/sheet.png"
