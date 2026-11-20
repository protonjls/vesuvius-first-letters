#!/bin/bash
# Known-answer demo on an already-awarded region: the PHerc0306B First Letters line (awarded October 2026,
# github.com/SirProdigle/pherc0306b-first-letters). Input: one HerculeX piece of that region, shipped in
# ../results/herculex-306b/ (HerculeX github.com/adambetka2/HerculeX PR#1 0d73c4c + band/shard patches, CPU, scan
# 20250521133212 band rows 30-42, piece 7000024, with its CT face and reader-v3 'taught' ink images).
#   1. piece_confirm.sh: render the whole piece, reader-v3 + rectina-m4 in both orders on d-2..d+2, off+-15, shuf.
#   2. place_check.py: the winners' claim mesh (pinned commit, CC BY-NC 4.0, fetched) vs the piece, normal distance.
#   3. piece_sheet.py: HerculeX ink | CT | both models, controls row, claim box drawn; face/ink alignment r.
#
#   demo_306b.sh <out_dir>      (env as ../followup/confirm_region.sh: VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, ...)
# Optional env: PROFILES (default reader-v3 + rectina-m4).
# CPU timings (16 cores): render ~1 min, each model ~3-6 min per variant, 16 runs.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
KPY=${KPY:-python}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
RES=$KIT/../results/herculex-306b
PIECE=$RES/piece-7000024
CT=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
PUM=8.64
ORDER=${ORDER:-fwd} bash $KIT/piece_confirm.sh $OUT/confirm $PIECE $CT $PUM
W=https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc
mkdir -p $OUT/winners_mesh
for f in x.tif y.tif z.tif meta.json; do curl -sfL --max-time 120 -o $OUT/winners_mesh/$f $W/$f || echo "fetch $f failed"; done
BOX=""
if [ -s $OUT/winners_mesh/z.tif ]; then
  # rows 82:137 cols 121:341 of their mesh = their claim box (submission/mesh/box_area.json tifxyz_grid_row_col corners)
  run $KPY $KIT/place_check.py $OUT/winners_mesh $PIECE --rows 82:137 --cols 121:341 | tee $OUT/place_check.json
  BOX=$(run $KPY -c "import json; b = json.load(open('$OUT/place_check.json')).get('cand_render_box_xyxy'); print(','.join(map(str, b)) if b else '')")
fi
run $KPY $KIT/piece_sheet.py $OUT/confirm --out $OUT/sheet.png --ct-face $RES/faces/piece-7000024.png \
    --ink-face $RES/faces/piece-7000024-taught.png ${BOX:+--box $BOX} | tee $OUT/sheet.json
{ echo "herculex demo_306b summary"; cat $OUT/confirm/piece.txt; grep -A 12 "^order" $OUT/confirm/run.log 2>/dev/null
  echo "place_check: $(cat $OUT/place_check.json 2>/dev/null)"; echo "sheet: $(cat $OUT/sheet.json)"; } > $OUT/summary.txt
echo "done: $OUT/summary.txt"
