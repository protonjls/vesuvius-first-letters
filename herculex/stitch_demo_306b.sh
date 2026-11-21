#!/bin/bash
# Known-answer demo for stitch_blocks.py on the awarded PHerc0306B line (awarded October 2026,
# github.com/SirProdigle/pherc0306b-first-letters). Input: the 9 written blocks of HerculeX source surface 24 (pieces
# 24, 1000024, ..., 9000024) of the same HerculeX run as ../results/herculex-306b/ (HerculeX PR#1 0d73c4c + band/shard
# patches, CPU, scan 20250521133212 band rows 30-42), shipped in ../results/herculex-306b-stitch/blocks/.
#   1. stitch_blocks.py: the 9 blocks -> one tifxyz; overlap disagreement per block.
#   2. place_check.py: the winners' claim mesh (pinned commit, CC BY-NC 4.0, fetched) vs the stitched surface.
#   3. ../followup/confirm_region.sh on the stitched render, window = the claim box on the stitched surface + 128 px:
#      reader-v3 + rectina-m4, both depth orders, d0 + off+15 + shuf.
#   4. stitch_sheet.py: CT | both models, controls, block boundaries and claim box drawn.
#
#   stitch_demo_306b.sh <out_dir>      (env as ../followup/confirm_region.sh: VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, ...)
# Optional env: VSEL (default "d0 off+15 shuf"), MARGIN (default 128 render px).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
KPY=${KPY:-python}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
RES=$KIT/../results/herculex-306b-stitch
CT=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
PUM=8.64; MARGIN=${MARGIN:-128}
run $KPY $KIT/stitch_blocks.py --out $OUT/stitched $RES/blocks/piece-* | tee $OUT/stitch.txt
[ -s $OUT/stitched/z.tif ] || { echo "stitch failed"; exit 2; }
W=https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc
mkdir -p $OUT/winners_mesh
for f in x.tif y.tif z.tif meta.json; do curl -sfL --max-time 120 -o $OUT/winners_mesh/$f $W/$f || echo "fetch $f failed"; done
[ -s $OUT/winners_mesh/z.tif ] || { echo "winners' mesh not fetched"; exit 3; }
# rows 82:137 cols 121:341 of their mesh = their claim box (submission/mesh/box_area.json tifxyz_grid_row_col corners)
run $KPY $KIT/place_check.py $OUT/winners_mesh $OUT/stitched --rows 82:137 --cols 121:341 | tee $OUT/place_check.json
read X0 Y0 SIDE CH BOX <<< "$(run $KPY -c "
import json
b = json.load(open('$OUT/place_check.json'))['cand_render_box_xyxy']
m = json.load(open('$OUT/stitched/meta.json')); st = int(round(1 / float(m['scale'][0])))
r = json.load(open('$OUT/stitched/stitch_report.json')); H, W = r['shape'][0] * st, r['shape'][1] * st
x0 = max(0, b[0] - $MARGIN); y0 = max(0, b[1] - $MARGIN); x1 = min(W, b[2] + $MARGIN); y1 = min(H, b[3] + $MARGIN)
print(x0, y0, x1 - x0, y1 - y0, ','.join(map(str, b)))")"
[ -n "${CH:-}" ] || { echo "no claim box on the stitched surface"; exit 4; }
echo "window x0 $X0 y0 $Y0 side $SIDE crop_h $CH (claim box $BOX)" | tee $OUT/window.txt
CROP_H=$CH VSEL="${VSEL:-d0 off+15 shuf}" bash $KIT/../followup/confirm_region.sh $OUT/confirm $OUT/stitched $CT $PUM $X0 $Y0 $SIDE
run $KPY $KIT/stitch_sheet.py $OUT/confirm $OUT/stitched --x0 $X0 --y0 $Y0 --box $BOX --out $OUT/sheet.jpg | tee $OUT/sheet.json
{ echo "herculex stitch_demo_306b summary"; head -1 $OUT/stitch.txt; cat $OUT/window.txt
  echo "place_check: $(cat $OUT/place_check.json)"; echo "sheet: $(cat $OUT/sheet.json)"; } > $OUT/summary.txt
echo "done: $OUT/summary.txt"
