#!/bin/bash
# Known-answer demo of iso_regrid.py on public PHerc0211 data (CPU only, about 1-2 min; downloads about 0.3 MB).
#   demo_0211.sh <out_dir>      env: PY = a Python with numpy, scipy and tifffile (default python3)
# Three arms, each: input mesh -> iso_regrid.py --step 20 -> arm_check.py (distortion of input and output, normal distance
# of the output nodes to the reference surface, bars fixed before the first clean-checkout run):
#   NOOP   the winners' hand-refined v9 mesh of the awarded PHerc0211 region (already isometric) -> must stay isometric
#          and on the same surface: output distortion median < 0.02, p95 < 0.06; dn(output -> v9) median <= 0.5 voxel.
#   SQUASH v9 with its columns re-sampled by squash.py (edges 0.4x-1.6x) -> input p95 >= 0.5; output median < 0.02,
#          p95 < 0.06; dn(output -> unsquashed v9) median <= 0.5 voxel.
#   FIT    data/w034_fit, a raw villa fit_spiral winding over the same region (column edges 15-53 voxels) -> input
#          p95 >= 0.5; output median < 0.02, p95 < 0.08; dn(output -> w034_fit) median <= 0.5 voxel.
# Writes <out>/{noop,squash,fit}.json and <out>/demo_summary.txt; exits 1 if any bar fails.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
PY=${PY:-python3}
V9URL=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0211/segments/20261028201525-crop1711-lines123-v9/mesh/20261028201525-on-20250821151803-9.362um.tifxyz
D=$OUT/data; mkdir -p $D/v9
get() { [ -s "$2" ] || curl -sfL --retry 3 -o "$2" "$1" || { echo "download failed: $1"; exit 2; }; }
for f in meta.json x.tif y.tif z.tif; do get $V9URL/$f $D/v9/$f; done
(cd $D/v9 && sha256sum meta.json x.tif y.tif z.tif) > $OUT/v9.sha256
(cd $KIT/data/w034_fit && sha256sum meta.json x.tif y.tif z.tif) > $OUT/w034_fit.sha256
iso() { $PY $KIT/iso_regrid.py "$1" "$2" --step 20 > "$2.log" 2>&1 || { echo "iso_regrid failed on $1"; tail -5 "$2.log"; exit 3; }; }
$PY $KIT/squash.py $D/v9 $OUT/squash_in > $OUT/squash_in.log 2>&1 || { echo "squash failed"; exit 3; }
iso $D/v9 $OUT/noop_iso
iso $OUT/squash_in $OUT/squash_iso
iso $KIT/data/w034_fit $OUT/fit_iso
: > $OUT/demo_summary.txt
$PY $KIT/arm_check.py NOOP $D/v9 $OUT/noop_iso $D/v9 $OUT/noop.json --out-med 0.02 --out-p95 0.06 --dn-med 0.5 \
  | tee -a $OUT/demo_summary.txt || exit 4
$PY $KIT/arm_check.py SQUASH $OUT/squash_in $OUT/squash_iso $D/v9 $OUT/squash.json --out-med 0.02 --out-p95 0.06 \
  --dn-med 0.5 --in-p95-min 0.5 | tee -a $OUT/demo_summary.txt || exit 4
$PY $KIT/arm_check.py FIT $KIT/data/w034_fit $OUT/fit_iso $KIT/data/w034_fit $OUT/fit.json --out-med 0.02 --out-p95 0.08 \
  --dn-med 0.5 --in-p95-min 0.5 | tee -a $OUT/demo_summary.txt || exit 4
n=$(grep -c ' -> PASS$' $OUT/demo_summary.txt)
if [ "$n" -eq 3 ]; then echo "ISOREGRID DEMO PASS (3/3 arms)" | tee -a $OUT/demo_summary.txt; exit 0; fi
echo "ISOREGRID DEMO FAIL ($n/3 arms pass)" | tee -a $OUT/demo_summary.txt; exit 1
