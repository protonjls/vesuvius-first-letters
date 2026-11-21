#!/bin/bash
# Known-answer demo on an already-awarded region: the PHerc0306B First Letters line (awarded October 2026,
# github.com/SirProdigle/pherc0306b-first-letters). Two meshes of that place made by different methods:
#   A = HerculeX piece 7000024 (../results/herculex-306b/piece-7000024, see ../herculex/README.md)
#   B = a GrowPatch mesh grown on the public m7 surface prediction (../results/crosscheck-306b/grow-mesh: VC3D
#       f0e4410 vc_grow_seg_from_seed, seed 3559 3224 9348, 140 generations, step 20; same recipe as
#       ../followup/grow_region.sh)
# Steps: reader-v3 + rectina-m4 at d0 on the whole piece (../herculex/piece_confirm.sh, VSEL=d0) and on a window of B
# covering the piece (../followup/confirm_region.sh, VSEL=d0), then reappear.py on A rows 2:90 cols 0:143 (the
# part of the piece that overlaps the winners' claim box, ../results/herculex-306b/place_check.json), once as is and
# once with --shift 160 (about half a letter) as the negative control.
#
#   demo_306b.sh <out_dir>      (env as ../followup/confirm_region.sh: VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, ...)
# CPU timings (16 cores): ~10 min for the piece, ~10-30 min for the B window, reappear ~1 min.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
KPY=${KPY:-python}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
A=$KIT/../results/herculex-306b/piece-7000024
B=$KIT/../results/crosscheck-306b/grow-mesh
CT=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
PUM=8.64
VSEL=d0 ORDER=fwd bash $KIT/../herculex/piece_confirm.sh $OUT/mapsA $A $CT $PUM
# window of B (render pixels x0 y0 side) that holds A rows 2:90 cols 0:143 (B grid rows 62-188 cols 77-201)
VSEL=d0 ORDER=fwd bash $KIT/../followup/confirm_region.sh $OUT/mapsB $B $CT $PUM 1500 1200 2600
for SH in 0 160; do
  run $KPY $KIT/reappear.py $A $OUT/mapsA $B $OUT/mapsB $OUT/reappear_shift$SH --rows 2:90 --cols 0:143 \
      --offB 1500,1200 --shift $SH | tee $OUT/reappear_shift$SH.log
done
{ echo "crosscheck demo_306b summary"; echo "A $A"; echo "B $B (window 1500 1200 2600)"
  for SH in 0 160; do echo "--- shift $SH"; cat $OUT/reappear_shift$SH/reappear.tsv; done; } > $OUT/summary.txt
echo "done: $OUT/summary.txt"
