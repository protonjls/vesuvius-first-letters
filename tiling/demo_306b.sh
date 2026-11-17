#!/bin/bash
# Known-answer demo of the tiling tools on PHerc0306B (8.64 um volume 20250521133212), whose First Letters line was
# awarded in October 2026 (github.com/SirProdigle/pherc0306b-first-letters).
#
#   demo_306b.sh <out_dir>
#
# 1. seeds.py on the public m7 surface prediction, slice z=9348, window x2659-4459 y2324-4124 (the winners' seed
#    3559,3224 +- 900 vox): 827 seeds on every predicted sheet crossing of the window, no umbilicus, no hand picks.
#    The file is compared with results/tiling-306b/seeds_kat_window.tsv (the seeds of the full run reported in README).
# 2. tile_grow.sh on the seeds selected by ONLY (default 405: the seed whose patch covered the awarded region in the
#    full run). ONLY=all grows the whole window (about 230 grows, ~18 h on 16 CPUs; split it with PART=k/K jobs).
# 3. ink_patches.sh: CT render (24 layers) + reader-v3, both depth orders, for every kept patch.
# 4. The winners' claim mesh (pinned commit; CC BY-NC 4.0, fetched, not redistributed) -> coverage.py per kept patch
#    (their claim rows 82:137, cols 121:341, tol 5 vox) -> coverage.tsv.
# 5. For the best-covering patch: its reader-v3 maps resampled into the winners' frame (rows 70:150, cols 100:360) and
#    compared with the v0.2 followup reference reading view (results/followup-306b) by demo_check.py.
# 6. rank.py over all maps -> rank.json, rank.png.
# Env: as tile_grow.sh and ink_patches.sh (VC3D, PY/KPY, INKSDK, VILLA_SRC, MODELS_DIR, WRAP, WORK, NCPU);
#   ONLY (405), GENS (100), PROFILES (ink_9um@reader-v3).
# CPU (16 cores): seeds ~1 min, one 100-generation grow ~5 min, reader-v3 on one patch 3-50 min (node-dependent).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd $KIT/.. && pwd)
OUT=${1:?out_dir}; mkdir -p $OUT; OUT=$(cd $OUT && pwd)
ONLY=${ONLY:-405}; GENS=${GENS:-100}; export MAXGROWS=${MAXGROWS:-400}
PY=${PY:-${KPY:-python3}}; WRAP=${WRAP:-}
run() { $WRAP "$@"; }
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B
S=$B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2
CT=$B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr
X=3559; Y=3224; Z=9348; HALF=900
t0=$(date +%s)
# 1. seeds
[ -s $OUT/seeds.tsv ] || run $PY $KIT/seeds.py --pred $S.zarr --z $Z --x0 $((X - HALF)) --x1 $((X + HALF)) \
    --y0 $((Y - HALF)) --y1 $((Y + HALF)) --out $OUT/seeds.tsv --png $OUT/seeds.png || { echo "seeds failed"; exit 2; }
if cmp -s $OUT/seeds.tsv $REPO/results/tiling-306b/seeds_kat_window.tsv; then SEEDS_CHECK=identical; else SEEDS_CHECK=DIFFERENT; fi
echo "seeds $(($(wc -l < $OUT/seeds.tsv) - 1)) ($SEEDS_CHECK to results/tiling-306b/seeds_kat_window.tsv) $(( $(date +%s) - t0 )) s" | tee $OUT/demo.log
# 2. grow (all seeds, or the ONLY subset)
if [ "$ONLY" = all ]; then cp $OUT/seeds.tsv $OUT/seeds_grow.tsv
else awk -F'\t' -v L="$ONLY" 'BEGIN{n=split(L,a,",");for(i=1;i<=n;i++)k[a[i]]=1} NR==1||($1 in k)' $OUT/seeds.tsv > $OUT/seeds_grow.tsv; fi
PY=$PY bash $KIT/tile_grow.sh $OUT/tile $OUT/seeds_grow.tsv $GENS $S.zarr $S.normal-grids 8.64 || { echo "tile_grow failed"; exit 3; }
t1=$(date +%s)
# 3. ink
KPY=${KPY:-$PY} bash $KIT/ink_patches.sh $OUT/tile $CT 8.64 demo306b || { echo "ink_patches failed"; exit 4; }
t2=$(date +%s)
# 4. coverage of the winners' claim surface
W=https://raw.githubusercontent.com/SirProdigle/pherc0306b-first-letters/feabff6c54f95bd8e52a90e0a7a76881ee67d4cc/submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc
mkdir -p $OUT/winners_mesh
for f in x.tif y.tif z.tif meta.json; do
  [ -s $OUT/winners_mesh/$f ] || curl -sfL --max-time 120 -o $OUT/winners_mesh/$f $W/$f || echo "fetch $f failed"
done
printf 'patch\tfrac_within_5\tmedian_dn\n' > $OUT/coverage.tsv
for P in $OUT/tile/patches/w*; do
  c=$(run $PY $KIT/coverage.py $OUT/winners_mesh $P --rows 82:137 --cols 121:341 --tol 5 | tail -1)
  run $PY -c "import json,sys; d=json.loads(sys.argv[1]); print('$(basename $P)', '%.3f' % d['frac_within_tol'], d['median_dn_within_tol'], sep='\t')" "$c" >> $OUT/coverage.tsv
done
BEST=$(tail -n +2 $OUT/coverage.tsv | sort -t$'\t' -k2,2gr | head -1 | cut -f1)
# 5. best patch in the winners' frame vs the v0.2 reference reading view
if [ -n "$BEST" ] && [ -f $OUT/tile/maps/demo306b/$BEST/reader-v3_fwd.png ]; then
  for o in fwd rev; do
    run $PY $REPO/followup/mesh_resample.py $OUT/winners_mesh $OUT/tile/patches/$BEST $OUT/tile/maps/demo306b/$BEST/reader-v3_$o.png \
        $OUT/${BEST}_${o}_in_winners.png --rows 70:150 --cols 100:360 --ppcA 10 --ppcB 10 | tee -a $OUT/resample.txt
  done
  run $PY $KIT/demo_check.py $OUT/${BEST}_fwd_in_winners.png $OUT/${BEST}_rev_in_winners.png \
      $REPO/results/followup-306b/reader-v3_fwd_in_winners_frame_reading.jpg $OUT/check | tee $OUT/check.txt
fi
# 6. rank
run $PY $KIT/rank.py $OUT/tile/maps --out-json $OUT/rank.json --sheet $OUT/rank.png --top 36 > $OUT/rank.txt || echo "rank failed"
{ echo "grow $((t1 - t0)) s, ink $((t2 - t1)) s, total $(( $(date +%s) - t0 )) s; kept patches $(ls $OUT/tile/patches | wc -l)"
  echo "seeds: $SEEDS_CHECK"; echo "best-covering patch: $BEST"; cat $OUT/coverage.tsv; cat $OUT/check.txt 2>/dev/null; } | tee -a $OUT/demo.log
