#!/bin/bash
# False-positive controls for one window of a surface: render N layers around the mesh, build control variants
# (variants.py: centred at d-2..d+2, 15 voxels off the surface on either side, depth-shuffled), run the ink models on
# every variant, then persistence statistics and a montage (confirm_stats.py). Real surface ink persists at d+-1, d+-2
# and vanishes off the surface and under depth shuffling; a model artefact or an ink-like CT feature off the sheet does not.
#
#   confirm_region.sh <out_dir> <tifxyz dir> <ct_url> <pixel_um> <x0> <y0> <side>
#
#   x0 y0 side: square window in RENDER pixels of the full-mesh render at scale 1 (grow_region.sh: grid node (r, c) =
#   render pixel (r*STEP, c*STEP)).
# Env: as grow_region.sh (VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR, WRAP, WORK, NCPU, CACHE_GB) plus
#   PROFILES (default "ink_9um@reader-v3 ink_9um@rectina-m4"), N (render layers, 56: the off+-15 variants need N >= 54),
#   ORDER (fwd|rev: depth order shown in the montage, default rev), VSEL (space-separated variant names to run; default all)
# Output in out_dir: maps/<variant>/<model>_{fwd,rev}.png, ct_d0.png, manifest.json, stats.json, montage_<order>.png, run.log
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; MESH=${2:?tifxyz}; URL=${3:?}; PUM=${4:?}; X0=${5:?}; Y0=${6:?}; SIDE=${7:?}
VC3D=${VC3D:?set VC3D}; INKSDK=${INKSDK:-ink-sdk}; KPY=${KPY:-python}; VILLA_SRC=${VILLA_SRC:?set VILLA_SRC}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}; WRAP=${WRAP:-}
NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-4}}; N=${N:-56}; ORDER=${ORDER:-rev}
PROFILES=${PROFILES:-"ink_9um@reader-v3 ink_9um@rectina-m4"}
mkdir -p $OUT/maps; OUT=$(cd $OUT && pwd); MESH=$(cd $MESH && pwd)
T=$(mktemp -d ${WORK:-${SLURM_TMPDIR:-/tmp}}/confirm.XXXXXX); mkdir -p $T/home
LOG=$OUT/run.log
run() { $WRAP "$@"; }
echo "confirm_region: mesh $MESH ct $URL um=$PUM window x0=$X0 y0=$Y0 side=$SIDE N=$N profiles=$PROFILES" | tee $LOG
t0=$(date +%s)
env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$NCPU \
  $VC3D vc_render_tifxyz --volume "$URL" --group-idx 0 --scale 1 --segmentation $MESH \
  --num-slices $N --slice-step 1 --cache-gb ${CACHE_GB:-10} --crop-x $X0 --crop-y $Y0 --crop-width $SIDE --crop-height $SIDE \
  --zarr-output $T/sv.zarr > $T/render.log 2>&1 || { echo "render failed"; tail -5 $T/render.log; exit 4; }
echo "render $(( $(date +%s) - t0 )) s" | tee -a $LOG
run $KPY $KIT/variants.py --sv $T/sv.zarr --outdir $T/var > $OUT/manifest.json || { echo "variants failed"; exit 5; }
run $KPY $KIT/kit_util.py ctd0 $T/var $OUT/ct_d0.png
rm -rf $T/sv.zarr
VARS=$(ls $T/var | sed 's/\.npy$//' | sort)
if [ -n "${VSEL:-}" ]; then VARS=$(for v in $VARS; do for w in $VSEL; do [ "$v" = "$w" ] && echo $v; done; done); fi
echo "variants: $(echo $VARS)" | tee -a $LOG
for p in $PROFILES; do
  pn=${p#*@}; pn=${pn%%-step*}
  for vn in $VARS; do
    t1=$(date +%s)
    run $INKSDK infer --profile $p --surface-volume $T/var/$vn.npy --pixel-um $PUM --villa $VILLA_SRC --python $KPY \
        --models-dir $MODELS_DIR --out $T/o --on-degenerate warn --overwrite --cache-dir $T/sdkcache > $T/infer.log 2>&1
    prc=$?
    if [ $prc != 0 ]; then echo "$vn $p infer rc=$prc"; tail -3 $T/infer.log | tee -a $LOG; rm -rf $T/o; continue; fi
    mkdir -p $OUT/maps/$vn
    run $KPY $KIT/kit_util.py prob2png $T/o $OUT/maps/$vn/$pn || echo "$vn $p convert failed" | tee -a $LOG
    rm -rf $T/o
    echo "$vn $pn $(( $(date +%s) - t1 )) s" | tee -a $LOG
  done
done
rm -rf $T
run $KPY $KIT/confirm_stats.py --dir $OUT --order $ORDER 2>&1 | tee -a $LOG
echo "done" | tee -a $LOG
