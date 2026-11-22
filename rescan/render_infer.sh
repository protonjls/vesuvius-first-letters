#!/bin/bash
# Render a window of a tifxyz mesh on a ~2.4 um rescan and run the Vesuvius Challenge canonical 2.4 um ink model on it.
# The mesh may live in another volume of the same scroll (e.g. a 9.362 um segment): pass the official villa
# transform.json that maps the rescan (moving) to that volume (fixed) with ':inv', and VC3D maps every mesh vertex into
# the rescan before sampling along the grid normals (vc_render_tifxyz --affine). The canonical model reads one depth
# order only, so both orders are run as separate inputs; pick the one that matches known ink (fwd/rev differ strongly).
#
#   render_infer.sh <out_dir> <tifxyz dir> <ct_url> <pixel_um> <affine.json[:inv] | -> <x0> <y0> <width> <height>
#
#   x0 y0 width height: window in render pixels of the full-mesh render at --scale 1 in the TARGET volume. With an affine,
#   1 mesh-grid voxel becomes sA rescan voxels (sA = cube root of |det| of the inverse affine, ~3.9 for 9.362 -> 2.4 um),
#   so a window of the 9 um render at (x, y) sits at (x * sA, y * sA) here.
# Env: VC3D (command that runs VC3D tools, e.g. <AppImage dir>/squashfs-root/AppRun), INKSDK, KPY, VILLA_SRC, MODELS_DIR,
#   WRAP (prefix for KPY/INKSDK, e.g. an apptainer exec), WORK (scratch dir), NCPU, CACHE_GB (VC3D chunk cache, 10),
#   N (render layers, 96), DEPTH (layers given to the model, 62 = the canonical profile's input depth, centred),
#   PROFILE (canonical_2um@r152-3ddec-v2-l5-epoch13), ORDERS ("fwd rev").
# Output in out_dir: maps/d0_<order>.png (probability x 255), maps/d0_<order>.npy (float), ct.png (mean of the 4 central
#   layers), manifest.json (variants.py), run.log.
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; MESH=${2:?tifxyz}; URL=${3:?ct_url}; PUM=${4:?pixel_um}; AFF=${5:?affine or -}
X0=${6:?}; Y0=${7:?}; CW=${8:?}; CH=${9:?}
VC3D=${VC3D:?set VC3D}; INKSDK=${INKSDK:-ink-sdk}; KPY=${KPY:-python}; VILLA_SRC=${VILLA_SRC:?set VILLA_SRC}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}; WRAP=${WRAP:-}
NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-4}}; N=${N:-96}; DEPTH=${DEPTH:-62}; ORDERS=${ORDERS:-"fwd rev"}
PROFILE=${PROFILE:-canonical_2um@r152-3ddec-v2-l5-epoch13}
mkdir -p $OUT/maps; OUT=$(cd $OUT && pwd); MESH=$(cd $MESH && pwd)
T=$(mktemp -d ${WORK:-${SLURM_TMPDIR:-/tmp}}/rescan.XXXXXX); mkdir -p $T/home
LOG=$OUT/run.log
run() { $WRAP "$@"; }
AARG=""; [ "$AFF" != "-" ] && AARG="--affine $AFF"
echo "render_infer: mesh $MESH ct $URL um=$PUM affine=$AFF window x0=$X0 y0=$Y0 ${CW}x$CH N=$N depth=$DEPTH profile=$PROFILE" | tee $LOG
t0=$(date +%s)
env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$NCPU \
  $VC3D vc_render_tifxyz --volume "$URL" --group-idx 0 --scale 1 --segmentation $MESH $AARG \
  --num-slices $N --slice-step 1 --cache-gb ${CACHE_GB:-10} --crop-x $X0 --crop-y $Y0 --crop-width $CW --crop-height $CH \
  --zarr-output $T/sv.zarr > $T/render.log 2>&1 || { echo "render failed"; tail -5 $T/render.log | tee -a $LOG; exit 4; }
grep -i "affine\|voxel size\|scale" $T/render.log | head -8 | tee -a $LOG
echo "render $(( $(date +%s) - t0 )) s" | tee -a $LOG
run $KPY $KIT/../followup/variants.py --sv $T/sv.zarr --outdir $T/var --depth $DEPTH --offsets 0 --off ${OFF:-58} > $OUT/manifest.json \
  || { echo "variants failed"; exit 5; }
run $KPY $KIT/../followup/kit_util.py ctd0 $T/var $OUT/ct.png
rm -rf $T/sv.zarr
[ -f $T/var/d0.npy ] || { echo "no d0 variant (N < DEPTH?)"; exit 5; }
for ord in $ORDERS; do
  f=$T/var/d0.npy
  if [ $ord = rev ]; then
    run $KPY -c "import numpy as np; v = np.load('$f'); np.save('$T/var/d0_rev.npy', np.ascontiguousarray(v[::-1]))"
    f=$T/var/d0_rev.npy
  fi
  t1=$(date +%s)
  run $INKSDK infer --profile $PROFILE --surface-volume $f --pixel-um $PUM --villa $VILLA_SRC --python $KPY \
      --models-dir $MODELS_DIR --out $T/o --on-degenerate warn --overwrite --cache-dir $T/sdkcache > $T/infer.log 2>&1
  prc=$?
  if [ $prc != 0 ] || [ ! -f $T/o/probability.npy ]; then echo "d0 $ord infer rc=$prc"; tail -5 $T/infer.log | tee -a $LOG; rm -rf $T/o; continue; fi
  run $KPY -c "
import numpy as np
from PIL import Image
a = np.nan_to_num(np.load('$T/o/probability.npy').astype(np.float32))
np.save('$OUT/maps/d0_$ord.npy', a)
Image.fromarray(np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)).save('$OUT/maps/d0_$ord.png')
print('d0 $ord map', a.shape, 'mean %.4f p99 %.4f' % (a.mean(), np.percentile(a, 99)))" | tee -a $LOG
  rm -rf $T/o
  echo "d0 $ord infer $(( $(date +%s) - t1 )) s" | tee -a $LOG
done
rm -rf $T
echo "done" | tee -a $LOG
