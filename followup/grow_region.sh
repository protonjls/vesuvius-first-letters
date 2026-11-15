#!/bin/bash
# Grow a fresh surface around a spot of interest with VC3D GrowPatch (vc_grow_seg_from_seed on the public m7 surface
# predictions), render it from the CT volume and run 9 um ink models on it, both depth orders. Runs on CPU only.
#
#   grow_region.sh <out_dir> <seed_x> <seed_y> <seed_z> <generations> <surface_pred_url> <normal_grids_url> <ct_url> <pixel_um>
#
#   seed            a point ON the papyrus sheet, volume voxels (x y z), e.g. a point of an existing mesh near the spot
#   surface_pred    <scroll>/representations/predictions/surfaces/<volume>-surface-...-surface-m7-L0-th0.2.zarr (S3 URL)
#   normal_grids    the matching ...normal-grids directory URL (VC3D streams it; see README "Normal grids")
#   ct_url          the CT OME-Zarr the render samples (<scroll>/volumes/<volume>-....zarr)
#
# Required env:
#   VC3D        command that runs VC3D command-line tools, e.g. "/path/squashfs-root/AppRun" (extracted release AppImage)
#   INKSDK      ink-sdk executable (LimeGS/ink-sdk @2df3ab50)          KPY  python with koine_machines, numpy, pillow, zarr 2,
#   VILLA_SRC   villa source dir holding ink-detection/ and vesuvius/   tifffile, scipy (villa ink-detection uv venv)
#   MODELS_DIR  ink-sdk models dir (ink-sdk fetch --profile ink_9um@reader-v3 --models-dir ...)
# Optional env:
#   WRAP (prefix for every python / ink-sdk call, e.g. an apptainer exec ...), WORK (scratch; default $SLURM_TMPDIR or mktemp),
#   NCPU (threads), NL (render layers, 24), STEP (GrowPatch step in voxels, 20), CACHE_GB (VC3D chunk cache, 10),
#   PROFILES (default "ink_9um@reader-v3 ink_9um@rectina-m4"), SKIP_INFER=1 (stop after the render), GROW_TIMEOUT (s, 10800),
#   KEEP_RENDER=1 (keep render.npy after inference; it is ~1 GB for a 150-generation patch)
# Output in out_dir: mesh_grow/ (tifxyz), params.json, grow.log, ct_ds2.png, render.npy (NL x H x W uint8, 1 px = 1 voxel
#   along the surface; written to WORK and kept in out_dir only with KEEP_RENDER=1 or SKIP_INFER=1), maps/<model>_{fwd,rev}.png (full res), <model>_{fwd,rev}_ds2.png, run.log.
#   Grid node (r, c) of mesh_grow is render pixel (r*STEP, c*STEP) and ds2 pixel (r*STEP/2, c*STEP/2).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; SX=${2:?}; SY=${3:?}; SZ=${4:?}; GENS=${5:?}; SURL=${6:?}; NGURL=${7:?}; URL=${8:?}; PUM=${9:?}
VC3D=${VC3D:?set VC3D}; INKSDK=${INKSDK:-ink-sdk}; KPY=${KPY:-python}; VILLA_SRC=${VILLA_SRC:?set VILLA_SRC}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}; WRAP=${WRAP:-}
NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-4}}; NL=${NL:-24}; STEP=${STEP:-20}
PROFILES=${PROFILES:-"ink_9um@reader-v3 ink_9um@rectina-m4"}
mkdir -p $OUT/maps; OUT=$(cd $OUT && pwd)
T=$(mktemp -d ${WORK:-${SLURM_TMPDIR:-/tmp}}/grow.XXXXXX)
LOG=$OUT/run.log
run() { $WRAP "$@"; }
vc() { env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$NCPU $VC3D "$@"; }
echo "grow_region: seed $SX $SY $SZ gens=$GENS step=$STEP surf=$SURL ngrid=$NGURL ct=$URL um=$PUM NL=$NL profiles=$PROFILES" | tee $LOG
for u in "$SURL/.zattrs" "$NGURL/metadata.json" "$URL/.zattrs"; do
  echo "probe $u $(curl -s -o /dev/null -w '%{http_code}' --max-time 30 $u)" | tee -a $LOG; done
# 1. GrowPatch. VC3D f0e4410 only streams normal grids from <its remote-cache root>/s3/<bucket>/<key> holding a
#    normal-grids-remote.json marker; any other normal_grid_path fails.
mkdir -p $T/home $T/grow
CROOT=$(vc vc_cache_root 2>/dev/null | tail -1); [ -n "$CROOT" ] || { echo "no VC3D cache root (vc_cache_root)"; exit 4; }
NGHOST=${NGURL#https://}; NGBUCKET=${NGHOST%%.s3.*}; NGKEY=${NGHOST#*/}
NGDIR=$CROOT/s3/$NGBUCKET/$NGKEY; mkdir -p $NGDIR
printf '{\n  "url": "%s"\n}\n' "$NGURL" > $NGDIR/normal-grids-remote.json
cat > $OUT/params.json <<EOF
{"mode": "seed", "step_size": $STEP, "min_area_cm": 0.5, "generations": $GENS, "thread_limit": $NCPU,
 "normal_grid_path": "$NGDIR", "voxelsize": $PUM, "cache_size": 4000000000}
EOF
t0=$(date +%s)
timeout ${GROW_TIMEOUT:-10800} env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen \
  OMP_NUM_THREADS=$NCPU $VC3D vc_grow_seg_from_seed -v "$SURL" -t $T/grow -p $OUT/params.json --seed $SX $SY $SZ > $OUT/grow.log 2>&1
echo "grow rc=$? $(( $(date +%s) - t0 )) s" | tee -a $LOG; tail -3 $OUT/grow.log | tee -a $LOG
G=$(ls -d $T/grow/auto_grown_* 2>/dev/null | head -1)
[ -n "$G" ] && [ -f $G/x.tif ] || { echo "no grown surface"; exit 5; }
rm -rf $OUT/mesh_grow; mkdir -p $OUT/mesh_grow; cp $G/x.tif $G/y.tif $G/z.tif $G/meta.json $OUT/mesh_grow/
# 2. render, same settings as the sweep (1 voxel per pixel along the surface, NL layers centred on it)
t0=$(date +%s)
vc vc_render_tifxyz --volume "$URL" --group-idx 0 --scale 1 --segmentation $OUT/mesh_grow --num-slices $NL --slice-step 1 \
   --cache-gb ${CACHE_GB:-10} --zarr-output $T/sv.zarr > $T/render.log 2>&1 || { echo "render failed"; tail -5 $T/render.log; exit 6; }
echo "render $(( $(date +%s) - t0 )) s" | tee -a $LOG
SV=$T/render.npy; run $KPY $KIT/kit_util.py render2npy $T/sv.zarr $SV $OUT/ct_ds2.png | tee -a $LOG
rm -rf $T/sv.zarr
[ "${SKIP_INFER:-0}" = 1 ] && { mv $SV $OUT/render.npy; echo "done (SKIP_INFER)" | tee -a $LOG; rm -rf $T; exit 0; }
# 3. ink models, both depth orders (fwd = layer 0 first, rev = reversed); which one faces the ink depends on the mesh normal
for p in $PROFILES; do
  pn=${p#*@}; pn=${pn%%-step*}; t1=$(date +%s)
  run $INKSDK infer --profile $p --surface-volume $SV --pixel-um $PUM --villa $VILLA_SRC --python $KPY \
      --models-dir $MODELS_DIR --out $T/o --on-degenerate warn --overwrite --cache-dir $T/sdkcache > $T/infer.log 2>&1
  prc=$?
  if [ $prc != 0 ]; then echo "$p infer rc=$prc"; tail -5 $T/infer.log | tee -a $LOG; rm -rf $T/o $T/sdkcache; continue; fi
  run $KPY $KIT/kit_util.py prob2png $T/o $OUT/maps/$pn $OUT/$pn || echo "$p convert failed" | tee -a $LOG
  rm -rf $T/o $T/sdkcache
  echo "$pn infer $(( $(date +%s) - t1 )) s" | tee -a $LOG
done
[ "${KEEP_RENDER:-0}" = 1 ] && mv $SV $OUT/render.npy
rm -rf $T
echo "done" | tee -a $LOG
