#!/bin/bash
# Tiling step 3: render every kept patch from the CT and run ink models on it, both depth orders, in the layout rank.py
# reads. CPU works (ink-sdk falls back to CPU); a GPU makes the models ~10x faster.
#
#   ink_patches.sh <tile_out_dir> <ct_url> <pixel_um> [<name>]
#
# Reads <tile_out_dir>/patches/wNNNN (tile_grow.sh). Writes <tile_out_dir>/maps/<name>/wNNNN/{ct.png, <model>_fwd.png,
# <model>_rev.png, meta.json}: PNGs at half the render resolution (png_ds 2; ct = mean of the 4 central layers, maps =
# round(p*255)), meta.json {pixel_um, png_ds, shape}. Patches with a meta.json there already are skipped (resumable).
# <name> defaults to the basename of <tile_out_dir>.
#
# Required env: VC3D, INKSDK, KPY, VILLA_SRC, MODELS_DIR as in ../followup/grow_region.sh.
# Optional env: WRAP, WORK, NCPU, NL (24 render layers), CACHE_GB (10), PROFILES (default "ink_9um@reader-v3"),
#   WSEL (regex on wNNNN: only matching patches, for parallel jobs), KEEP_FULL=1 (also keep full-res maps).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FU=$KIT/../followup
TOUT=${1:?tile_out_dir}; URL=${2:?ct_url}; PUM=${3:?pixel_um}
TOUT=$(cd $TOUT && pwd); NAME=${4:-$(basename $TOUT)}
VC3D=${VC3D:?set VC3D}; INKSDK=${INKSDK:-ink-sdk}; KPY=${KPY:-python}; VILLA_SRC=${VILLA_SRC:?set VILLA_SRC}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}; WRAP=${WRAP:-}
NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-4}}; NL=${NL:-24}
PROFILES=${PROFILES:-"ink_9um@reader-v3"}
OUTM=$TOUT/maps/$NAME; mkdir -p $OUTM
T=$(mktemp -d ${WORK:-${SLURM_TMPDIR:-/tmp}}/ink.XXXXXX)
LOG=$TOUT/ink_${NAME}.log
run() { $WRAP "$@"; }
vc() { env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$NCPU $VC3D "$@"; }
mkdir -p $T/home
echo "ink_patches: $TOUT -> $OUTM ct=$URL um=$PUM NL=$NL profiles=$PROFILES wsel=${WSEL:-all} $(date)" | tee -a $LOG
for P in $TOUT/patches/w*; do
  w=$(basename $P); [ -f $P/x.tif ] || continue
  [ -z "${WSEL:-}" ] || [[ $w =~ $WSEL ]] || continue
  R=$OUTM/$w; [ -f $R/meta.json ] && continue
  mkdir -p $R; t0=$(date +%s)
  vc vc_render_tifxyz --volume "$URL" --group-idx 0 --scale 1 --segmentation $P --num-slices $NL --slice-step 1 \
     --cache-gb ${CACHE_GB:-10} --zarr-output $T/sv.zarr > $T/render.log 2>&1 || { echo "$w render failed"; tail -3 $T/render.log; rm -rf $T/sv.zarr; continue; }
  SV=$T/render.npy; run $KPY $FU/kit_util.py render2npy $T/sv.zarr $SV $R/ct.png > /dev/null; rm -rf $T/sv.zarr
  for p in $PROFILES; do
    pn=${p#*@}; pn=${pn%%-step*}
    run $INKSDK infer --profile $p --surface-volume $SV --pixel-um $PUM --villa $VILLA_SRC --python $KPY \
        --models-dir $MODELS_DIR --out $T/o --on-degenerate warn --overwrite --cache-dir $T/sdkcache > $T/infer.log 2>&1 \
      || { echo "$w $p infer failed"; tail -3 $T/infer.log; rm -rf $T/o $T/sdkcache; continue; }
    run $KPY $FU/kit_util.py prob2png $T/o $T/full_$pn $T/m_$pn
    mv $T/m_${pn}_fwd_ds2.png $R/${pn}_fwd.png; mv $T/m_${pn}_rev_ds2.png $R/${pn}_rev.png
    if [ "${KEEP_FULL:-0}" = 1 ]; then mv $T/full_${pn}_fwd.png $R/${pn}_fwd_full.png; mv $T/full_${pn}_rev.png $R/${pn}_rev_full.png; fi
    rm -rf $T/o $T/sdkcache $T/full_${pn}_*.png
  done
  run $KPY -c "import json,numpy as np; a=np.load('$SV', mmap_mode='r'); json.dump({'patch': '$w', 'shape': list(a.shape), 'pixel_um': $PUM, 'png_ds': 2, 'ct_url': '$URL'}, open('$R/meta.json','w'))"
  rm -f $SV
  echo "$w $(( $(date +%s) - t0 )) s" | tee -a $LOG
done
rm -rf $T
echo "done: $(ls $OUTM | wc -l) patches in $OUTM" | tee -a $LOG
