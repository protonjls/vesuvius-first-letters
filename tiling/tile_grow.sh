#!/bin/bash
# Tiling step 2: cover every predicted sheet of a slice window with masked VC3D GrowPatch patches. CPU only.
#
#   tile_grow.sh <out_dir> <seeds.tsv> <generations> <surface_pred_url> <normal_grids_url> <pixel_um>
#
#   seeds.tsv       seeds.py output (idx tile_r tile_c x y z runlen coherence)
#   surface_pred    <scroll>/representations/predictions/surfaces/<volume>-surface-...-surface-m7-L0-th0.2.zarr (S3 URL)
#   normal_grids    the matching ...normal-grids directory URL
#
# Per seed, in file order, skipping seeds already done or covered by an earlier patch:
#   vc_grow_seg_from_seed (mode seed, step STEP) -> the surface prediction rendered along the mesh (7 layers) ->
#   patch.py mask: keep the on-sheet connected component holding the seed (off-sheet nodes set to -1), mark every
#   seed within DEDUP voxels of the kept surface as covered -> <out_dir>/patches/wNNNN/{x,y,z.tif,meta.json}.
# Resumable: <out_dir>/{seeds.tsv,done.txt,covered.txt,patches.jsonl} hold the state; rerun the same command to continue.
#
# Required env: VC3D (command running VC3D command-line tools, e.g. /path/squashfs-root/AppRun of the f0e4410 AppImage),
#   PY (python with numpy, scipy, tifffile, zarr 2, opencv, numcodecs).
# Optional env: WRAP (prefix for python calls, e.g. an apptainer exec ...), WORK (scratch; default $SLURM_TMPDIR or /tmp),
#   NCPU, STEP (20), MAXGROWS (200 per run), GROW_TIMEOUT (s per grow, 1800), DEDUP (5 vox), MINAREA (0.3 cm2),
#   PART (k/K: only seeds of the k-th of K contiguous tile-column groups, for parallel jobs with separate out_dirs).
set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?out_dir}; SEEDS=${2:?seeds.tsv}; GENS=${3:?generations}; SURL=${4:?surface_pred_url}; NGURL=${5:?normal_grids_url}
PUM=${6:?pixel_um}
VC3D=${VC3D:?set VC3D}; PY=${PY:-python3}; WRAP=${WRAP:-}
NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-4}}; STEP=${STEP:-20}; MAXGROWS=${MAXGROWS:-200}
mkdir -p $OUT/patches; OUT=$(cd $OUT && pwd)
T=$(mktemp -d ${WORK:-${SLURM_TMPDIR:-/tmp}}/tile.XXXXXX)
LOG=$OUT/run.log
run() { $WRAP "$@"; }
vc() { env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=$NCPU $VC3D "$@"; }
echo "tile_grow: seeds=$SEEDS gens=$GENS step=$STEP part=${PART:-all} max=$MAXGROWS surf=$SURL um=$PUM $(date)" | tee -a $LOG
[ -f $OUT/seeds.tsv ] || cp $SEEDS $OUT/seeds.tsv
touch $OUT/done.txt $OUT/covered.txt
# VC3D f0e4410 streams normal grids only from <its remote-cache root>/s3/<bucket>/<key> holding a
# normal-grids-remote.json marker; any other normal_grid_path fails.
mkdir -p $T/home
CROOT=$(vc vc_cache_root 2>/dev/null | tail -1); [ -n "$CROOT" ] || { echo "no VC3D cache root (vc_cache_root)"; exit 4; }
NGHOST=${NGURL#https://}; NGBUCKET=${NGHOST%%.s3.*}; NGKEY=${NGHOST#*/}
NGDIR=$CROOT/s3/$NGBUCKET/$NGKEY; mkdir -p $NGDIR
printf '{\n  "url": "%s"\n}\n' "$NGURL" > $NGDIR/normal-grids-remote.json
cat > $OUT/params.json <<EOF
{"mode": "seed", "step_size": $STEP, "min_area_cm": 0.1, "generations": $GENS, "thread_limit": $NCPU,
 "normal_grid_path": "$NGDIR", "voxelsize": $PUM, "cache_size": 4000000000}
EOF
n=0; t00=$(date +%s)
while [ $n -lt $MAXGROWS ]; do
  NX=$(run $PY $KIT/patch.py next $OUT/seeds.tsv $OUT/done.txt $OUT/covered.txt ${PART:+--part $PART} | tail -1)
  [ -n "$NX" ] || { echo "no seeds left" | tee -a $LOG; break; }
  read -r IDX SX SY SZ LEFT <<< "$NX"
  [[ $IDX =~ ^[0-9]+$ ]] || { echo "bad next line: $NX" | tee -a $LOG; exit 6; }
  W=w$(printf %04d $IDX); G=$T/grow_$IDX; rm -rf $G; mkdir -p $G
  t0=$(date +%s)
  timeout ${GROW_TIMEOUT:-1800} env HOME=$T/home XDG_CACHE_HOME=$T/xdgcache XDG_CONFIG_HOME=$T/xdgconfig QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=$NCPU $VC3D vc_grow_seg_from_seed -v "$SURL" -t $G -p $OUT/params.json --seed $SX $SY $SZ \
    > $T/grow.log 2>&1; grc=$?
  tg=$(( $(date +%s) - t0 ))
  GD=$(ls -d $G/auto_grown_* 2>/dev/null | head -1)
  if [ -z "$GD" ] || [ ! -f $GD/x.tif ]; then
    echo "{\"seed\": $IDX, \"action\": \"grow_failed\", \"rc\": $grc, \"t_grow\": $tg}" >> $OUT/patches.jsonl
    echo $IDX >> $OUT/done.txt; rm -rf $G; n=$((n + 1)); continue
  fi
  t1=$(date +%s)
  vc vc_render_tifxyz --volume "$SURL" --group-idx 0 --scale 1 --segmentation $GD --num-slices 7 --slice-step 1 \
     --cache-gb 8 --zarr-output $T/m7.zarr > $T/render.log 2>&1 || { tail -3 $T/render.log | tee -a $LOG; }
  tr=$(( $(date +%s) - t1 ))
  ST=$(run $PY $KIT/patch.py mask $GD $T/m7.zarr $OUT/seeds.tsv $IDX $OUT/patches/$W $OUT/covered.txt \
       --dedup ${DEDUP:-5} --minarea ${MINAREA:-0.3} --um $PUM | tail -1)
  if [[ $ST == \{* ]]; then
    echo "${ST%\}}, \"t_grow\": $tg, \"t_m7\": $tr, \"grow_rc\": $grc, \"left\": $LEFT}" >> $OUT/patches.jsonl
  else
    echo "{\"seed\": $IDX, \"action\": \"mask_failed\", \"msg\": \"$(echo $ST | tr -d '"' | head -c 200)\"}" >> $OUT/patches.jsonl
  fi
  echo $IDX >> $OUT/done.txt
  rm -rf $G $T/m7.zarr
  n=$((n + 1))
  echo "seed $IDX grow ${tg}s m7 ${tr}s left $LEFT -> $(echo $ST | head -c 160)" | tee -a $LOG
done
echo "grows this run $n, $(( $(date +%s) - t00 )) s; patches kept $(ls $OUT/patches | wc -l)" | tee -a $LOG
rm -rf $T
