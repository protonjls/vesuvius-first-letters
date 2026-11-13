#!/bin/bash
# Fetch the PHerc1447 test surfaces used by run_adapt.sh and tta/tta_eval.sh from the Hugging Face dataset
# YoussefMoNader/ink-8um-pherc1447-surfaces (CC BY-NC 4.0; 24-layer 8.64 um renders + refined ink labels).
#   fetch_1447.sh <DATA_DIR>        -> <DATA_DIR>/{w058,w060,w062}/layers/00..23.tif, {w058,w060}/labels/{inklabels,mask}.png
# Re-running skips files that already exist. About 1.6 GB.
set -euo pipefail
D=${1:?usage: fetch_1447.sh DATA_DIR}
HFB=https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/main
for w in w058 w060 w062; do
  mkdir -p $D/$w/layers $D/$w/labels
  for i in $(seq -w 0 23); do
    [ -s $D/$w/layers/$i.tif ] || curl -sSL --fail --retry 3 -o $D/$w/layers/$i.tif $HFB/$w/layers/$i.tif
  done
  [ $w = w062 ] && continue          # w062 has no labels in the dataset: it is the unlabelled training winding
  for f in inklabels.png mask.png; do
    [ -s $D/$w/labels/$f ] || curl -sSL --fail --retry 3 -o $D/$w/labels/$f $HFB/$w/labels/$f
  done
done
echo "fetched into $D"
