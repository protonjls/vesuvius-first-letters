# Environment

All scripts call one Python (`$PY`) that can import `koine_machines` (villa's ink-detection package) plus
`tifffile`, `zarr` 2.x, `numcodecs`, `scipy` and `pillow`. Every dependency comes from villa's uv lockfile, so nothing
here pins its own versions.

## Build (tested 2026-11-13, Linux x86_64, CUDA 12.8 driver, H100)
```bash
git clone --branch merge-ink-pipelines https://github.com/ScrollPrize/villa.git
cd villa && git checkout 47a9c37f6            # the commit every result in results/ was produced with
cd ink-detection
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --frozen --python 3.12   # pin 3.12: uv may otherwise pick a newer CPython
export PY=$PWD/.venv/bin/python
$PY -c "import koine_machines.inference.infer, koine_machines.training.train; print('ok')"
```
That venv holds torch 2.10 (cu128), `koine_machines`, `vesuvius` and zarr 2.18. On a cluster, build it inside a
container and set `WRAP` to the container prefix (for example
`WRAP="apptainer exec --nv -B /data img.sif"`); the scripts prefix every Python call with `$WRAP`.

## Model weights
- reader-v3: `reader-v3.pth` from DomRusso2's reader-v3 release (MIT). The sha256 is `ea4054…84d8a`, as pinned by
  ink-sdk profile `ink_9um-reader-v3`.
- rectina-m4: `rectina/m4/ckpt_020000.pth`, fetched by ink-sdk (`ink-sdk fetch --profile ink_9um@rectina-m4`).

[LimeGS/ink-sdk](https://github.com/LimeGS/ink-sdk) at `2df3ab50` downloads both and checks their checksums:
`ink-sdk fetch --profile ink_9um@reader-v3 --models-dir M` (and the same for `ink_9um@rectina-m4`).

## Data
- PHerc1447 w058/w060/w062: `scroll_adapt/fetch_1447.sh DATA` downloads them from
  [YoussefMoNader/ink-8um-pherc1447-surfaces](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces)
  (CC BY-NC 4.0): 24-layer 8.64 µm surface renders, plus refined ink labels for w058 and w060.
- PHerc0343 (optional TTA test): a 66-layer render of the winning First Letters mesh from
  [Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters), made with
  VC3D's `vc_render_tifxyz` (the render needs about 1 GB of streamed chunks), together with that repository's
  `inputs/concat_w047-w048_R5B2_z9500-11000_letter_boxes.json`.

## Compute
On a MIG 2g.20gb slice of an H100, one scroll_adapt run (teachers, no-op control, a 2000-iteration fine-tune at
batch 4 and four checkpoint evaluations) takes about 30 min. One tta_eval run over w058/w060/343 takes about 20 min.
