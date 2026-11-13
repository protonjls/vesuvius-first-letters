# Notes: fine-tuning reader-v3 with koine_machines on your own segments (pseudo-labels + ignore mask)

Compiled while building scroll_adapt/ (2026-11-13). Every statement cites the code it was read from; items marked
UNVERIFIED were not checked against code or a run. Items verified by runs since are marked VERIFIED with the run.

Sources read 2026-11-13. Citations `file:line` are at villa `47a9c37f6` (merge-ink-pipelines); paths are
relative to `ink-detection/koine_machines/` unless prefixed `vesuvius/` (= `vesuvius/src/vesuvius/`) or `configs/`
(= `ink-detection/configs/`). ink-sdk at `2df3ab5`. Statements marked UNVERIFIED were not checked against code or a run.

## 0. What reader-v3 actually is (read from the checkpoint, no torch)
- `reader-v3.pth` (sha256 `ea4054…84d8a` per
  `ink-sdk/src/ink_sdk/profiles/ink_9um-reader-v3.toml:17`). Top-level keys: `model` (508 tensors), `config`, `step`=19999.
  **No `optimizer`, no `lr_scheduler`, no `ema_model`.** Read by unpickling `reader-v3/data.pkl` from the zip with torch
  classes stubbed (no torch needed).
- Embedded config == `train_config.json` of the DomRusso2 reader-v3 release (checked key by key): `mode=flat`,
  `model_type=vesuvius_unet_3d_stem_2d`, `patch_size=crop_size=[17,384,384]`, `in_channels=1`,
  `model_config={autoconfigure:true, basic_encoder_block:BasicBlockD, basic_decoder_block:ConvBlock, stem_channels:16,
  z_projection_mode:none, features_per_stage:[32,64,128,256,320,320], n_blocks_per_stage:[1,3,4,6,6,6],
  n_conv_per_stage_decoder:[1,1,1,1,1], strides=pool_op_kernel_sizes=[[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]]}`,
  `image_normalization={robust_mad, 1, 99}`, `flat_z_window_jitter={enabled, window_depth 17, max_offset 2, forbidden}`,
  `loss={bce_label_smoothing 0.5, dice_label_smoothing 0}`, `mixed_precision=fp16`, sgd lr 0.003, warmup 200,
  batch 10, 20000 it, `ema.enabled=false`, and it was itself trained with `"checkpoint": <v2 ckpt>, "weights_only": true`.

## 1. On-disk layout of a training segment (flat mode)
- **Discovery** (`data/ink_dataset.py:1483-1573`): for each `datasets[i]`, every sub-directory of `segments_path`
  (sorted, `unused` skipped) is a segment, filtered by the optional allowlist `segments` (`:1487-1495`).
  With `surface_volume_paths` given and flat mode, no tifxyz (`x.tif`) is needed (`:1496-1505`); the volume path is
  looked up by relpath or dir name (`:1510-1525`), else `surface_volume_path` (only with exactly one allowlisted
  segment, `:1526-1533`), else `<segdir>/<segname>.zarr` (`:1535`). Missing volume or labels raise (`:1560-1571`).
- **Label names** (`data/segment.py:3-8,160-237`): files directly inside the segment dir named
  `<segdir_name>_inklabels.zarr`, `<segdir_name>_supervision_mask.zarr`, optional `<segdir_name>_validation_mask.zarr`
  (prefix must equal the dir name, `:181`; training discovery asks for `.zarr`, `ink_dataset.py:1552`).
  Optional version suffix `_vN` (`segment.py:19-22`): without `label_version` the highest N is taken **per kind
  independently** (`:219-232`); with `label_version:"vN"` inklabels+supervision of that N are required and the
  validation mask is taken only from the same N (`:190-210`).
- **Zarr container** (`common/common.py:136-194`): `zarr.open(path, path=str(resolution))`, so volume AND every label
  must be a zarr **v2** group (env pins `zarr==2.18.7`, `ink-detection/pyproject.toml:17`) holding an array at key
  `str(volume_scale)` (normally `"0"`). Labels are opened with the same `resolution=patch.segment.scale` as the image
  (`ink_dataset.py:1964,1972-1973`).
- **Volume**: 3D, axis order **Z,Y,X** (crops index `[z0:z1, y0:y1, x0:x1]`, `common.py:202-241`); Z = layers along the
  normal, ascending index = "forward". Any integer dtype is read and cast to float32 before normalisation
  (`ink_dataset.py:1994-1999`); use **uint8** (ink-sdk refuses non-uint8, `ink_sdk/io.py:138-142`).
  Patch Z window: `surface = Z//2`, `z0 = surface - 17//2` (`data/patch_finding/default.py:8-18,72`); with jitter
  ±2 the window must stay inside real layers or it raises (`ink_dataset.py:91-163,1965-1966`).
  **=> Z >= 21 with jitter on (21 = what reader-v3 trained on); Z == 17 requires `max_offset: 0` or jitter off.**
- **Label arrays are 3D [Z,Y,X], same Y,X grid and same Z axis as the volume.** In flat mode labels are cropped with
  the *image's* jittered Z-bbox (`ink_dataset.py:1953-1991`) and reduced by max over Z:
  `targets = amax(inklabels, Z) > 0`, `ignore = amax(supervision_mask, Z) <= 0` (`training/train.py:740-742`).
  Patch finding reads the label's own centre slice `shape[0]//2` (`default.py:92,101,163`).
  **Recipe: label Z == volume Z, 2D map written into every Z slice** (robust to jitter; centre-only also works since
  every window contains `Z//2`).
- **Values**: uint8 0/1 (anything >0 counts; 0/255 also works in flat mode). Spatial augments for a [17,384,384] patch
  are Rot90/transpose in the YX plane only (`vesuvius/models/augmentation/pipelines/training_transforms.py:172-182`),
  no interpolation, so labels stay binary and Z order is never flipped.
- **Lower-resolution labels: NO.** The label level `volume_scale` is cropped with the image bbox; a smaller array is
  silently zero-padded (`common.py:202-241`), not resampled. Only `patch_finding_scale` reads coarser levels (volume
  and masks) for patch *finding* (`default.py:57-74,84-88`), so a pyramid is needed only if you set it.
- **Do not use `preprocessing/create_label_zarrs.py` unchanged**: it writes depth-65 labels with the 2D map only in
  slice 32 (`create_label_zarrs.py:43-46,300-310,588-602`). With a 21-layer volume the crop (z 0..21) never reaches
  slice 32 -> all-zero supervision -> every pixel ignored, with no error (patch finding still succeeds at slice 32).
- Chunking: free. Suggest `(Z,512,512)`, Blosc zstd, `write_empty_chunks=False` (build volume+labels on node-local disk:
  a zarr is many small files).
- Pseudo-label grid: koine/ink-sdk maps have the volume's H,W (`ink_sdk/adapters/ink9um.py:174-178`), so maps made
  from the same zarr are already on the label grid.

## 2. `datasets` block and sampling
Required top-level keys (KeyError otherwise): `patch_size`, `patch_overlap`, `patch_min_labeled_coverage`
(`default.py:113,167`), `seed`, `batch_size`, `num_iterations`, `targets.ink` (`train.py:315`), `in_channels`
(`models/make_model.py:25`), `out_dir`, `datasets`. Per dataset: `segments_path`, `volume_scale`
(`ink_dataset.py:1486,1540`), plus `surface_volume_paths` (recommended) and `segments` (allowlist).
`sampling_strategy` (`train.py:413-456`): `uniform` (default; plain shuffled DataLoader, **needs no `sampling_*`
keys**), `scroll_segment_balanced` (needs `sampling_scroll` etc., `training/samplers.py:333-336`),
`fixed_scroll_prior_stratified` (needs `fixed_scroll_prior` with seed == `seed`). Use **`uniform`**.
Patch grid: corners every `int(patch_size[1]*patch_overlap)` px over nonzero supervision|validation at the centre
slice (`default.py:100-121`); `patch_overlap` is a *stride fraction* (0.25 -> 96 px stride at 384, ~16x overlap).
A segment whose supervision and validation are both empty raises (`default.py:109-110`); patches may hang over the
mask/volume edge (zero-padded). `patch_min_labeled_coverage: 0.0` keeps ink-free patches (`default.py:167`).
`segment.py:240` imports `patch_finding.default` directly, so `patch_finding_type: subtiling` is ignored on this path.

## 3. Weights-only init from reader-v3; keeping its contract
- `config["checkpoint"]` (relative paths resolve against the config file's dir) and `config["weights_only"]`
  (`models/load_checkpoint.py:36-59`). With `weights_only: true`, `restore_training_state` loads `model` and returns
  step 0 (`:62-77`); optimizer/scheduler start fresh. **`weights_only: false` would raise KeyError on reader-v3**
  (no optimizer/scheduler, `:79-84`). Load is strict (`model.load_state_dict`, with `module.` prefix retries,
  `:6-33`): any architecture mismatch fails loudly. It happens after He init (`train.py:562-563,599-610`).
- The model is built from **our** config, not the checkpoint's (`train.py:486`). `autoconfigure:true` derives the
  net from `patch_size`, but explicit `features_per_stage`, `n_blocks_per_stage`, `n_conv_per_stage_decoder`,
  `strides`/`pool_op_kernel_sizes` override it (`vesuvius/models/build/build_network_from_config.py:355-386,405-450,
  504-515`); conv kernel sizes stay auto. => **copy `model_type`, the whole `model_config`, `targets`,
  `in_channels`, `patch_size [17,384,384]`, `image_normalization`, `flat_z_window_jitter`, `loss` verbatim from §0.**
  The 2D U-Net needs Y,X divisible by 32 (5 stride-2 stages).
- Inference rebuilds everything from the fine-tuned checkpoint's saved `config` (`train.py:771-776`;
  `inference/infer.py:1331-1365`), so whatever the fine-tune config says becomes the inference contract.

## 4. Validation
- Validation patches = patches touching `*_validation_mask` (`default.py:133-160`); for them the validation mask
  replaces the supervision mask (`supervision_mask_override`, `:158`), i.e. it is the scored region (ink and blank).
  Training never sees validation pixels: training patches need supervision outside it (`default.py:142-145`) and
  its voxels are zeroed from training supervision in every crop (`ink_dataset.py:349-370,1977-1990`). No Z jitter
  or augmentation on validation (`ink_dataset.py:144`; `train.py:393-398`).
- Runs at `step % val_every == 0 and step > 0` (`train.py:922`), `min(len(val_dl), val_steps)` batches (`:953`),
  logs `out_dir/validation_metrics.jsonl` incl. `val_bce_unsmoothed` and balanced accuracy at p=0.5
  (`:1128-1143`), keeps `best_<metric>.pth` (`:1144-1165`).
- **Held-out segment**: give it a non-empty `_validation_mask`, an all-zero `_supervision_mask` (must exist,
  `ink_dataset.py:1565-1571`) and `_inklabels`; it then yields validation patches only.
- **Disable**: omit validation masks (empty val set -> checkpoint save and `continue`, `train.py:954-960`), or set
  `val_every > num_iterations`. VERIFIED: `val_every = num_iterations + 1` with no validation masks trains and saves normally (scroll_adapt runs).

## 5. Loss and the ignore mask
- Default loss = `LabelSmoothedDCAndBCELoss`, `weight_ce 1.0`, `weight_dice 0.25`, `use_ignore_label=True`
  (`training/loss/losses.py:40-60,116-131`). Train loop appends the ignore channel (`train.py:815-832`).
- `supervision_mask==0` (after max over Z) => **ignored** by both terms: BCE averaged only over valid pixels, Dice
  given `loss_mask` (`vesuvius/models/training/loss/nnunet_losses.py:374-392`). Inside the supervision mask,
  `inklabels>0` is ink and `==0` is **asserted blank** (there is no third "unknown" value: put unknown pixels outside
  the supervision mask).
- BCE label smoothing 0.5 => targets 0.75/0.25 (`nnunet_losses.py:368-372`). Keep 0.5: it preserves the output scale
  ink-sdk assumes (`no_ink_floor = 0.5`, `ink_9um-reader-v3.toml:42`). Dice smoothing 0.
- cucim dilation runs only in `full_3d*` modes with dilation distances > 0 (`train.py:145-162,373-382,798-799`):
  **not reached in flat mode**, no cucim needed.

## 6. Inference and like-for-like comparison
- ink-sdk `ink_9um@reader-v3` runs exactly (`ink_sdk/adapters/ink9um.py:135-153`):
  `python -m koine_machines.inference.infer <zarr> reader-v3.pth ink.tif --overlap 0.5 --blend-mode hann
  --direction both --batch-size 1 --amp-dtype bf16 --no-compile` (profile values `ink_9um-reader-v3.toml:27-37`).
  It passes the *whole* zarr (no `--layer-start/end`); for native input (8.64 and 9.36 µm both within 9.362±0.75,
  `ink_sdk/prepare.py:37-38,170-172`) the zarr is the source itself.
- koine infer then: tile = patch Y (384) (`infer.py:243-257,1638`), stride `round(384*(1-0.5))`=192 (`:337-357`),
  Hann weights floored at 1e-3 (`:311-321`); 17 layers centre-cropped as `Z//2-8 … Z//2+8` (`:360-377,1655-1669`),
  = the training canonical window; `reverse` = same 17 indices reversed (`:1670-1671`); each 17×384×384 tile
  normalised by `normalize_robust(patch)` with default percentiles 1/99 (`:195-221,1087-1090`;
  `vesuvius/image_proc/intensity/normalization.py:169-178`); output uint8 = round(255·p) (`out.tif`,
  `out_reverse.tif`).
- **Fine-tuned run, like-for-like**: same command with the new `.pth`. It is the same code path as long as the
  fine-tune keeps patch_size, model_type/model_config, `robust_mad`, and `ema` off. Caveats:
  (a) `--amp-dtype auto` (the CLI default) reads `mixed_precision` (fp16) from the checkpoint (`infer.py:1267-1328`)
  while ink-sdk forces bf16 — pass the same `--amp-dtype` to both arms; (b) inference prefers `ema_model` if present
  (`infer.py:1110-1124,1338-1342`); (c) state dict loads with `strict=False` and only logs missing/unexpected counts
  (`infer.py:1343-1350`) — require `missing_keys=0 unexpected_keys=0` in the log; (d) inference ignores configured
  percentiles (always 1/99) — do not change them; (e) training normalises only the in-volume part of a crop
  (`ink_dataset.py:1995-1999`) while inference normalises the zero-padded tile: affects edge tiles only, identically
  for both arms.
- Depth order: training always reads Z ascending. If `reverse` is the reading side for a render (as on the PHerc0343 concat render
  and the PHerc1447 HF renders), **store the training zarr Z-flipped** so pseudo-labels and training agree with "forward"; with even Z a
  flip shifts the centred window by one layer, so compare base vs fine-tuned on the *same* zarr (or use odd Z).

## 7. Pitfalls
- **Patch cache**: `out_dir/flat_ink_patches_dm-…_pf-…_ps-17x384x384_labels-auto.json` (or `patch_cache_filename`)
  is reused when label *paths* and finding params match (`common.py:24-76`, `ink_dataset.py:1282-1326`) — label
  *content* is not checked. After regenerating pseudo-labels in place: delete the cache, use a new `out_dir`, or bump
  `_vN`.
- No final save: checkpoints only at `(step+1) % save_every == 0` (`train.py:766-785,1186`) -> make
  `num_iterations` a multiple of `save_every`. Each `ckpt_*.pth` holds model+SGD momentum (~2×138 MB, UNVERIFIED size):
  `out_dir` on scratch. `save_every` defaults to `val_every` (`:322`). Previews PNGs go to `out_dir/{train,val}_previews`.
- `dataloader_workers>0` => `spawn` + persistent workers (`train.py:402-409`); also the thread count for patch
  finding (`ink_dataset.py:1224,1350`). Set to `$SLURM_CPUS_PER_TASK - 1`.
- `mixed_precision` default fp16 (`train.py:342`); bf16 is valid on H100 but changes the checkpoint's `auto` infer dtype.
  `max_amp_overflow_events` only applies when `verify_finite_gradients_steps>0` (`:839-863`).
- wandb: `import wandb` is unconditional (`train.py:7`) but `wandb.init` only if `wandb_project` is in the config
  (`:348-361`). Omit it; also export `WANDB_MODE=disabled`. Other hard imports: `accelerate`, `click`, `cc3d`
  (`ink_dataset.py:7`), `numba`, `aiohttp`+`fsspec` (`common.py:1-5`), `diffusers` (scheduler
  `diffusers_cosine_warmup`, `vesuvius/models/training/lr_schedulers.py:189-197`), `pytorch_optimizer`, `vesuvius`.
  All are in `ink-detection/pyproject.toml:10-37`, i.e. in a `uv sync --frozen` venv of
  `ink-detection/`. VERIFIED: training runs in that venv (scroll_adapt runs, results/).
- `augmentation_preset: default` ignores `augmentation_rotation_axes` (`ink_dataset.py:1268`); keep as reader-v3 had it.
- Do not set `model_config.pretrained_backbone` (changes init/freeze paths, `train.py:488-517,562`).
- Single GPU: `python -m koine_machines.training.train cfg.json` (`configs/README.md:19`). Batch size that fits a
  MIG 2g.20gb at 17×384×384: VERIFIED for batch 4 (2000 iterations in ~600 s, results/adapt-1447-a); otherwise measure with `"benchmark": {"enabled": true, "warmup_steps": 5}`
  (`train.py:245-249,1194-1225`, writes `out_dir/benchmark_summary.json` with peak memory).

## 8. Checks to run before trusting a fine-tune (known-answer controls)
1. Base parity: koine CLI on `reader-v3.pth` with the ink-sdk flags vs the ink-sdk map on the same zarr: identical.
2. Pipeline no-op: fine-tune config with `learning_rate: 0`, `num_iterations = save_every = 2` (SGD weight decay is
   scaled by lr), infer `ckpt_000002.pth` -> map must equal base. Catches config/arch drift and any train-mode state
   (norm running stats). VERIFIED: the no-op checkpoint reproduces base reader-v3 exactly on PHerc1447 w058 (uint8 map
   max |diff| 0, results/adapt-1447-a/info.txt), so the fine-tune path itself adds no drift.
3. Label sanity: in the first `train_previews/` image, ignore mask and target must be non-empty (catches the slice-32 trap).

## 9. Minimal config (single scroll, uniform sampling)
Training-schedule values (lr, iterations, batch) are starting guesses, not taken from code.
```json
{
  "description": "reader-v3 scroll-adapt fine-tune on <SCROLL> pseudo-labels",
  "mode": "flat",
  "model_type": "vesuvius_unet_3d_stem_2d",
  "model_config": {"autoconfigure": true, "basic_encoder_block": "BasicBlockD", "basic_decoder_block": "ConvBlock",
    "stem_channels": 16, "z_projection_mode": "none",
    "features_per_stage": [32, 64, 128, 256, 320, 320], "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "strides": [[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]], "pool_op_kernel_sizes": [[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]]},
  "targets": {"ink": {"activation": "none", "out_channels": 1, "z_projection_mode": "none"}},
  "in_channels": 1,
  "patch_size": [17, 384, 384],
  "patch_overlap": 0.25,
  "patch_min_labeled_coverage": 0.0,
  "flat_z_window_jitter": {"enabled": true, "window_depth": 17, "max_offset": 2, "probability": 1.0, "padding": "forbidden"},
  "image_normalization": {"mode": "robust_mad", "percentile_lower": 1.0, "percentile_upper": 99.0},
  "augmentation_preset": "default",
  "augmentation_rotation_axes": [0],
  "loss": {"bce_label_smoothing": 0.5, "dice_label_smoothing": 0.0},
  "optimizer": "sgd", "learning_rate": 0.001, "weight_decay": 3e-05,
  "scheduler": {"name": "diffusers_cosine_warmup"}, "warmup_steps": 100, "grad_clip": 1.0,
  "mixed_precision": "fp16",
  "batch_size": 4, "num_iterations": 3000, "seed": 42,
  "sampling_strategy": "uniform",
  "best_checkpoint_metric": "val_balanced_accuracy",
  "save_every": 500, "val_every": 500, "val_steps": 16, "val_preview_batches": 1, "log_every": 50,
  "dataloader_workers": 5, "pin_memory": true,
  "ema": {"enabled": false},
  "checkpoint": "/path/to/reader-v3.pth",
  "weights_only": true,
  "out_dir": "<SCRATCH>/adapt/<RUN>/train",
  "datasets": [{
    "segments_path": "<ROOT>/segments",
    "segments": ["<SEG_TRAIN>", "<SEG_HELDOUT>"],
    "surface_volume_paths": {"<SEG_TRAIN>": "<ROOT>/volumes/<SEG_TRAIN>.zarr",
                             "<SEG_HELDOUT>": "<ROOT>/volumes/<SEG_HELDOUT>.zarr"},
    "volume_scale": 0
  }]
}
```
Layout it expects: `<ROOT>/segments/<SEG>/{<SEG>_inklabels.zarr, <SEG>_supervision_mask.zarr[, <SEG>_validation_mask.zarr]}`,
each a zarr v2 group with array `"0"` uint8 `[Z,H,W]` (Z, H, W = the volume's); volume `<ROOT>/volumes/<SEG>.zarr/0`
uint8 `[Z>=21,H,W]`. Writing a label (zarr 2): `g = zarr.open_group(p, mode="w"); g.create_dataset("0",
data=np.broadcast_to(m2d[None], (Z,H,W)).astype("u1"), chunks=(Z,512,512), compressor=Blosc("zstd",3,Blosc.BITSHUFFLE))`.
Run: `WANDB_MODE=disabled python -m koine_machines.training.train cfg.json`; infer with §6's command and
`--amp-dtype bf16` for both base and fine-tuned.
