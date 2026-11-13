#!/usr/bin/env python3
"""koine_machines flat-mode fine-tune config for reader-v3 (contract copied verbatim from the checkpoint's embedded
config, docs/koine_finetune_notes.md s0/s3/s9). Schedule values are arguments.

  make_config.py --out cfg.json --out-dir DIR --root ROOT --seg w062 [--seg ...] [--lr 0.001] [--iters 2000]
                 [--batch 4] [--save-every 500] [--warmup 100] --checkpoint reader-v3.pth [--workers N]
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--root', required=True)
ap.add_argument('--seg', action='append', required=True)
ap.add_argument('--lr', type=float, default=0.001); ap.add_argument('--iters', type=int, default=2000)
ap.add_argument('--batch', type=int, default=4); ap.add_argument('--save-every', type=int, default=500)
ap.add_argument('--warmup', type=int, default=100); ap.add_argument('--workers', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--checkpoint', required=True, help='reader-v3.pth (weights-only init)')
a = ap.parse_args()
assert a.iters % a.save_every == 0, 'koine saves only at (step+1) % save_every == 0 (spec s7)'
cfg = {
 "description": "reader-v3 scroll-adapt fine-tune on consensus pseudo-labels",
 "mode": "flat", "model_type": "vesuvius_unet_3d_stem_2d",
 "model_config": {"autoconfigure": True, "basic_encoder_block": "BasicBlockD", "basic_decoder_block": "ConvBlock",
   "stem_channels": 16, "z_projection_mode": "none",
   "features_per_stage": [32, 64, 128, 256, 320, 320], "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
   "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
   "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
   "pool_op_kernel_sizes": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]},
 "targets": {"ink": {"activation": "none", "out_channels": 1, "z_projection_mode": "none"}},
 "in_channels": 1, "patch_size": [17, 384, 384], "patch_overlap": 0.25, "patch_min_labeled_coverage": 0.0,
 "flat_z_window_jitter": {"enabled": True, "window_depth": 17, "max_offset": 2, "probability": 1.0, "padding": "forbidden"},
 "image_normalization": {"mode": "robust_mad", "percentile_lower": 1.0, "percentile_upper": 99.0},
 "augmentation_preset": "default", "augmentation_rotation_axes": [0],
 "loss": {"bce_label_smoothing": 0.5, "dice_label_smoothing": 0.0},
 "optimizer": "sgd", "learning_rate": a.lr, "weight_decay": 3e-05,
 "scheduler": {"name": "diffusers_cosine_warmup"}, "warmup_steps": a.warmup, "grad_clip": 1.0,
 "mixed_precision": "fp16", "batch_size": a.batch, "num_iterations": a.iters, "seed": a.seed,
 "sampling_strategy": "uniform", "best_checkpoint_metric": "val_balanced_accuracy",
 "save_every": a.save_every, "val_every": a.iters + 1, "val_steps": 1, "val_preview_batches": 1, "log_every": 50,
 "dataloader_workers": a.workers, "pin_memory": True, "ema": {"enabled": False},
 "checkpoint": a.checkpoint, "weights_only": True, "out_dir": a.out_dir,
 "datasets": [{"segments_path": os.path.join(a.root, "segments"), "segments": a.seg,
   "surface_volume_paths": {s: os.path.join(a.root, "volumes", s + ".zarr") for s in a.seg}, "volume_scale": 0}],
}
json.dump(cfg, open(a.out, 'w'), indent=1)
print(a.out)
