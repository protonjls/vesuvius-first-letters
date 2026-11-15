# followup-306b: reproduction of `followup/demo_306b.sh`

Run on a clean clone of the commit in `commit.txt`, scripts unmodified, on one 16-core CPU node (no GPU), 2026-11-15.
Total 11,241 s.

| file | content |
|---|---|
| `summary.txt` | demo log: data probes, grow (1,057 s, 25.7 cm²), render (24 x 5640 x 5640), reader-v3 (2,509 s), resample, controls |
| `grow_params.json` | VC3D `vc_grow_seg_from_seed` parameters (seed 3559, 3224, 9348; 140 generations; step 20) |
| `reader-v3_fwd_in_winners_frame_reading.jpg` | reader-v3, forward order, resampled into the PHerc0306B winners' claim-mesh frame, flipped to their reading orientation, ink dark |
| `resample.txt` | normal distance from the winners' claim surface to the grown surface over the window: median 0.5 voxels, p90 16.1 |
| `anchor.txt` | grid node of the grown mesh used as the control window anchor |
| `controls_stats.json`, `controls_manifest.json`, `controls_montage_fwd.jpg` | `confirm_region.sh` output for reader-v3 and rectina-m4 on a 2400-pixel window: variants d−2..d+2, ±15 voxels off the sheet, depth shuffle |

Pass bar, fixed before the run: the demo exits 0; reader-v3 forward shows at least 5 letters of the awarded line at
the positions of the earlier run; normal distance median ≤ 15 voxels; for both models r ≥ 0.8 at d±1 and d±2 and
r ≤ 0.3 off the sheet and under shuffling. All parts passed.
