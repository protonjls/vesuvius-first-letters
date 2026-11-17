# tiling: cover every sheet of a scroll region with GrowPatch patches, then rank and sheet them for review (CPU)

Spiral-fit sweeps give one mesh per winding. Those meshes are a few voxels off the sheet in many places, and in crushed
or folded regions they leave the sheet entirely. The PHerc0306B and PHerc0211 First Letters winners both read their text
on locally grown or hand-refined surfaces. This kit grows such surfaces automatically for **every** sheet that crosses a
window of one axial slice, with no umbilicus and no hand picks. It then renders them, runs a public ink model, ranks
4 cm² windows and lays out every patch on contact sheets for a human or agent reviewer.

| step | script | what it does |
|---|---|---|
| 1 | `seeds.py` | Reads one slice window of a public surface prediction (e.g. m7 L0 th0.2) over plain HTTP. In each 200×200-voxel tile it finds the dominant sheet normal (structure tensor) and puts one seed on every predicted sheet that a probe line along that normal crosses. Works where sheets run in any direction. |
| 2 | `tile_grow.sh` | Takes the seeds in order and skips any seed already covered by an earlier patch. For each remaining seed it runs VC3D `vc_grow_seg_from_seed` (step 20, N generations, public normal grids), renders the surface prediction along the new mesh (7 layers) and calls `patch.py mask`. That step keeps only the connected on-sheet part that holds the seed and marks every seed within 5 voxels of it as covered. It is resumable, and `PART=k/K` splits the window into tile-column groups for parallel jobs. |
| 3 | `ink_patches.sh` | Runs a CT render (24 layers) and ink-sdk models (default reader-v3) in both depth orders for every kept patch. |
| 4 | `rank.py` | Ranks 20 mm windows by text-likeness: letter-scale ink excess, plus line and letter spacing in the row and column profiles. It writes a JSON list and a contact sheet of the top windows. It is a review aid, not a detector. |
| 5 | `sheets.py` | Contact sheets of **every** patch (forward and reverse map side by side, 1 cm bar), under anonymised codes in a seeded random order, with the code-to-patch key written separately. This is for a full review that does not depend on `rank.py` windows. `REVIEW.md` is the reviewer brief we use. |
| – | `review_coverage.py` | Reports how much of the on-sheet surface, counted once in 3D, lies inside scored `rank.py` windows and inside the top N. |
| – | `coverage.py` | Reports the fraction of a reference mesh (e.g. a published claim mesh) that lies within `tol` voxels of a patch, measured along the patch normal. |
| – | `demo_check.py` | Compares a map resampled into a reference frame (`../followup/mesh_resample.py`) with a reference reading view, with shifted controls. |
| – | `demo_306b.sh` | Known-answer demo (below). |
| – | `tests/test_tiling.py` | Synthetic known-answer tests, CPU, about 1 min: seeds on parallel and ring sheets, the mask CLI, seed order and parts, rank order (text rows > blob field > streaks > rotated rows), coverage, sheet layout and key, and review coverage. |

Environment: as `../followup/` (VC3D f0e4410 command-line tools, ink-sdk with koine, python with numpy, scipy,
tifffile, zarr 2, opencv and numcodecs); see `../docs/environment.md`. VC3D streams the CT, the predictions and the
normal grids from the open-data bucket.

## Known answer 1: PHerc0306B (awarded October 2026), full window
We fixed the setup before running: slice z=9348, window x2659–4459, y2324–4124. That is the winners' published seed
(3559, 3224) ± 900 voxels, and we chose it knowing where the text is. The test is whether automatic seeding, masking and
ranking find the text among all the competing sheets of the window. It is not a test of finding the spot. We ran
GrowPatch at 100 generations, step 20.

- **Seeds and grows.** 827 seeds. 228 grows covered all of them. 199 patches were kept, 11 were below 0.3 cm² and 19
  had no on-sheet component at the seed. Growing took 66 ks of CPU on 16-core nodes (about 6 h wall in 4 parallel
  parts). The kept patches cover 1,269 cm² of on-sheet area, counting overlaps.
- **Coverage.** Two patches, from automatic seeds 405 and 231, each hold 82% of the nodes of the winners' claim surface
  within 5 voxels: rows 82–137, columns 121–341 of their `w074own_touchup_V5Rc` mesh. The median normal distance is
  0.26 and 0.28 voxels. No other patch exceeds 3%.
- **Legibility.** On patch 405, public reader-v3 in forward order shows the awarded first line, with the same letter
  forms at the same positions as the v0.2 followup reference: Pearson r 0.861 between reading views. With the
  reference shifted ±15 cells, r is 0.36 and 0.29. Reverse order gives 0.16, with blobs only.
- **Triage by rank.py.** With 20 mm windows that are at least 60% on-sheet, `rank.py` scored 896 windows on 74 of the
  199 patches. A forward-order window of patch 231 ranked 4th, and that window contains the awarded line as visible
  letter forms. The best window of patch 405 ranked 91st. So the text reached the top-36 review sheet, but the ranking
  has a blind spot. Counted once in 3D (`review_coverage.py`, 20-voxel cubes, 15-voxel tolerance), the scored windows
  reach only 41% of the window's on-sheet surface and the top 36 reach 5%. Text on the other 59% would never be ranked.
- **Full review with sheets.py.** We fixed the test before making the sheets. A fresh review agent (Claude Opus 5.5)
  graded every tile of 25 sheets (199 patches, both orders, 480 px per map, anonymised codes, key withheld), knowing
  only that the window might or might not hold text. It graded patch 231 forward **strong** ("a horizontal line of
  about 7–9 separate stroke shapes") and patch 405 forward **candidate**, the two patches that cover the claim. It
  graded 5 other tiles candidate (the bar was at most 10) and 8 weak, all on patches that do not touch the claim
  surface. Every patch appears on a sheet in both orders, so the full review sees the whole tiled surface,
  at 4× reduced resolution. We now run both reviews on every window.

`demo_306b.sh` reproduces the covering patch from the released scripts. It computes the 827 seeds (and checks them
against `results/tiling-306b/seeds_kat_window.tsv`), grows seed 405, runs reader-v3, measures coverage against the
winners' mesh and compares the map with the reference. `ONLY=all` grows the whole window. Results of a clean-checkout
run are in [`results/tiling-306b/`](../results/tiling-306b/).

## Known answer 2: PHerc0211 (awarded November 2026), a crushed region: negative for legibility
The PHerc0211 award text lies on a crushed winding with two hairpin folds (Z-folds). Spiral-fit windings run 11–16
voxels off it, and a single GrowPatch follows one arm and stops at each fold. The same tools at 60 generations on slice
z=8520 (568 seeds, 164 grows, 156 kept patches, ~10.6 ks CPU) did reach all three arms: 75%, 92% and 97% of the
winner's mesh lies within 5 voxels of a patch. But the patch surfaces lie a median 1.3 voxels off his hand-refined
surface. On them, reader-v3 shows only about 3 of his letters as letter shapes; the rest are blob rows. Even his own
soft-loss model gets r 0.72 against his published map, compared with 0.98 on his surface. Snapping the patches to the
centre of the m7 prediction did not move them toward his surface, because his surface itself lies 1.0–1.3 voxels off
that centre. `rank.py` put the text windows at ranks 3, 24 and 34 of 80. But with 20 mm windows it scored only 40 of
the 156 small patches, and 12 mm windows did worse. So in crushed regions, sub-voxel surface accuracy limits legibility,
and the m7 prediction alone does not supply it.

## Limits
- Each patch is a local surface. Patches from different seeds on one sheet overlap and can have opposite normals, so
  the same ink can appear in one patch's forward map and in another's reverse map. Review both orders.
- `rank.py` skips patches that have no 20 mm window at least 60% on-sheet, even at 100 generations (on PHerc0306B,
  125 of 199 patches and 59% of the on-sheet surface). Review the `sheets.py` sheets as well, not only the top windows.
- A review agent's grades are proposals to verify, not results. On PHerc0306B it also flagged 5 tiles away from the
  claim; we have not checked whether they are text or false positives, and we report no reading from them.
- Cost: on PHerc0306B, about 230 grows of 100 generations per 1800×1800-voxel window (~18 h on 16 CPUs), plus
  reader-v3 at 3–50 min per patch on CPU, depending on the node, or a few minutes on a GPU slice.
