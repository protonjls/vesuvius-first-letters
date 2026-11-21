# crosscheck: do the ink forms of one mesh reappear on another mesh of the same place?

If letters are real ink on a papyrus surface, they should not depend on the mesh used to find them. A second mesh of the
same sheet should show the same forms at the same places: one made by another unwrapping method (a HerculeX piece vs.
a GrowPatch mesh), one grown from another seed, or a regrow that extends a small piece. If a model is hallucinating
texture, if the mesh has slipped to the next winding, or if the "letters" are a CT feature off the sheet, the forms
will not line up. `reappear.py` measures this directly. It needs no hand alignment and no annotation.

For every pixel of mesh A's window it finds the same surface point on mesh B. That point is the nearest B node,
refined on B's local tangent plane, and it is dropped if it lies more than `--maxd` voxels from B along B's normal.
B's ink map is sampled there, which puts it in A's pixel frame. The output is the Pearson r between A's map and B's
resampled map over pixels with CT on both meshes, for every model and every pair of depth orders (the two meshes'
normals may point opposite ways). `--shift N` repeats the comparison with B's map moved N pixels: use about half a
letter as the negative control.

| file | what it does |
|---|---|
| `reappear.py` | the check (numpy, scipy, tifffile, Pillow); writes `reappear.tsv`, `reappear.json` (normal-distance stats, r table, best order pair per model) and `reappear_sheet.jpg` (A map and resampled B map side by side) |
| `demo_306b.sh` | known-answer demo on the awarded PHerc0306B line: a HerculeX piece against a GrowPatch mesh of the same place |
| `tests/test_reappear.py` | synthetic known answers: the same sheet sampled on 16- and 20-voxel grids gives r > 0.95, while a shift of one blob, a wrong window offset, or a mesh 30 voxels away (another sheet) does not |

## Usage

```
python3 crosscheck/reappear.py <meshA> <mapsA> <meshB> <mapsB> <out_dir> --rows r0:r1 --cols c0:c1 \
    [--offB x0,y0] [--maxd 10] [--models reader-v3,rectina-m4] [--shift 0]
```

- `meshA`, `meshB`: tifxyz directories (`x.tif y.tif z.tif meta.json`). Grid spacing is read from `meta.json`
  (`scale` 0.0625 for HerculeX pieces, so 16 px per node; 0.05 for GrowPatch, so 20 px per node). You can override
  it with `--ppcA`/`--ppcB`.
- `mapsA`, `mapsB`: output directories of `followup/confirm_region.sh` or `herculex/piece_confirm.sh`. Only the d0
  variant is needed: `VSEL=d0`. Each must hold `ct_d0.png` and `maps/d0/<model>_{fwd,rev}.png`, rendered at 1
  voxel per pixel with grid node (r, c) at pixel (r·ppc, c·ppc).
- `--rows/--cols`: A's grid window to compare.
- `--offB`: if B's maps cover a window of B's render, give that window's origin (the `x0 y0` passed to
  `confirm_region.sh`).

## Reading the result

- A high r in one order pair, a low r in the other pairs, and a low r under `--shift` mean the same ink forms sit at
  the same places on both meshes.
- Check `normal_dn_median` in `reappear.json` too: it should be a few voxels. A large value means B follows another
  sheet over the window.
- r is a property of the whole window. Look at `reappear_sheet.jpg` to see which forms match. A bright crack or
  edge present in both meshes also correlates, so compare the CT.
- A low r does not prove there is no ink. B may cover only part of the window (see `valid_pixels`), or the two
  meshes may sample the sheet at different depths.

## Known answer (PHerc0306B)

`demo_306b.sh <out_dir>` runs both meshes through reader-v3 and rectina-m4 at d0, then `reappear.py` on piece rows
2:90 cols 0:143, the part of the piece that overlaps the winners' claim box. The GrowPatch mesh is shipped in
`../results/crosscheck-306b/grow-mesh`: VC3D f0e4410 `vc_grow_seg_from_seed` on the public m7 surface prediction,
seed 3559 3224 9348, 140 generations, step 20. Results from a clean checkout are in `../results/crosscheck-306b/`.

Clean-checkout result (commit b6b56d7, CPU, 16 cores, demo 425 s; `results/crosscheck-306b/summary.txt`):

| | reader-v3 | rectina-m4 |
|---|---|---|
| best order pair (A rev, B fwd), r | 0.685 | 0.736 |
| other order pairs, r | −0.085 to 0.283 | −0.057 to 0.304 |
| same pair, B shifted 160 px (about half a letter), r | 0.122 | 0.112 |

The two meshes lie a median 3.0 voxels apart along B's normal (90th percentile 8.2), and 88% of the window has CT on
both. In `reappear_shift0/reappear_sheet.jpg` the awarded line sits at the same place on the HerculeX piece and on the
resampled GrowPatch mesh, in both models. The pass bar was fixed before the run: r ≥ 0.4 in both models, with the
shifted r below half of it. The synthetic tests passed 4/4. They are sensitive: a 4-pixel misregistration of the same
sheet drops r to 0.82, below their 0.95 bar.

Not measured: GPU inference, Docker, other hosts, and meshes from unwrapping methods other than HerculeX and GrowPatch.
