# results/tiling-306b: clean-checkout run of `tiling/demo_306b.sh` (PHerc0306B known answer)

Run on 2026-11-17 from a clean clone of commit 4fc4159 (`commit.txt`), scripts unmodified, on one 16-core CPU node
with no GPU. Tests took about 1 min and the demo 2,218 s (37 min). The pass bar below was written down before the run.

| check | bar | result |
|---|---|---|
| synthetic tests `tiling/tests/test_tiling.py` | 6/6 | 6/6 (`tests.log`) |
| `seeds.py` on slice z=9348, window x2659–4459, y2324–4124 | identical to `seeds_kat_window.tsv` (827 seeds of the full-window run) | identical (`seeds.png` shows them on the m7 slice) |
| coverage of the winners' claim surface by the regrown patch from seed 405 (`coverage.py`, their `w074own_touchup_V5Rc` mesh, rows 82–137, cols 121–341) | fraction within 5 vox ≥ 0.5 | 0.817, median normal distance 0.26 vox (`coverage.tsv`) |
| reader-v3 forward map of that patch, resampled into the winners' frame, against the v0.2 reference reading view (`demo_check.py`) | r ≥ 0.7 and ≥ 0.3 above both ±150 px shifted controls; reverse order r < forward r | forward r 0.861 (shifted 0.382 / 0.315); reverse r 0.173 (`demo_check.json`) |
| `rank.py` output | written | `rank.json`, `rank.txt`, `rank.png` |

Files:
- `w0405_fwd_in_winners.jpg`, `w0405_rev_in_winners.jpg`: reader-v3 forward and reverse maps of patch 405, resampled into
  the winners' mesh frame (`followup/mesh_resample.py`). The awarded first line is visible in the forward map.
- `patches.jsonl`: the grow/mask record of patch 405 (grid, on-sheet fraction, area, timing).
- `resample.txt`: normal distance between the regrown patch and the winners' mesh (median 0.5 vox over the shared area).
- `seeds_kat_window.tsv`: the 827 seeds of the full-window run, which the demo checks itself against.

The demo grows only seed 405 (`ONLY=405`, the default). `ONLY=all` grows the whole window. That took 228 grows and
about 18 h on 16 CPUs in our full run, and its numbers are quoted in `tiling/README.md`.

The triage of the full 199-patch window (rank of the text window, review coverage, blind full-sheet review) is in
`triage_full_window.md`.
