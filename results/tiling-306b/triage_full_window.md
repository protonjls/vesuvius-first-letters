# PHerc0306B known answer: triage of the full window (199 patches)

From our full run of the window (slice z=9348, x2659–4459, y2324–4124; 827 seeds, 228 grows of 100 generations,
199 kept patches; reader-v3, 24 layers, both orders). The tests below were fixed before they ran.

| check | bar | result |
|---|---|---|
| `rank.py` (20 mm, min-valid 0.6): rank of the best window of a claim-covering patch (231 or 405) among all windows | top 36 | 4th of 896 (patch 231, forward); the window contains the awarded line; best patch-405 window 91st |
| patches with at least one scored window | (reported) | 74 of 199 |
| on-sheet surface inside scored windows / top-36 windows (`review_coverage.py`, 20-vox cubes, 15-vox tolerance) | (reported) | 0.414 / 0.051 |
| blind full review of `sheets.py` sheets (25 sheets, 480 px maps, anonymised codes, `REVIEW.md`, Claude Opus 5.5) | patch 231 or 405 forward graded candidate or strong, and at most 10 other tiles candidate or stronger | patch 231 forward strong, patch 405 forward candidate; 5 other tiles candidate, 8 weak |

The release's `review_coverage.py` and `sheets.py` reproduce these numbers on the run's files (0.4138 / 0.0515, 74 of 199
patches; 25 sheets). The other tiles flagged by the review lie on patches that do not touch the claim surface; we did
not follow them up.
