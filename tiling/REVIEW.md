# Reviewer brief for `sheets.py` contact sheets

This is the brief we give a review agent (or a person) for a full review of a tiling run. Replace SHEETS and OUT.

You screen ink-detection maps for visible text. A tiling window of a carbonised Herculaneum scroll was covered with
~100–300 overlapping surface patches; each patch was rendered from the CT and run through an ink model. Each contact
sheet `SHEETS/sheet_NN.jpg` shows 8 patches; each patch is one tile pair `<code> A` | `<code> B`, the same patch read
from its two sides (one of them is the written side; which one is unknown and may differ between patches). Ink is
BRIGHT. Black = no surface. The white bar at the lower left of every map is 1 cm. Greek letters in these scrolls are
~2–4 mm tall (1/5 to 2/5 of the bar), in horizontal or tilted rows with a row pitch of ~5–8 mm; a line of text is a
row of separate, similarly sized stroke shapes (bars, arcs, angles, loops) with regular gaps.

Do: view every sheet, in order. For any tile with letter-like forms, zoom by cropping the sheet image. Read only the
sheets (the code-to-patch key stays with whoever runs the review).

Known false positives: bright streaks along cracks or seams (long, straight or curved, one stroke); ring outlines around
dark voids; cm-scale amorphous bright patches; uniform blob or speckle fields with no rows; fibre texture (fine parallel
lines over the whole tile); staircase edges along the black no-surface border; the same pattern in both A and B
(structure, not ink on one side).

Grade every tile with any notable content: none / weak (one isolated letter-like form) / candidate (>=3 letter-like
forms in a row) / strong (a legible line or several rows). Write OUT as TSV with header `code side grade where notes`,
one line per tile graded weak or higher, ending with `# sheets viewed: N; tiles graded candidate+: M`.

A grade is a proposal to verify. Follow a candidate up on the full-resolution map and the CT (`../followup/`):
persistence across models and depth, absence in shifted and shuffled controls.
