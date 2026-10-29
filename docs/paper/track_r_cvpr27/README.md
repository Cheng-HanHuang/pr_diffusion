# Track R paper draft (CVPR 2027 attempt)

**Status:** skeleton and first prose (planner, 2026-10-26).
- Deadlines: enrollment Nov 10; paper Nov 16 (AoE); supplement Nov 23.
- The planner's go/no-go is Nov 8 (`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`).

## Build

```
latexmk -pdf main.tex
```

- Uses the official CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`) from github.com/cvpr-org/author-kit @ `2917585` (the CVPR 2026 kit). Swap in the 2027 kit if one is released.
- The `review` option gives line numbers and the anonymous header.
- Do not commit build products (`*.aux`, `*.log`, `*.pdf`).

## Draft conventions

- **`\pending{...}`** (red) marks text waiting for data or proofs. At submission, set `\draftmarksfalse` in `preamble.tex`, and `grep -n "\\pending" sec/*.tex` must return nothing.
- **`\claim{ID}`** tags a statement with its entry in `CLAIMS_LEDGER.md`. Only VERIFIED claims, or claims proven or measured by Nov 14, may stay.
- **`\TODO{...}`** marks writing tasks.
- **Double-blind:**
  - cite SITCOM and every other prior work in the third person;
  - no repository links, cluster paths, author names or acknowledgments;
  - the in-house NP family is not described as ours.
- **Numbers** come only from committed aggregates or registered prediction files (`FIGURES_PLAN.md`).

## Section map (8 pages; restructured 2026-10-29, direction (a), prereg A11)

| File | Content | Depends on |
|---|---|---|
| `fig/teaser.tex` | (a) protocol diagram (TikZ); (b) frontier, `fig/protocol_frontier.pdf` from `fig/make_fig_protocol_frontier.py` | Protocol dev outputs; held-out frontier panels optional |
| `sec/1_intro.tex` | Problem; three findings; protocol; contributions | Held-out protocol results |
| `sec/2_related.tex` | Solvers; evaluation practice; best-of-n, verifiers, adaptive sampling; uniqueness and symmetry breaking | Verified citations (C-L1) |
| `sec/3_setup.tex` | Operator; orbit as classical, with the two consequences for selection; labels; policies; data | — |
| `sec/4_protocol.tex` | **The method:** rule; residual as comparison; τ calibration; orientation; why stopping helps; dev results; frozen held-out test; Tab. 1 | Held-out D1-P2, B16, F8 (A11) |
| `sec/5_selection.tex` | Selection model; estimators; two prospective tests; "selection is predictable, generation is not"; identification limit; Fig. kcurves | B16/F8 vs v1/v3; ImageNet panel |
| `sec/6_audit.tex` | What survives without GT: protocols table; FFHQ; ImageNet; unsolved-images figure (placeholder) | A7 per-metric; A8.6; E′; XP-I |
| `sec/7_discussion.tex` | Recommendations; limitations, including symmetry-breaking measurements | All |
| `sec/X_suppl.tex` | Supplement outline. The old Sec. 4 (symmetry theorems) is kept verbatim in `sec/X_symmetry_moved_to_suppl.tex` | — |

**Title** (planner, 2026-10-29; Cheng may change it): "Best-of-K Without Ground Truth: Selection, Stopping and Unsolved Images in Diffusion Phase Retrieval". Earlier candidates are retired.
