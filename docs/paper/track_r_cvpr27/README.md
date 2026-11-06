# Track R paper draft (CVPR 2027 attempt)

**Status:** complete draft; writing pass and defect sweep on 2026-11-05; answers to two AI reviews on 2026-11-06 (planner; triage in `docs/track_r/reviews/AI_REVIEWS_20261105_TRIAGE.md`). Main text ends at the bottom of page 8 with about 2 lines of slack: **every new main-text line must be offset by a cut.** Claim tags are zero-width (set above the line), so the draft and the clean copy have the same layout.
- Deadlines: enrollment Nov 10; paper Nov 16 (AoE); supplement Nov 23.
- The planner's go/no-go is Nov 8 (`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`).

## Build

```
latexmk -pdf main.tex
# clean review copy (claim tags hidden), same sources:
pdflatex -jobname=main_clean "\def\cleancopy{}\input{main}" && bibtex main_clean && pdflatex -jobname=main_clean "\def\cleancopy{}\input{main}" && pdflatex -jobname=main_clean "\def\cleancopy{}\input{main}"
```

- Uses the official CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`) from github.com/cvpr-org/author-kit @ `2917585` (the CVPR 2026 kit). Swap in the 2027 kit if one is released.
- The `review` option gives line numbers and the anonymous header.
- Do not commit build products (`*.aux`, `*.log`, `*.pdf`).

## Draft conventions

- **`\pending{...}`** (red) marks text waiting for data or proofs. At submission, set `\draftmarksfalse` in `preamble.tex`, and `grep -n "\\pending" sec/*.tex` must return nothing.
- **`\claim{ID}`** tags a statement with its entry in `CLAIMS_LEDGER.md`. Only VERIFIED claims, or claims proven or measured by Nov 14, may stay.
- **`\TODO{...}`** marks writing tasks.
- **Pending items tracked here** (no `\pending` marker is left in `sec/0`–`sec/7` since 2026-11-03; XP-I2 and XP-C3 are in the text since 2026-11-03, 18:00):
  - Supplement: σ ∈ {0.02, 0.10} detail, the 512 canvas, DPS.
  - Supplement S1–S3 (theory drop-ins) still use internal stage names; theory round 21 cleans them (`docs/track_r/executors/theory/21_PLANNER_GUIDANCE_20261105.md`).
- **Names used in the paper** (2026-11-05; never the internal stage names such as XP-D4, F8, G5, B22, DEV80, D1-P2, "bag"):
  - **DAPS-4K** = DAPS 400 annealing × 10 ODE steps (published PR configuration, 4,000 denoiser calls per run); **DAPS-2K** = 400 × 5 (the historical FFHQ runs).
  - **ImageNet** = the first 100 images of the DAPS test archive; **ImageNet′** = the 100 unseen validation images.
  - **Cohort A** (100 FFHQ validation images, internally B22) and **cohort B** (80 enriched images, internally DEV80).
  - **Panel** = one solver on one image set at a fixed number of runs or budget. **Orbit oracle** = O_orbit.
  - **PC-DAPS sample** = its four interacting particles for one measurement (not "bag").
  - **Forecasts** (not "points") for registered prediction checks; **points** = percentage points.
  - Orbit index **b** in R_b (s is the pass probability).
- **Framing (Cheng, 2026-11-05):** lead with what survives without ground truth; attribute unsolved images to the ill-posedness of phase retrieval and to the solver; SITCOM's consistent-but-wrong runs are outside the scope of selection; short sentences.
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
| `sec/4_protocol.tex` | **The method:** rule; residual as comparison; τ calibration (the DAPS-4K panels use the label-free τ); orientation; why stopping helps; dev results; frozen held-out test; Tab. 1 (ten panels) | C-G*, C-E5, C-N2, C-T14 |
| `sec/5_selection.tex` | Selection model; estimators; six prospective tests (Tab. 2, Fig. 2); "selection is predictable, generation is not"; identification limit | C-T*, C-R*, C-F*, C-N3 |
| `sec/6_audit.tex` | What survives without GT: FFHQ (Tab. 3); ImageNet (E1 beats the per-metric oracle at K = 4); gap decomposition; prior test; PC-DAPS on FFHQ and ImageNet; Fig. 3 (unsolved images). The protocols table is in the supplement | C-A*, C-E1, C-X*, C-N4 |
| `sec/7_discussion.tex` | Recommendations; limitations, including symmetry-breaking measurements | All |
| `sec/X_suppl.tex` | Supplement outline. The old Sec. 4 (symmetry theorems) is kept verbatim in `sec/X_symmetry_moved_to_suppl.tex` | — |

**Title** (planner, 2026-10-29; Cheng may change it): "Best-of-K Without Ground Truth: Selection, Stopping and Unsolved Images in Diffusion Phase Retrieval". Earlier candidates are retired.
