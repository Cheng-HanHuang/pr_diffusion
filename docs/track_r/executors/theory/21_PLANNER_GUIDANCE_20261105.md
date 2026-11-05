# Theory guidance, round 21 (planner, 2026-11-05)

**Context.** Cheng asked for a writing pass on the main text and a sweep of paper defects: complete references, complete tables and numbers, explicit notation, and no internal names. The main text is done (planner commit of today). This round asks the theory executor to do the same for its supplement drop-ins S1–S3. Nothing in the theory itself changes.

**Due:** Nov 12, 12:00 ET (the supplement is due Nov 23; this leaves time for one integration round).

## 1. Names the paper now uses

Use these everywhere in S1–S3. Internal stage names may stay in file names, commit messages and JSON keys, never in the LaTeX text.

| Internal | Paper |
|---|---|
| B22 | cohort A (100 random FFHQ validation images) |
| DEV80 | cohort B (80 enriched images; 71 from the training split) |
| DAPS (FFHQ historical, 400 × 5) | **DAPS-2K** |
| DAPS 400 × 10 (published PR configuration) | **DAPS-4K** |
| F8 | the eight-run DAPS-2K panel (test 4) |
| B16 | the sixteen-run SITCOM panel (test 3) |
| D1, D1-P2 | the sixteen-run FAST-DIPS ImageNet panel (test 2; held-out protocol panel) |
| XP-D4 | the 64-run FAST-DIPS ImageNet panel (test 5) |
| XP-N, "new images" | **ImageNet′** (100 unseen validation images; test 6) |
| "first image set" | **ImageNet** (the first 100 images of the DAPS test archive) |
| XP-G | the noise-level panels of FAST-DIPS (σ = 0.02 and 0.10) |
| XP-C3 | the prior orientation test (paired denoising-loss score) |
| A8 calibration | the historical runs of the three solvers on the 180 FFHQ images (two or four per image; 1,960 runs) |
| bag | a PC-DAPS **sample** (its four interacting particles for one measurement) |
| E0, E1, O_raw, O_orbit | unchanged (defined in main Sec. 3) |
| v1–v5, H1–H4, H-G1, H-G2 | unchanged (defined in the supplement registry and hypothesis tables) |

Notation:
- The orbit index is now **b**: `R_b`, `b ∈ {0,1}^3`, `R_{111}`. `s` is reserved for the pass probability (main Sec. 4 and 5). Please check S1, S2-proofs and S3 for `R_s`.
- "points" means percentage points. Registered prediction checks are **forecasts**, and a forecast **covers** its outcome. Please avoid "points" for forecast checks.

## 2. Requested edits

1. **Rename** per the table. Lines in the integrated copies (planner copy, `docs/paper/track_r_cvpr27/supp/`):
   - S1: 19, 69, 87, 134, 215, 232, 241–242, 269–270, 329, 342–344, 380–388, 392–415 (the PC-DAPS "bag" paragraph), 427.
   - S3: 33, 90, 102, 105, 197, 230–238, 249, 259, 278, 287, 322–324, 336–337, 357–359, 378, 413–414, 496, 554.
   - S2 proofs audit: 155.
2. **Remove process language.** No imperatives addressed to the planner ("Do not claim ..."), no "authorized", "released", "planner". Turn dated labels such as "Dated XP-C3 interpretation (2026-11-03)" into plain prose or a footnote without the date.
3. **Define the two race sensitivity variants in S3.** Main Sec. 5 now says: "two sensitivity variants of the race, one with a tighter prior on its log-weights (supplement)", and later "those of the tighter-prior variant within 0.3". S3 should state:
   - the primary fit: zero-mean Gaussian penalty on log w_O and log w_C with scale 2.5;
   - `race_tau1`: the same with scale 1.0 (shrinks the weights toward exchangeability);
   - the transferred-tail variant: the pooled C-over-G rate 5/205 applied to every solver.
4. **Partial orbit errors share w_O.** Main Sec. 5 now writes the race ceiling as `p/(p + (o_g + o_p) w_O + c w_C)`, following your fit code ("Partial O shares O's rate by convention"). Please check that S3 says so once, in words.
5. **Short sentences.** Cheng prefers short, simple sentences. One claim per sentence where possible.
6. **Adopt the planner edits upstream** in `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex`:
   - `R_s` → `R_b`;
   - "DAPS" → "DAPS-2K" for the FFHQ orbit-error share;
   - ImageNet/ImageNet′ names;
   - the G1/G2 labels in `tab:s2-gt` replaced by descriptions;
   - the last `\pending` (per-solver source audit) removed: the sentence now ends "the denoiser need not."
7. Keep every `\label` unchanged; the main text and S4 cross-reference them.

## 3. Quick review of three main-text sentences (reply in your notes)

1. Intro: "The noise energy, shared by two runs of the same measurement, cancels in this comparison." (Sec. 4 gives the identity with the remaining cross term.)
2. Intro: "With n runs per image, only the first n moments of the per-image success rates are identified, and these do not fix the share of images whose rate is near zero." (C-T8)
3. Sec. 5: the race ceiling `p/(p+(o_g+o_p)w_O+cw_C)` (item 4 above).

## 4. Unchanged

- No reads of `xp/returns`. You may read `xp/theory_inputs/` and the paper directory.
- Only the planner edits `PREDICTIONS_REGISTRY.md`.
- Double-blind: no repository links, cluster paths or author names; cite SITCOM in the third person.

## Planner's view

- **Current results.** The main text is complete at 8 pages: references verified, every number traced to the ledger, internal names gone. The new writing leads with what survives without ground truth. Unsolved images are framed as the ill-posedness of phase retrieval. SITCOM's consistent-but-wrong runs are a solver limit, outside the scope of selection.
- **Plan.** Theory cleans S1–S3 by Nov 12. The planner integrates on Nov 12–13. The supplement freezes on Nov 22.
- **Image of the publication.** An empirical-and-theory audit. The headline result: ground-truth-free selection nearly matches the orbit oracle on FFHQ. It beats the published per-metric oracle on ImageNet with four DAPS runs. The theory supplement explains the selection (the race), stopping (the Proposition) and the orbit symmetry.
