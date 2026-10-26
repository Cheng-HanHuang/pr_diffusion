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

## Section map (8 pages)

| File | Content | Depends on |
|---|---|---|
| `sec/1_intro.tex` | Problem, symmetry view, contributions | Final numbers |
| `sec/2_related.tex` | Solvers, evaluation practice, classical PR, best-of-n theory | Verified citations |
| `sec/3_setup.tex` | Operator, orbit lemma, identifiability, labels, policies, data | R0-N.2 B2 (training membership); A4.4 (DAPS config) |
| `sec/4_symmetry.tex` | Prop. 1, Cor. 1, Thms. 1–2, measurements, orientation tests | TH1.1 proofs; XP-C/C2; XP-D |
| `sec/5_selection.tex` | Selection model, monotone gap, estimators, frozen predictions | TH1.2–TH1.5; XP-A/B/F |
| `sec/6_audit.tex` | Protocol table; decomposition; anatomy; ImageNet; controls | R0-N.2 A4.3/A7; XP-D; XP-T |
| `sec/7_discussion.tex` | Recommendations; limitations | All |

**Title candidates** (Cheng is flexible):
- (D, current) "Orientation-Blind Guidance and Ground-Truth-Free Selection in Diffusion Phase Retrieval"
- (C) "What Does Best-of-K Measure? Symmetry, Selection Ceilings and an Audit of Diffusion Phase Retrieval"
- (A) "Best-of-K Without Ground Truth: An Audit and a Selection Theory for Diffusion Phase Retrieval"
