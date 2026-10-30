# Track R S2/S3 integration handoff — October 30, 2026

**Status:** Theory-owned, LaTeX-syntax-checked supplementary drafts; **not** edits to the planner-owned paper or any registered prediction. No `xp/returns` or unrelayed XP-D4 outcome was opened. Sources: repository theory branch at `06fda4e`, the SHA-verified A11 paper excerpt from `theory_upload_20261029.zip`, and the already relayed FAST64, D1-P2, SITCOM B16, DAPS F8 comparisons. Do not substitute this source snapshot for a later approved planner draft.

## Deliverables and insertion plan

1. `TH_SUPPL_S2_RELOCATED_SOURCE_VERBATIM_20261030.tex`: **exact copy** of `paper/X_symmetry_moved_to_suppl.tex` from the A11 planner upload, SHA-256 `fc34f945d27c31c550a66fa4dee239cc64feefd34555d4ae4c3d6ba05bac32b3`. No theorem hypothesis, label, textual claim, `\claim`, or `\pending` was changed.
2. `TH_SUPPL_S2_PROOFS_AUDIT_20261030.tex`: can be inserted **after** the authoritative source. Includes proofs of orbit invariance, posterior ratio and conditional orientation blindness, equivariant-kernel propagation, Gaussian path KL/Pinsker/TV, posterior–sampler triangle, a non-equivariant terminal-map counterexample, source-specific DAPS/SITCOM/FAST return semantics, and why shift-invariant blur and translation cannot break the twin.
3. `TH_SUPPL_S2_INTEGRATED_DRAFT_20261030.tex`: small wrapper that inputs those two files in order. It is **not** appropriate to input the authoritative source twice if the paper already includes it under another path.
4. `TH_SUPPL_S3_COMPLETE_DRAFT_20261030.tex`: a single theory-owned complete selection/bounds draft, superseding the earlier `TH_SUPPL_S3_SELECTION_BOUNDS_DROPIN_20261029.tex` *for planner review only*. Contains the class-race integral and conditional class winner proof, explicit E0/E1 repair law, calibrated rank evidence, nonmonotone race gap witness, exact full-panel U-statistics and fixed-prefix posterior oracle, the 4-moment Gauss–Radau witness (including shallower two-run case), conservative continuum LP outer bounds, and the now-released F8/B16/FAST/ImageNet test results with correct model/interval qualifications.

## QA

- Both S2 integrated wrapper and S3 compile through `latex -halt-on-error` under an isolated article harness with the expected paper macros and theorem environments stubbed. No TeX syntax errors; external main-paper references and bibliography keys remain unresolved under the isolated harness by design.
- The authoritative S2 source is byte-for-byte unchanged (SHA stated above).
- Independent numerical tests reproduce moments 0 through 4 of the Gauss–Radau pair to floating-point precision, its K64 oracle gap `0.0957264957`, and the exact finite-panel subset weights summing to 1 for several n/K grids.

## What still needs planner/editor input, NOT new experiments for the proofs

- The **verbatim historical S2 source** contains unresolved empirical `\pending` markers (XP-C2 denoiser-defect profiles, ImageNet orbit-error counts, FFHQ prompt quotation, generic real/fake detector, image-level orientation tests). Do **not** treat those sentences as confirmed. When those results are available through permitted channels, the planner can resolve them; otherwise delete or qualify the dependent empirical sentences. The accompanying proofs and audit do not require these values.
- Align supplement numbering, cross-reference labels, macros (`\claim`, `\pending`, `\good`, `\Eo`, `\Eone`, `\cA`, `\cG`, `\TV`), and bibliography keys with the current planner-owned TeX preamble. If S2 uses the original main-text labels, remove the superseded definitions from any other paper file before integrating.
- S3 numbers are only from already released registered comparisons. ImageNet E1 6/6 counts 3 K times **two uncertainty constructions on the same observations**, not 6 independent held tests. FAST's realized-class selection diagnostic is **post-outcome**. Four-run moment LP bounds concern an iid population oracle, **not** the fixed-image realized-panel conditional estimand. DEV80 was deliberately enriched; nominal 95% coverage requires an extra sampling assumption.
- The theory results are usable without any new held data. Future XP-D4 or other released experiment outcomes, once authorized, belong in a separate dated outcome addendum rather than an undated rewrite of the mathematical statements.

**Planner action:** integrate these theory-owned drafts when ready, preserve the authoritative symmetry theorem source and immutable registered forecasts, and resolve the open empirical markers rather than inventing them. No Sec. 4/Sec. 5 paper edits have been made here.
