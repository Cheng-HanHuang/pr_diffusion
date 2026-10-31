# Planner guidance to the theory executor, round 12 (2026-10-31, 14:25 ET)

**Responds to:** round 11 (theory head `69cefc9`): S1 with three dated A15 paragraphs, S3 complete draft 20261031, and the 16-moment witness.

**Governing amendment:** prereg **A16** (planner `21babef`).

## 1. Round 11 accepted and integrated

- **The supplement builds:** 15 pages, no errors, no undefined references, no overfull boxes. PDF in this upload.
- **Integration edits** (listed in `supp/README.md`). Please adopt them upstream, so the next copy needs none:
  - **S1:**
    - title `FLAG_C` → `$\textsc{flag}_C$`;
    - "In the planner's post-hoc" → "In a post-hoc".
  - **S3:**
    - **two literal `\n` sequences** in the A15 addendum (the sentence "…so the coverage tally\nis descriptive…\nThe DAPS exception…") were undefined control sequences; replaced by line breaks;
    - "The planner also reports" → "We also report".
  - **S2 audit:**
    - "planner-confirmed" → "in the pinned source";
    - the internal code path → "the lightweight SITCOM reimplementation in our code base";
    - the paragraph "Remaining source/figure checks for paper integration" is commented out, since it is an internal note.
  - **Double-blind rule for all theory text:** no "planner", "executor", "theory chat", commit-message wording, internal paths or names in rendered text. Comments are fine.
- **The 16-moment witness is accepted** (Gauss–Radau; exact through degree 16; O64 gap ≈ 0.00094; atom 1/81). So is your caveat that it proves non-identification without explaining the 0.09–0.12 v4 miss quantitatively. The main text cites C-T8 only qualitatively.

## 2. Relayed: CP3 (now reproduced exactly) and the planner's exploratory analyses

Everything below is post hoc, on panels already seen.
- **CP3:** all protocol outputs reproduced exactly. At σ = 0.02 the scaled τ rejects every run, so FLAG_C fires on 180/180 images (13 unsolved).
- **(a) Oracle-gap decomposition.** O_orbit − E1 = selection loss (the pick is C or I although an orbit-GOOD run exists) + orientation loss.
  - **FFHQ:** at most 1.1 points for every solver at 8–16 runs, PC-DAPS included.
  - **DAPS-4K ImageNet, K = 8:** 5.0 = 1.0 + 4.0.
  - **FAST ImageNet, K = 16:** 11.3 = 6.6 + 4.8.
  - **FAST ImageNet, K = 64:** 13.0 = 8.0 (all C) + 5.0 (4.0 rotations unfixed, 1.0 GOOD broken).
  - **Partial orbit:** at most 0.6 points anywhere.
- **(b) Budget frontier, XP-D4** (frozen rule, N_MAX ∈ {2, …, 64}, runs drawn from the 64).
  - Gain: +0.66 / +0.80 / +0.52 / +0.12 at N_MAX = 3 / 4 / 6 / 8, then −0.49, −0.89, −1.66, −2.26, −2.96, −3.44 at 12–64.
  - Protocol success rises only from 0.440 to 0.492, against 0.441 to 0.650 for fixed E1 at K = N_MAX.
- **(c) Confidence-conditioned stopping** (stop only if the pick's |CLIP margin| ≥ 0.00763, fixed in advance). It is worse at every N_MAX ≥ 8: −0.18, −1.84, −4.06, −6.45 points at 8–64. Abandoned.
- **(d) Label-free τ (`self_mad3`) at other noise levels.**
  - **Distance from the labeled τ:** within 3.5% at σ = 0.02 and 1.2% at σ = 0.10.
  - **Protocol gain**, with τ cross-fitted from the other cohort's four-run pilot: +4.06 at σ = 0.02, +0.89 at σ = 0.10, +3.67 on DAPS-4K FFHQ, +0.38 on DAPS-4K ImageNet.
- **(e) Gain intervals.** Image bootstrap, stratified by cohort, τ held fixed, 95%:
  - DAPS F8 [+2.1, +4.7];
  - DAPS-4K FFHQ [+2.8, +5.1];
  - SITCOM [−2.2, +1.7];
  - FAST ImageNet 16 [−2.7, +2.8];
  - DAPS-4K ImageNet [−1.6, +1.8];
  - XP-D4 32 [−4.2, +0.7];
  - XP-D4 64 [−6.5, −0.3].

  H1's one-point margin is below the resolution of a single panel.
- **(f) PC-DAPS, image level.**
  - 135 bags have all particles GOOD, 27 have none GOOD in any orientation, and 18 are mixed.
  - E1 keeps a GOOD particle in 17 of the 18 mixed bags (random particle: 7.25).
  - E1 is 84.4%, against 85.0% for the four-particle orbit oracle.

## 3. Requests

**(a) Optional theory forecast `v5` for XP-N, the image-disjoint ImageNet panel (A16.5).**
- **What XP-N is:** the 100 ImageNet validation images `00048900`–`00048999`, with 64 FAST-DIPS runs each and D1's configuration. Its runs 4–63 stay sealed until `v5` is registered or declined.
- **Deliverable:** **code only, frozen before any XP-N run exists**, by **Nov 2, 12:00 ET**. It is the v2 model and fitting code (including the v2 E1 addendum's decision model), generalized to an arbitrary 100-image FAST ImageNet panel with a 4-run prefix:
  - the image ids come from the input; no D1 id or provenance assertions;
  - targets E0, E1 (G1), O_raw and O_orbit at K ∈ {4, 8, 16, 32, 64};
  - the four variants, with the race as primary;
  - the same predictive-interval rule.
- **Freeze:** a launcher and sealer like v2/v4's, plus a synthetic contract test, committed with the SHA-256 of the code.
- **Then:** the planner runs it unchanged on the released runs 0–3 and registers the output before the unseal. Without code by the deadline, decline in one line; XP-N goes ahead without a forecast test.

**(b) Optional pre-score note for XP-N's protocol hypotheses**, like your H2 note. It must be committed **before any XP-N run exists** (by Nov 2, 12:00 ET).
- Predict N-H1a (N_MAX = 16, gain ≥ −0.01) and N-H1b (N_MAX = 64, gain < −0.01) from your S1 large-N mixture analysis, using only the released D1/XP-D4 strings.
- The planner registered a population-transfer forecast (A16.5): +0.3 [−3.6, +4.2] at 16 and −3.4 [−7.8, +0.9] at 64. Yours should come from the mechanism, not from resampling.

**(c) S1 edits** (one or two sentences each; dated, earlier text preserved):
- Use "equal mean runs" for the comparator instead of "equal compute" throughout. The paper does not match NFE or wall-clock.
- In the H-G1 paragraph, name the denominator. +11.0% is relative to the scaled value; the main text's 10% is relative to the labeled one. Both are right.
- Optional remark: why a label-free statistic can track the labeled τ across noise levels when linear scaling does not. The per-image minimum over four runs is dominated by GOOD runs' residuals, so its median tracks the GOOD-residual quantile at each σ. Post hoc; state the assumption.

**(d) Final S1–S3 by Nov 6** (unchanged), with the edits in §1 adopted.

## 4. For your planning

- **Main-text spine:**
  - selection first: GT-free selection is near the oracle on faces (≤ 1.1 points), predictable (57/59 against 17/59; the points are correlated) and transfers to PC-DAPS;
  - coverage is the limit;
  - stopping is conditional: it gains only where failed runs are inconsistent.
- **No protocol v2 before submission.** The paper reports XP-N and XP-C3 if they finish.
