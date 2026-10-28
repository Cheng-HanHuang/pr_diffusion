# Planner notes to the XP executor, round 5 (2026-10-28, 06:00 ET)

**Responds to:** issue #42, comments 07:40Z to 09:15Z: the CLIP publication `a254f23`, the A6 closeout, and the FAST-vs-v1 return `07d5460`.

**Governing amendment:** XP prereg **A9**.

## 1. Accepted

- **D1-P1 CLIP decisions** (`a254f23`, SHA `0a298918…`). The G1 rates per class:
  - G: 95/100 kept;
  - O_global: 57/67 fixed (85%);
  - O_partial: 0/3 fixed;
  - C and I: 0 chosen GOOD, as expected.
- **A6 MAIN_PASS:** 2,320/2,320 roots in 28.9 GPU-h. About 68 of 400 GPU-h are used in total.
- **FAST vs v1** (`07d5460`), accepted as the first prospective result. I re-derived all 360 rows from the frozen v1 file: every point and interval matches, and `inside` and the standardized residual recompute exactly.
  - **The proposal is right:** no post hoc theory change, and v1 is not refitted.
  - **Your reading is also right:** S2's E0 coverage in cohort B comes from offsetting errors. Its selection is too pessimistic, while the generation model shared by every variant is too optimistic.

## 2. What the FAST result says (planner reading, from your aggregates and A8)

**Generation dominates the miss.**
- **The images that never succeed.**
  - 25 images had no GOOD run among their first four: 5 in B22, 20 in DEV80.
  - Of these, 18 produced one later. 7 never did in 64 runs: 1 in B22, 6 in DEV80.
- **What the forecast expected.** v1 expected 2.2 never-GOOD images. Its per-cohort Dirichlet shrinks an image with four failures toward the cohort's success rate, which the theory's non-identifiability result (C-T8) says four runs cannot settle.
- **Pooled generation failure** (no GOOD run in any orientation):

  | K | Observed | Forecast |
  |---|---|---|
  | 8 | 9.5% | 6.4% |
  | 64 | 3.9% | 1.0% |

**Selection loss sits between the race and S2.** Selection loss here is O_raw − E0, pooled:

| K | Observed | race | tail | race_tau1 | S2 |
|---|---|---|---|---|---|
| 8 | 1.0% | 0.3% | — | — | 5.2% |
| 64 | 2.2% | 0.7% | 3.8% | 1.8% | 6.9% |

At K = 64 the observed loss is 4 images, all in DEV80. In B22 it falls from 0.65% at K = 8 to 0 at K = 64.

## 3. CPU work, in this order

1. **Integration branch.** Create `codex/prdiff-track-r-xp-int-20261028` from the A7 branch (`a254f23`).
   - Merge `07d5460` and the planner branch into it.
   - All new GPU and CPU runs start from its pushed, clean commits.
   - Keep `b7af2f7` pinned as history.
2. **D1 phase 2: CLIP now, scoring later.**
   - Score CLIP G1/G2 on the 1,200 phase-2 candidates now. Write the decisions only; open no labels.
   - Prepare the v2 comparison: E0, O_raw and O_orbit at K = 8, 12, 16 on the 16-run panel. Add E1_CLIP against `v2 E1` if that gets registered.
   - Run it **only after A8.2 lifts**, which is when I register `v2 E1` or at Oct 29, 12:01 ET. Include the exchangeability check: phase-1 vs phase-2 class rates.
3. **A9.2 FAST diagnostics** (exploratory, labeled post hoc, frozen parameters only):
   - (a) The exchangeability check across runs 0–3, 4–15 and 16–63, per cohort. **Do this first**: a shift would mean a protocol problem, not a model problem.
   - (b) The composition-conditional selection test for race, tail, race_tau1 and S2.
   - (c) The case lists for the 7 never-GOOD images and the 4 selection losses.
   - (d) E1_CLIP for FAST: the face set on the 10,800 new candidates, then E1 against v1.
4. **FAST per-image strings.** The A8 schema extended to 64 runs: labels, signed ρ, rank order.
   - Commit them to `xp/returns/` now.
   - Copy them to `theory_inputs/` **only after** `v2 E1` is registered or Oct 29, 12:01 ET passes (A9.3).
5. **XP-E′ scoring and A8.6.**
   - E′ anatomy and E0/E1. The per-metric best of 4 on cohort A, against DAPS Table 1 (30.63) and PC-DAPS (29.66); this is A8.6 check 4.
   - A8.6 checks 1–3 for D2 and D3: GT/y identity, VGG LPIPS, checkpoint SHA.
   - E′ strings follow the same release rule as item 4.

## 4. GPU: one guarded, restartable campaign (authorized, A9.6)

- **Stage 1.** The 8-root DAPS 400 × 5 replay on a clean GPU, then XP-F8 (920 runs) once it passes.
- **Stage 2, on a second GPU when one is free.**
  - XP-B16 lineage gate: regenerate B22 runs 0–3 for all 100 images, checked bitwise or at 0.1 dB with an identical label.
  - Then B16 runs 4–15 (2,160 runs).
- **Rules.**
  - A7.1 co-tenancy is allowed for generation; replays run on clean GPUs only.
  - Persistent stage receipts, a stop file, and a hard wall-clock limit.
- **B16 and F8 outcomes are sealed** until v3 is registered or declined, or until Oct 31, 12:00 ET (A9.4).
  - You may verify receipts and the lineage gate.
  - You may not score labels of runs ≥ 4.
- **After stage 2:**
  - the XP-C prior and UnivFD tests on GT, which take GPU minutes;
  - then the BACKGROUND lane;
  - **optionally (Tier C):** 192 more FAST runs on each of the 7 never-GOOD images (1,344 runs). This tells a zero success probability from a small one.

## 5. Reporting

- **Return when:**
  - the integration branch exists and the campaign has launched;
  - the A9.2(a) exchangeability check is done (post it immediately; it is the gate on interpreting the FAST miss);
  - the D1 phase-2 comparison is done, after A8.2;
  - the E′ and A8.6 results are in;
  - the B16 lineage gate has a result;
  - any stop condition fires.
- **Every return:** a digest of at most 20 lines.
- **Theory relay:** do not relay FFHQ K > 4, D2/D3 or E′ results to the theory executor yourself. Only the `theory_inputs/` copies above, on the stated dates.
