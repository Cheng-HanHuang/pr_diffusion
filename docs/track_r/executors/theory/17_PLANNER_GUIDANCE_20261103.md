# Planner guidance to the theory executor, round 17 (2026-11-03)

**Responds to:** round 16 (accepted; S1–S3 final for Nov 6).

**New since round 16:** test 1's E1 points are now scored. Your S3 test-1 bullet reports E0 only.

## 1. Test 1 (FAST-DIPS, FFHQ, v1): the registered E1_CLIP points

**Why only now.** v1 registered E1_CLIP forecasts for the 64-run panel. Scoring them needed the frozen N2 face test on all 11,520 runs, which the XP executor published on Nov 3 (XP `953e2f0`; CSV `1be37fa9…` in `theory_inputs/`, which you may read). I reproduced it independently:
- exact curves equal to 6.7e-16;
- all 120 comparisons identical.

**Coverage**, frozen comparison rule, no refit:

| Variant | Primary (n = 64; K = 8–64; B22, DEV80, pooled) | Secondary (n = 16) |
|---|---|---|
| race (primary) | **3/21** (B22 3/7, DEV80 0/7, pooled 0/7) | 4/9 |
| S2 | **21/21** | 9/9 |
| race_tau1 | 10/21 | 7/9 |
| tail | 12/21 | 4/9 |

- **Race direction:** above the observed value at every primary point, by 0.8–8.0 points.
- **Test 1, selected run (E0 + E1):** race **9/42**, S2 **36/42**. Table 2 now shows this.

**Selection-only check** (post hoc, exploratory, frozen parameters; extends the E0 diagnostic your bullet already cites):
- **Inputs held at their realized values:** each image's 64 classes and each run's face-test decision. Only the class of the residual pick is modeled.
- **Pooled E1:**
  - the frozen race predicts it within **+0.6 to +0.8** points at every K (B22 −0.3 to +0.5; DEV80 +1.0 to +1.8);
  - race_tau1 within 0.3;
  - S2 falls **2.5–4.7** points short.
- **Where the miss comes from.** The registered race E1 exceeds the conditional one by 2.9–3.5 points pooled. So the E1 miss is generation, as for E0, and S2's coverage again comes from offsetting errors.
- **Code and output:** `docs/track_r/planner_analysis/A21_nov4_checks/` on the planner branch.

**Correction.** In the E0 diagnostic, S2's shortfall is **4.0–6.8** points by cohort (pooled 4.3–6.2). The main text said "4–8" and now says "4–7". If S3 quotes "4–8" anywhere, please change it.

## 2. Request (with the final S1–S3, Nov 6)

**One dated sentence in S3's test-1 bullet:**
- the E1 coverage (race 3/21, S2 21/21);
- the selection-only result above.

Your wording; keep "post hoc" and "frozen parameters".

**Your four-test tally is unaffected.** It excludes test 1.

## 3. For your information

- **The unsolved FFHQ images are mostly near misses.** Of the 7 FFHQ images that FAST-DIPS never solves in 64 runs, 5 reach 22.8–24.7 dB at best, just under the 25 dB threshold. Your S3 sentence, "six being INCONSISTENT in every one", stays accurate.
- **ImageNet is different.** The 8 unsolved ImageNet images plotted reach only 12.2–19.8 dB.
- **Where this appears:** main-text Fig. 3 (one image per dataset) and supplement S6 (all 15).
- **XP-C3 compute approved (A21).** Main text if it returns by Nov 6, 12:00 ET.
