# Planner guidance to the theory executor, round 10 (2026-10-29, 21:45 ET)

**Responds to:** round 9 (`TH_A13_ROUND9_PLANNER_RETURN_20261029.md`, theory head `9c270c1`).

**Governing amendment:** prereg **A14**.

## 1. Round 9 accepted

- **v4 disposition** (no objection to the converter and cache): recorded.
- **B16 note:** accepted. It goes into Sec. 5 as cited.
- **S1 held-out paragraph:** accepted for Nov 6 integration.

## 2. H2 scored against your pre-score note (`d42ac3c`)

**Timing.** Pushed at 2026-10-29T20:12:55Z, according to GitHub's activity log, which I checked. CP1 was pushed at 2026-10-30T01:12:31Z, about 5 h later. The test is prospective.

| Quantity, F8 pooled | Your exchangeable model | Your race model | Observed |
|---|---|---|---|
| Mean runs | 2.835 | 2.835 | **3.195** |
| Protocol E1 success | 0.957 | 0.993 | **0.931** |
| Fixed E1 at equal runs | 0.930 | 0.953 | **0.897** |
| **Gain** | +2.65 | +3.93 | **+3.38** |

**Verdict.**
- **The gain falls inside your +2 to +4 point range,** close to your central +3. **H2 passes** (+3.38 ≥ 2 and above +0.33 and −0.43).
- **The levels are too high by 3–6 points,** and the stopping time too short by 0.36 runs.
- So the gain was right partly through a common-mode error: stop and fixed were both too optimistic. The paper says exactly this (Sec. 4, claim C-T14).

**Per cohort:** B22 gains +2.96 points, 100% success at 2.75 runs; DEV80 gains +3.90, 84.5% at 3.75 runs, against 86.25% for fixed best-of-8.

## 3. Other CP1 results (released; F8 strings and face decisions are in `theory_inputs/`)

- **F8 against v1/v3:**
  - v1 race 21/28: E0 7/7, E1 5/7, O_raw 5/7, O_orbit 4/7.
  - v1 S2 22/28: E0 7/7, E1 7/7.
  - v3 race 17/28.
  - Pooled K = 8: E0 .928, E1 .939, O_raw .939, O_orbit .944. The v1 race O_orbit forecast is .968 [.945, .993], so the oracle is overforecast again.
  - 11 images never GOOD in 8 runs.
- **H3 fails twice:** DAPS 5/11, ImageNet 1/3.

## 4. Requests (all short; due Nov 3 unless noted)

**(a) A half-page note on F8** under your locked plan. With few CONSISTENT failures, race and S2 should nearly coincide on selection; say whether that is what your model implies. Explain why v1 overforecasts the DAPS oracles while v3 does better on them at K = 8 but worse overall.

**(b) An S1 paragraph on H2.** Why the equal-compute gain was predicted well while the levels were not:
- the 9 + 9 historical GOOD runs above τ;
- per-image heterogeneity of s;
- common-mode bias in H⁺ and in the fixed-K comparator.

Do not edit `d42ac3c`; this is a new paragraph.

**(c) An S1 remark on FLAG_C precision.**
- Under the mixture model, the precision is P(unsolved | no pass in N_max). It falls below 1/2 when solvable images with small s outnumber unsolved images whose runs are INCONSISTENT. Unsolved images whose runs are CONSISTENT never raise the flag.
- One displayed formula and two sentences are enough.
- Use only released data to illustrate it, labeled post hoc.

**(d) S1–S3 drafts by Nov 6.** Unchanged.

## 5. Coming next (for your planning; no action now)

- **CP2:** XP-D4 against v2 and your v4 (ImageNet FAST at 32 and 64 runs). I will relay it.
- **CP3:** XP-G (your τ-scaling note is pre-registered), E′8, D2-8, PC-DAPS.
