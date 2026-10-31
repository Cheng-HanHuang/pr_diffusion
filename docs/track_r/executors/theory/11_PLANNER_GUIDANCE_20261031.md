# Planner guidance to the theory executor, round 11 (2026-10-31, 01:20 ET)

**Responds to:** S1–S3 (theory head `061c618`) and the round-10 return (`06fda4e`).

**Governing amendment:** prereg **A15**.

## 1. S1–S3 accepted; first integration build done

- **The supplement builds:** 10 pages, no errors, no undefined references (`docs/paper/track_r_cvpr27/supp/`, PDF in this upload).
- **My integration edits, layout only** (listed in `supp/README.md`):
  - the S1 paragraph title `FLAG_C` set in math (it broke compilation);
  - main-paper references prefixed `M-` through `xr`;
  - the orbit lemma restated in the S2 wrapper, since the main text no longer contains it.
- **Your verbatim S2 source** is otherwise unchanged.
- **The stale empirical sentences in S2 that you flagged** stay marked until the data arrive (XP-C2 is in G7). I will resolve them; please do not invent values.

## 2. Relayed: CP2 (XP-D4 against v2 and your v4), released under A11.4

Strings and G1 decisions are at XP `ef1cf10`.

**Your v4:**

| | Race | S2 |
|---|---|---|
| Selection points (E0 + E1, both panels) | **18/18** | 1/18 |
| O_raw | 2/6 | |
| O_orbit | **0/6** | |

- O_orbit forecast 0.840 (K = 32) and 0.899 (K = 64), against observed 0.75 and 0.78.
- 22 images never orbit-GOOD in 64 runs (25 in 16).
- **Sixteen pilot runs did not fix the coverage forecast.**

**The protocol on XP-D4 (frozen A12.4 hypotheses):**
- **H1 FAILS:** −1.8 points at N_max = 32 and −3.4 at 64.
- **H3:** fails at 32; not evaluable at 64.
- **H4 passes.**
- **Decomposition** (mine; the planner's numbers): 4–6 images with fewer than 5 passing runs take about 43% of all runs; the other images stop after 2.6–3.0 runs, while the equal-compute comparator gives every image 4.3–5.0. This is your S1 counterexample, in the data.

**CP3 (aggregates):**
- **XP-G, H-G1 fails:** labeled τ 0.02223 against your pre-score 0.02003 at σ = 0.02; 0.09809 against 0.10017 at σ = 0.10. Your pre-score note said "unresolved, substantive risk", which is fair.
- **DAPS-4K FFHQ:** +3.93 at equal compute.
- **DAPS-4K ImageNet:** +0.10.
- **PC-DAPS (descriptive; the matched-NFE comparison is withdrawn, A15.3).** One bag of 4 particles, scored by its native mean over particles: 79.0% GOOD25 / 28.07 dB. E0 over the same particles: 82.8% / 28.56. E1: 84.4% / 29.10. One DAPS-4K run: 68.3% / 25.22.

## 3. Requests (due Nov 4 unless noted)

**(a) S3 dated addendum: "Outcomes registered after Oct 29".**
- Cover v4 against XP-D4 and F8 against v1/v3. **Do not edit the earlier statements.**
- Make the tally explicit: selection points race 57/59 against S2 17/59 over tests 2–5; oracles overforecast in four of five tests.
- Say what C-T8 implies for 16 runs (16 moments) and why that still did not identify coverage. My reading: the near-zero mass is not a low-order moment phenomenon; check it.

**(b) S1 paragraph on H1's failure at large N_max.**
- Under your marked-candidate law, with a fraction π₀ of images at s ≈ 0 (or s ≪ 1/N_max), give the equal-compute gain as a function of N_max. Show that it becomes negative once N_max·π₀ outweighs the improvement H₂⁺ − H_fixed on the remaining images.
- Use the released XP-D4 strings to illustrate it, **labeled post hoc**.
- Main-text space allows one sentence; the rest goes to S1.

**(c) S1 remark on calibration across noise levels (H-G1).** Why τ_S need not scale linearly in σ for FAST-DIPS: the fixed ε = 0.05 projection, and the denoiser's σ-dependent fit. Post hoc; one paragraph.

**(d) S1 remark on PC-DAPS.**
- Its particles interact, so they are not exchangeable runs. Residual and orientation selection among them is still ground-truth-free and well defined.
- Stopping across bags (a bag treated as one run) would satisfy the protocol's assumptions; no data exist for it.
- One paragraph; no new theory needed.

**(e) Final S1–S3 by Nov 6.**

## 4. For your planning

- **The main text's spine is now "Which run to keep is predictable; whether any run is good is not."** Sec. 5 has a five-test table (Tab. 2 in the draft).
- **No protocol v2 before submission:** any N_max cap would be post hoc.
