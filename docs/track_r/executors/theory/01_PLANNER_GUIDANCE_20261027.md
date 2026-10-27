# Planner guidance to the theory executor (2026-10-27, 00:00 ET)

**Responds to:** your interim handoff `TH_PLANNER_INTERIM_HANDOFF_20261026.md` @ `12e0ff3`.
**Verdict:** accepted. Corrections 1–9 are accepted and applied to the paper draft (planner branch). Continue TH1 and TH2 now.
**Freeze:** v1 target is **Oct 29, 18:00 ET**. The hard deadline stays **Oct 31, 23:59 ET**.
**Governing amendment:** XP prereg A5 (`docs/planning/track_r/XP_PREREG_v1_1.md`), on this branch.

## 1. A8 exists, on the XP branch

You checked the r0n branch; A8 was committed on the XP executor branch.

| Item | Value |
|---|---|
| Branch | `codex/prdiff-track-r-xp-20261026` (commit `83abf26` or later) |
| File | `docs/track_r/executors/r0n/returns/R0N2_20261026T191819Z/R0N2_PER_IMAGE_COUNTS.csv` |
| SHA-256 | `197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6` (760 rows) |

**You may read:**
- On the XP branch, everything under `docs/track_r/executors/r0n/returns/`:
  - `R0N2_20261026T191819Z/`: A8, the B2 membership report, the receipt;
  - `R0N2_A43_20261026T193644Z/`: the K ≤ 4 ladder and the post-selection anatomy;
  - the coming **A8b** (per-run ρ strings, K ≤ 4), which lands in the same tree.
- On the r0n branch, `docs/track_r/executors/r0n/returns/R0N_20261024T061111Z/N1c_RESIDUALS.csv`: residual quantiles by class, K ≤ 4.

**Still forbidden:** `docs/track_r/executors/xp/returns/`, and every outcome with run index ≥ 4 (≥ 2 for B22 DAPS).

## 2. What to fit

**Fit only these rows:**
- `S1_b22`: DAPS (n = 2) and SITCOM (n = 4);
- `S2_b24_dev80`: DAPS and SITCOM (n = 4);
- `S5_r0_fast`: FAST, both cohorts (n = 4).

**Exclude** NP, NP_LF, NP_S2, and the 8-image S3/S4 rows. NP is out of the paper (Cheng, 2026-10-26 23:01).

**Cohorts.** Keep them separate and pool 100:80, as your fitter now does.
- Cohort A = B22: 100 random images from FFHQ's validation split.
- Cohort B = DEV80: 80 hard-enriched images, 71 of them from FFHQ's training split. Whether the checkpoint saw them is unknown (B2).

**Cross-checks.** Reproduce these planner counts from A8 before fitting.

Run labels over the three families (1,960 runs):

| Family | Runs | G | O | P | C | I |
|---|---|---|---|---|---|---|
| DAPS | 520 | 335 | 23 | 1 | 18 | 143 |
| SITCOM | 720 | 414 | 25 | 0 | 174 | 107 |
| FAST | 720 | 422 | 10 | 0 | 23 | 265 |
| **Total** | **1,960** | **1,171** | **58** | **1** | **215** | **515** |

A4.3 anatomy under E1_CLIP at maximal K:
- 540 image–solver pairs; E0 fails on 84; CLIP fixes 14 and harms 0.
- 70 failures remain:
  - 66 have no orbit-correct run;
  - 3 are selection misses: SITCOM `CCCG` and `GGCC` (C picked over G), and FAST `IIGI` (I picked over G);
  - 1 is the only partial orbit error (DAPS, cohort B). CLIP cannot fix it; the 8-member S test does.

## 3. Main new fact: S2 is rejected at K ≤ 4

Please verify this independently from `e0_rank_labels` (index 0 has the smallest ρ).

**S1 holds:**
- G beats I in 570 of 571 within-image pairs;
- C beats I 92/92;
- O beats I 27/27.

The single violation, FAST `IIGI`, is also one of the three selection misses.

**S2 fails:**

| Pair (winner first) | DAPS | SITCOM | FAST | All |
|---|---|---|---|---|
| G beats C | 21/21 | 153/158 | 26/26 | **200/205** |
| G beats global O | 21/24 | 14/30 | 15/15 | **50/69** |
| O beats C | 2/2 | 12/14 | — | 14/16 |

SITCOM's G-vs-O rate is consistent with 1/2.

**At the selection level:** in the 100 image–solver pairs whose passing runs include G and another class, E0 picks G in **92**, against **51.9** expected under S2.

**Consequences for v1:**
1. **Keep the S1–S3 curves as the exchangeable-ranks reference** (Stroebl-type). Keep the monotone-gap shape test for that reference only. Under S2 the E0/E1 forecasts are far too pessimistic.
2. **Add a rank-calibrated forecast as the primary E0/E1 forecast.** Use a class-dependent residual race (for example Plackett–Luce/Bradley–Terry weights w_G, w_Og, w_Op, w_C, with I last per S1), fitted to within-image ranks and mixed over your per-image class model.
   - Under constant per-class weights, the per-image K → ∞ selection ceiling is p·w_G / (p·w_G + o_g·w_Og + o_p·w_Op + c·w_C), not p/s.
   - Derive and state this properly, including monotonicity of the gap under the race.
3. **Freeze at least one heavy-tail alternative** (Yano-type): within an image, C residuals have a heavier lower tail than G residuals, calibrated to reproduce the observed 5/205 C wins at K ≤ 4. The K-scaling data can then discriminate among S2, the race and the tail.
   - If A8b arrives before your freeze, fit the within-image gaps ρ_C − ρ_G directly.
4. **Use the audit's decomposition.**
   - A generation failure means no run reaches 25 dB in any orbit member, with probability (1 − q)^K.
   - The selection loss is O_orbit − E1.
   - Forecast the crossover K in this decomposition. If you also report the (1 − s)^K version, give it a different name.
5. **Target the same 180 images.** The quantity is the mean over these images of each image's K-run success. Condition each image on its own runs 0–3, as a posterior under the fitted mixture. Intervals are model-conditional and should include the finite-n U-statistic noise at n = 16, 32 and 64.
6. **Optional, recommended:** distribution-free identification bounds for O_raw(K) and O_orbit(K) from the first four moments, via the Markov–Krein principal representations. Your Gauss–Radau mixture is the lower one. Freeze them alongside. The planner re-ran `check_th1_radau_obstruction.py`: PASS, 9.57265 pp at K = 64.

## 4. Timing and deliverable

- **Freeze target:** Oct 29, 18:00 ET. The hard deadline is unchanged (Oct 31, 23:59 ET). Each day earlier gives one more day of analysis before Nov 16. XP-A16 (FAST runs 4–15) will be generated before your freeze and will wait sealed.
- **A8b:** if it lands by Oct 28, 18:00 ET, use it; otherwise freeze on the ranks alone and say so.
- **Deliverable:** commit `docs/track_r/theory/TH1_PREDICTIONS_v1.json` with a short README section. Send Cheng the commit SHA, the file's SHA-256 and the UTC commit time. The planner registers it.

## 5. Native source facts (your item 9): confirmed

| Solver | What it returns | Source |
|---|---|---|
| DAPS @ `e7a77d0` | The final scheduled σ is 0, so the last re-noise adds nothing: it returns x0y, the last Langevin iterate | scheduler |
| SITCOM-ODE @ `275ab67` | `config/sampler/edm_daps.yaml` sets `sigma_final: 0` in both schedulers, and `xt = x0y + randn·σ_T` with σ_T = 0, so it also returns the last Langevin iterate. Its Langevin step is SGD without momentum plus √(2·lr) noise; a NaN returns zeros, which are G-invariant | `sampler.py` |
| FAST-DIPS @ `a9758b38` | `clamp(x0hat)`: the denoiser output of the last step, not the constrained x0y | `main.py` l. 281–310 |

**What produced the historical candidates:**
- **FAST and SITCOM:** the native pinned code above. The executor's 8/8 exact replays (max |ΔPSNR| 1.9e-6 and 2.0e-6 dB) confirm it. The local `prdiffusion/algorithms/sitcom.py` did not produce them.
- **DAPS:** `e7a77d0` with default-off local edits. A replay gate precedes XP-F8 (A5.2).

**Data-term weights (ledger C-S5):**
- DAPS: the Langevin score uses −∇‖A(x)−y‖²/τ_y² with τ_y = 0.01, i.e. 50× the Gaussian weight 1/(2σ²) at σ = 0.05.
- SITCOM-ODE: the Langevin loss is ‖A(x)−y‖²/(2τ_y²), i.e. 25×. It also takes a unit gradient step on the SSE after every Euler step.
- FAST-DIPS: it projects onto ‖v − y‖₂ ≤ ε = 0.05 (σ√m ≈ 33.3), but returns x0hat.

**Proposed Theorem 2 scope:** it covers the returned outputs of DAPS and SITCOM-ODE, whose last step is a Gaussian Langevin transition with state-independent variance 2·lr. For FAST-DIPS it covers only the last state. Confirm or correct this in your solver audit; the per-solver audit is supplement S1.

## 6. TH2: the noise floor (physically grounded, testable now)

**Evidence at K ≤ 4** (N1c, allowed):
- Median signed ρ of GOOD runs:
  - DAPS: 0.04731 and 0.04733 (0.946σ);
  - SITCOM: 0.04418 (0.884σ);
  - FAST: 0.04983 and 0.04985 (0.997σ).
- Fraction of GOOD runs with ρ < 0.05: DAPS 1.0, SITCOM 1.0, FAST 0.88 and 0.79.
- SITCOM's C and even I medians (0.0443 and 0.0458) are below σ.
- ρ(x★) = σ‖n‖/√m = σ(1 ± 0.1%).

**Reading.** DAPS and SITCOM fit the noise; FAST's returned denoised estimate does not. This is your noise-adaptive consistency result appearing in real data, with an identifiable cause: an over-weighted data term and the terminal map.

**Program, in priority order:**
- **(a) Residual of over-weighted samplers.** Build a local-linear theory of the residual of *samples* (not MAPs) from a sampler whose data term is over-weighted by κ (DAPS 50, SITCOM 25), as a function of κ and the effective fitted dimension r.
  - The planner's back-of-envelope is ρ²/σ² ≈ 1 − (r/m)(1 − 1/κ).
  - The medians then give r/m ≈ 0.11 (DAPS) and 0.23 (SITCOM).
  - Check, correct and bound this.
- **(b) Why within-image ranks are nearly class-ordered.** Class medians nearly coincide (SITCOM G 0.04418, C 0.04434), yet within an image G almost always wins. The reason is the paired structure: every candidate is scored against the same y.
  - Formalize the race through D = ‖A(x̂) − A(x★)‖²/σ² and C = ⟨y − A(x★), A(x̂) − A(x★)⟩/σ², where Δ = D − 2C. These are the A5.5 quantities.
  - Derive when S2 holds and when it fails. Your Φ(δ/(2σ)) result is the seed.
- **(c) Mechanism forecasts.** Two interventions:
  - XP-K, a parameter-only calibration of the data term to σ: SITCOM τ_y = σ, DAPS τ_y = √2·σ, FAST ε = σ√m;
  - XP-G: FAST at σ ∈ {0.02, 0.10}, ε fixed.

  Forecast, for example, GOOD-run ρ/σ, the CONSISTENT share and the Δ < 0 rate. To count these as forecasts, commit `TH2_MECH_PREDICTIONS_v1.json` with v1, by Oct 31 at the latest. The planner will not show you XP-K or XP-G outcomes before then (registry row "mech").
- **(d) Orientation leak.** G beats global O in 87.5% of pairs for DAPS and 100% for FAST, but 47% for SITCOM. The candidate mechanism: in prior-dominated outputs (DAPS, FAST) a rotated run sits farther from the data fiber; in data-dominated outputs (SITCOM) every orientation fits equally well. Formalize it with Theorem 2 and κ.

No more classical toy proofs unless they directly serve (a)–(d).

## 7. Literature

**Yano, Ishibashi and Oyamada** (arXiv:2610.19362), verified by the planner from the arXiv HTML v1:
- authors at NEC; LLM best-of-N only (MATH, MiniF2F, GSM8K, CodeContests);
- assumes i.i.d. samples per problem and continuous reward CDFs;
- three regimes: correct-tail-dominant (accuracy → 1), incorrect-tail-dominant (→ 0) and critical;
- Gaussian "reward hacking" when μ_c > μ_w and σ_w > σ_c, with turnover n* = exp((μ_c − μ_w)²/(2(σ_w − σ_c)²));
- with an incomplete selector, convergence is polynomial rather than exponential.

It is now cited in related work and Sec. 5. In our setting the reward is −ρ. A global orbit error is the exact critical case only when the solver is equivariant (SITCOM-like data); otherwise it is not.

**Other items:**
- Gao et al. (ICML 2023) is not checked by the planner; do not rely on it.
- PC-DAPS: XP-I is promoted to CRITICAL-if-ready (data cut Nov 6), and XP-I2 adds ImageNet, so a same-y comparison will probably reach the main paper.

## 8. Paper changes made (planner branch)

1. Cross-product form of the posterior ratio.
2. Theorem 1's hypotheses: invariant start, equivariant kernels, distinct members, disjoint neighborhoods.
3. Theorem 2 for Gaussian transitions with positive, state-independent scales; FAST's scope stated; XP-C2 called a proxy.
4. TDS narrowed to likelihood-only weights.
5. S1–S3 and q > 0 stated.
6. c + o_p.
7. NP-free counts.

Also:
- Sec. 5 is rebuilt around the rejection of S2, with a race model and frozen forecasts (pending your v1).
- Sec. 6 has a noise-floor paragraph.
- Your contribution paragraph is adapted in the introduction.
- Ledger: `docs/paper/track_r_cvpr27/CLAIMS_LEDGER.md` (C-T1 to C-T8, C-A1 to C-A7).

## 9. Deliverables and dates

| By | Deliverable |
|---|---|
| Oct 29, 18:00 ET (target); Oct 31 (hard) | `TH1_PREDICTIONS_v1.json`: the S2 reference, the race forecast, the tail alternative, the anatomy decomposition and optional moment bounds. Optionally `TH2_MECH_PREDICTIONS_v1.json` |
| Nov 3 | The per-solver equivariance and terminal-step audit (supplement S1), and the noise-floor note (a)–(b) in paper-ready form |
| Nov 6 | Integration text for Secs. 4–5 (conditional on the data) and the supplement proofs |

**Stop condition:** if anything needs `xp/returns` or K > 4 data, stop and ask.
