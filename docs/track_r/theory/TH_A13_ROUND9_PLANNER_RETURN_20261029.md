# Track R theory executor — A13 / round 9 return (2026-10-29 evening)

**Source and access.** Unpacked theory_upload_20261029_EVE.zip, **checked all 4 listed SHA256SUMS entries first** (including 09_PLANNER_GUIDANCE_20261029_EVE.md, the converter, the 16-run pilot CSV and the paper snapshot). The original 16-run source A9 strings were **not** reinterpreted as held XP-D4 outcomes. No xp/returns, F8/CP1 results, or ImageNet 16–63 run outcomes accessed. This executor has only the explicitly released B16 and D1-P2 outcomes.

## 1. v4 QA review — NO OBJECTION

Committed TH_A13_V4_QA_REVIEW_20261029.md. Verified A8-style converted pilot SHA 12dd28af...; all 100 IDs match the previously registered D1 P1 K4 input, original first-four run classes/ranks are preserved (100/100), each converted 16-run rank matches the stable sorted signed residuals, 1600 class counts/checks agree. Inspected v4_tail_cache.py and the exhaustive original conditional_old_scores: cached probability vectors are computed with the same ordered statuses, score weights and prior factors, the same rng.choice is executed, and exponential spacing consumes the RNG identically. Planner independently reports bitwise identity on 6400 cases including generator states; this source review finds no model change. Accepted the unchanged v4 freeze SHA b0ff25bc... at theory 149331c, 120/24/8 contract. No erratum or refreeze needed.

## 2. SITCOM B16 half-page interpretation — DONE

Committed TH_B16_LOCKED_INTERPRETATION_20261029.md at theory **8e25d8c**. Under the previously registered 7a09153 interpretation rule:
- v1 race primary **18/18 E0/E1 coverage**, versus S2 **2/18**; generation oracles **13/18**; v1 all four policies **31/36**.
- At K16 pooled the observed E0/E1/Oraw/Oorbit are .822/.911/.917/.922. The immutable v1 race points are .853/.908/.929/.942, all covered. The exploratory *realized-class-conditional* race scores run +1.5/+2.2/+2.8 pp high at K8/12/16; S2 is -12.8 pp at K16.
- The independently registered **FAST-informed v3 loses** on SITCOM, 17/36, with K16 Oorbit .893 predicted vs .922 observed (outside its upper band .917). Its overly pessimistic cross-solver difficulty transfer is disclosed rather than retuned. It does not invalidate the selection-rank result in v1.
- SITCOM stopping has **-0.4 pp** equal-budget gain. The CONSISTENT-heavy DEV80 source and the S1 theorem explain *why a gain can disappear*, but do not establish that one explanatory variable is causal.

## 3. OPTIONAL H2 F8 PRE-SCORE — TIMESTAMPED AND COMMITTED BEFORE CP1 RELAY

**Immutable prospectively authored prediction:** theory commit **d42ac3cc0f7c40aabd824eef2a7fa1be654db4ff**, created **2026-10-29T20:12:55Z**, path docs/track_r/theory/TH_H2_F8_PRESCORE_PREFIX_PREDICTION_20261029.md. A13's CP1 score has NOT been received. Do not retrospectively edit this note.

H2 DAPS F8 stop-at-M2 Nmax8 vs the frozen interpolated fixed-K at the same *pooled* mean runs: **positive +2–4 percentage points**, central **+3 pp**, forecast **H2 pass**, low-to-moderate confidence. The historical-source-only calculation uses exactly 100 B22 x 2 and 80 DEV80 x 4 original A8/A8b runs, frozen family class-race rates, historic tau pass flags. The reduced exchangeable-rank limiting model gives +2.65 pp at mean 2.835 runs; the conditional passing race surrogate gives +3.93 pp. These approximations are not 95% forecast bands; GOOD 95th percentile cutoffs and uncalibrated conditional score law limit accuracy. Both methods transfer the class mix but not an identified image-specific future rate.

The same simplified model's SITCOM and ImageNet values are **explicitly marked POST HOC**, since their H1 results (+0.3 and -0.4 respectively) were already relayed: the race approximation overestimates both known gains, so the central +3 pp forecast is deliberately below the DAPS uncorrected race +3.93. Source SHA-256 A8 197d0c8e..., A8b e099138f..., original ImageNet P1 K4 c2177880... . **No DAPS F8 post-prefix candidate was accessed to prepare this prediction.**

## 4. S1 supplement accepted evidence paragraph — DONE

Updated docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex at theory **12ec464**. Added a post-outcome paragraph interpreting Proposition 1 against the held D1-P2 and SITCOM patterns: expected stopping compute/FLAG_C are exact but do not establish a gain; zero consistency flags are not unsolvability certificates; observed H1 +0.3 and B16 -0.4 illustrate the near-fixed-K curve on CONSISTENT-heavy solvers even when the GT-free residual rank selector succeeds. This is a theory-owned supplement draft, **not a change to main-paper Section 4**. The preexisting S2 symmetry/source and S3 selection/bounds drafts remain on the theory branch for Nov 6 integration.

## 5. Next scientific gate

Planner can score CP1 H2 only after citing the H2 pre-score theory commit **d42ac3c** and its UTC timestamp. If CP1 reports less than +2 points or negative gain, report the prediction failure without changing the note, and investigate whether conditional ranking/CLIP or image heterogeneity explains it *as a posthoc diagnosis*. After that, proceed with S1-S3 consolidation by Nov 6. No further refits of registered v1/v2/v3/v4 are allowed.
