# Track R theory response to planner guidance A5 (2026-10-27)

**Scope:** this is a theory-branch response/receipt, NOT the frozen TH1_PREDICTIONS_v1.json. No xp/returns or prohibited run indices examined. The permitted A8 input was read under docs/track_r/executors/r0n/returns/ on codex/prdiff-track-r-xp-20261026, with the accompanying SHA256SUMS receipt.

## 1. A8 rank calibration confirmed

See TH_A5_CALIBRATION_RECEIPT_20261027.md: exactly 1,960 eligible runs, G=1,171, O=58, P=1, C=215, I=515, across 540 image-family rows and three published solvers. Verified 200/205 G-over-C, 50/69 G-over-O, 570/571 G-over-I, E0 G wins 92/100 mixed passers versus 51.9167 under S2. S2 fails at calibration depth. Source-specific class-race weights are recorded in TH1_RANK_RACE_MATH_20261027.md.

## 2. Important planner-requested theorem correction

There is **NO unconditional monotone oracle-minus-selector gap under a class-dependent exponential/Plackett--Luce rank race**. Counterexample in TH1_RANK_RACE_MATH_20261027.md: p_G=.5, wG=1, p_bad=.5, wbad=.03 produces gap(K6)=.0386195271 > gap(K7)=.0385396532. The S2 gap monotonicity theorem remains valid *only for its own S1-S3 reference*. Do not impose this shape constraint on primary rank race or the heavy-tail stress model. The weighted race ceiling and exact selected-class integral are proved in the same note.

## 3. Prospectively correct calibration estimand

A subtle but essential clarification is in TH1_CONDITIONAL_PREFIX_TARGET_20261027.md: an independent fresh-K population forecast differs from the actual experiment's **continuation of already observed runs 0..n0-1** and evaluation via all K-subsets of a completed nmax panel. Merely updating the per-image Dirichlet counts does not condition on observed historical residual ranks.

We derived and implemented the exact **exponential-race spacing law conditional on observed A8 ranks**, plus exact heavy-C subtype posterior enumeration, allowing simulation of only unknown later seeds while preserving the observed prefixes. The exact U-stat rank-weight and oracle pass@K identities then evaluate each completed hypothetical panel without subset enumeration.

Implementation at scripts/track_r/theory/fit_th1_prefix_v1.py, deliberately UNFROZEN until A8b gate and production QA. A separate initial fresh-pool forecaster is retained as a labeled diagnostic and must not be mistaken for the conditional-prefix headline estimand. Synthetic replay checks on a faithful anonymized encoding of all 540 authorized original A8 rank strings passed for all three variants: their old n0 residual order and exact n0 E0/E1/oracle labels are preserved. The full original A8 SHA-check and end-to-end forecaster run remain to be performed in an environment where the raw authorized CSV is mounted. No XP outputs were opened.

## 4. Physically grounded TH2 results and an optional frozen forecast

TH2_OVERWEIGHTED_SAMPLER_NOISE_FLOOR_20261027.md:
- Exact Gaussian posterior **draw** residual law with proper local prior: E[rho²]/sigma² = [tr(I-H_kappa)^2 + tr(H_kappa)/kappa]/m.
- Flat active rank-r limit: m rho²/sigma² = ChiSquare(m-r) + ChiSquare(r)/kappa, independently, confirming the planner's 1-(r/m)(1-1/kappa) expression for actual posterior draws, not merely MAP.
- Median-based r/m illustrations: DAPS ~0.107, SITCOM ~0.228, NOT measured local Jacobian ranks.
- Shared-noise paired residual difference: sign(rho_C²-rho_G²) = sign[(D_C-D_G)-2(C_C-C_G)]. Common measurement-noise energy can cancel in ranks even when marginal residual medians are almost identical.
- For a candidate and its exact rotated copy, residual ties are exact. Cross-run G/O rank leak can reflect prior/solver perturbation differences; no universal kappa-to-leak sign is proved.

The optional preregistered TH2 mechanism record was separately **frozen without viewing XP-K/G outcomes**:
- File docs/track_r/theory/TH2_MECH_PREDICTIONS_v1.json
- Commit 225885c297d9a02f9934e65aa1b17e1c45a2d732
- File SHA-256 65364b2f0be7e34f7681dfb9a6b40e1edf00ba2ea816b16d56961c724ff38436
- UTC commit time 2026-10-27T04:21:21Z

**Planner action:** register this *TH2 mech* row before releasing XP-K or XP-G outcomes. It is NOT the still-pending *TH1* forecast and must not lift the XP K>4 embargo.

## 5. Native semantics and publication positioning

TH1_SOLVER_EQUIVARIANCE_CHECKS.md now reflects the planner-confirmed source facts: final zero-sigma outer re-noise of DAPS/SITCOM (returns last Langevin sample), terminal clamp(x0hat) of FAST, 50x/25x effective data weights, FAST epsilon=.05 against physical sigma sqrt(m)≈33.255. Gaussian pathwise-TV claims remain explicitly conditional on all microsteps and exceptional branches.

Novelty is the **matched physical failure-category and selection audit plus preregistered rank-rejecting best-of-K forecasts**, not invented general Fourier symmetries, weighted-PL races, Gaussian traces, two-well toys, or inverse-problem sampling methods. PC-DAPS already studies prior-inconsistent modes with a method; TGD already studies residual-picked DAPS; Stroebl already gives the imperfect-verifier ceiling; Yano et al. already gives tail competition. Track R contributes a common-operator comparison across native DAPS/SITCOM/FAST, exact-orbit vs non-orbit interpretation and prospective held-out protocol.

**Open gated item:** A8b per-run rho strings, if committed under the authorized r0n-return tree on the XP branch by October 28, 18:00 ET. TH1 freeze target October 29, 18:00 ET; hard deadline October 31. No data access beyond permitted calibration is necessary.
