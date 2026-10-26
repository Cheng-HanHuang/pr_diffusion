# Track R theory executor: same-day research return (2026-10-26)

**Status (provisional, not prediction freeze).** Theory work advanced immediately after Cheng's GO. TH1.1 core mathematical results are proved for abstract kernels, including exact Fourier symmetry, posterior orbit ratio, orbit-conditional information independence, output-law equivariance and Gaussian-path TV. TH1.2 curves, Stroebl correspondence, and monotone-gap/crossover lemmas are formally drafted. TH1.3/TH1.4 exact estimators and metric-oracle inflation are drafted. TH2's first classical zero-flip and functional-choice toys are proved and CPU-tested. Native source-level structural audit is now substantially more specific. Calibration data and embargoed outcomes were **not** opened; no v1 predictions are frozen.

**Branch:** `theory/prdiff-track-r-th-20261026` (commits at and before the commit carrying this report).

## New results and pointers

- [TH1 exact symmetry proofs](TH1_1_SYMMETRY_PROOFS.tex) and [TV proof](TH1_1_TV_BOUND_PROOF.tex): original canvas index identity, density-safe posterior ratio, conditioned orientation information equality, Markov invariance, Gaussian KL/Pinsker constant 1/4; explicit deterministic-output caveat.
- [Posterior--sampler symmetry bridge](TH1_POSTERIOR_SOLVER_SYMMETRY_BRIDGE_20261026.md): **PROVED** `TV(Pi,g#Pi)<=2 TV(Pi,L)+TV(L,g#L)`, corresponding necessary standardized trajectory defect if a Gaussian sampler approximates a posterior with nonuniform orbit preferences, and exact KL quotient+conditional-orientation decomposition. These are applications of standard inequalities, not novel general inequalities. Interpretation: a model prior's meaningful orientation preference can *require* solver non-equivariance; do not portray non-equivariance as automatically harmful.
- [TH1.2 selection proofs](TH1_2_SELECTION_PROOFS.tex): five-class closed curves, equality of argmin/accept-first selected-label laws under S1--S2, exact nondecreasing oracle--selector gaps for any K-independent orientation test probabilities, and generation-versus-selection crossover. The inherited imperfect-verifier effect is explicitly credited to Stroebl et al. (ICLR 2026).
- [Estimator/metric proofs](TH1_3_4_ESTIMATORS_AND_INFLATION.tex): exact U-statistic rank weights, pass@K with iid/subset-local/tie requirements, per-metric oracle envelope (standard tools; empirical protocol contribution).
- [Literature/story bridge](TH1_TH2_LITERATURE_BRIDGE_20261026.md): the proposed scientific narrative combines (a) measurement quotient, (b) learned posterior and actual solver occupancy, and (c) GT-free selection bottlenecks. Underlying classical results are credited rather than rediscovered.
- [TH2 toy](TH2_FOURIER_FIBER_FUNCTIONALS_20261026.md): two classical non-reversal real nonnegative zero-flip candidates have exactly equal magnitudes but potentially different frequencies under exact posterior sampling, conditional MAP, and anchor projection. The anchor-projection probability is `lambda Phi(d/(2sigma)) + (1-lambda)(1-Phi(d/(2sigma)))`. **PROVED** for a two-point fiber, **not** a native solver prediction.
- [CPU bridge tests](../../../scripts/track_r/theory/check_th1_theory_bridges.py): executed on CPU on synthetic numbers, passing 10,000 randomized monotone-gap checks, exact finite quotient-KL decomposition, TV inequalities, zero-flip Fourier equality and Gaussian residual-rank Monte Carlo.

## Code-verification finding that merits planner attention

The [pinned native solver audit](TH1_SOLVER_EQUIVARIANCE_CHECKS.md) inspected the exact published codepoints:

- **FAST-DIPS pixel @ a9758b38:** in `main.py` the last iteration returns `x0hat` (uncorrected denoiser), rather than computed `x0y` (corrected estimate). The final output is clamped. [Exact source](https://github.com/ququlza/FAST-DIPS/blob/a9758b38/main.py#L277-L310). This can affect how we describe the method's final likelihood enforcement. **The experiment wrapper remains UNCHECKED.**
- **DAPS @ e7a77d0:** the outer loop visits `num_steps-1` iterations but its final-branch comparison is `if step != num_steps-1`, which is always true inside this loop. As written it returns a next-noise-reperturbed state, provided the scheduled scale is nonzero. [Exact source](https://github.com/zhangbingliang2019/DAPS/blob/e7a77d0/sampler.py#L57-L100). Whether wrappers deviate is UNCHECKED.
- **SITCOM-ODE @ 275ab67:** this implementation uses ODE measurement-gradient correction and SGD/Langevin inner steps; it is not automatically equivalent to the separate local `prdiffusion/algorithms/sitcom.py` Adam/Tweedie implementation. [Exact source](https://github.com/sjames40/SITCOM_ODE/blob/275ab67/sampler.py#L125-L158).

None of these code observations demonstrates a broken experiment or justifies modifying a frozen protocol. They matter to methodological explanation and to whether the TV theorem applies unchanged to *returned outputs*.

## Limits and next dependent steps

- **Oct 28--29:** inspect only allowed A8 K<=4 per-image counts (if committed on the authorized branch) and validate the exact schema and S1/S2 residual-rank diagnostics; never infer class labels from incomplete columns. Else follow the authorized P4 K<=4 fallback.
- **By Oct 31:** execute/refine calibration and commit `TH1_PREDICTIONS_v1.json`; provide file SHA-256, commit SHA, and UTC freeze timestamp to Cheng for planner registration. Do not view XP K>4 returns before registration.
- **Nov 3--6:** finalize note and supplementary integration, including source-checked assumptions for each family and confidence/identifiability sensitivity.
- **Nov 13:** TH2 go/no-go focused on a solver-faithful mechanism, not on claiming that classic zero-flip or sample-vs-MAP is novel.

No new solver/method was proposed. No file outside the theory-owned directories was changed. `PREDICTIONS_REGISTRY.md` remains planner-owned and untouched.
