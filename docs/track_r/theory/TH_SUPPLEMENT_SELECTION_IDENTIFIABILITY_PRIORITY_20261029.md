# Proposed supplement priority: selection and identification before symmetry (2026-10-29)

**INTEGRATION PROPOSAL ONLY — NOT an edit to paper Sections 4–5.** Follows the planner's Oct 29 instruction to prioritize the selection mechanism, finite-run identification obstacle, and honest held-out results. Do not rewrite Sec. 4 while Cheng and the planner decide its framing. This note is a modular theory-owned source for Supplement S1/S2 after authorization.

## Recommended ordering and claim strength

**S2-A (lead, ~1.5 pages): exact finite-pool rank selection and its calibrated failure.** State the fixed-image, fixed-historical-prefix estimand and its exact all-subsets weighting *before* introducing any iid fresh-K theoretical simplification. For a completed pool of n residual-ordered candidates, the probability that overall rank r (1-indexed) is selected in a uniformly sampled K-subset is
\[
  \omega_r(n,K)=
  \frac{\binom{n-r}{K-1}}{\binom{n}{K}},
\]
with zero numerator when n-r < K-1. Thus empirical GT-free selection and oracle curves are exact U-statistics, not heuristic best-of-K plots. The original n0=2/4 runs remain present; new seeds are simulated only after conditioning on their known classes and ranks. Specify how raw GOOD, global ORBIT, partial ORBIT, CONSISTENT and INCONSISTENT categories map to oracle and deployed E0/G1 policies.

Calibrated A8 within-image ranking (three native FFHQ families, 1,960 eligible runs) rejects class-exchangeable S2 as a *description of calibration*: GOOD beats C in 200/205 within-image pair comparisons and is selected by E0 in 92/100 mixed-passing cases, versus 51.92 S2 expected. Those comparisons are **dependent pairs** sharing images and runs. A solver-specific class-score race takes \(T_j\sim{\rm Exp}(w_j)\), \(w_G=1\), with winner probability
\[
  W_j(K)=K\theta_j\int_0^\infty w_j e^{-w_jt}
  \left(\sum_\ell\theta_\ell e^{-w_\ell t}\right)^{K-1}dt.
\]
Given observed numbers \(N_j\), its exact conditional winner law is \(N_jw_j/\sum_\ell N_\ell w_\ell\). Note that the score is a **surrogate for residual rank**, not a physical-unit Fourier residual distribution, and that I-last is an approximation with observed exceptions. Show calibrated G-over-O **DAPS 21/24, SITCOM 14/30, FAST 15/15** and how the same measurement-invariant orbit can still show solver-dependent perturbation-dependent score preferences.

**S2-B (~0.5 page): what the rank model can and cannot imply.** Present the model's weighted large-K selection ceiling and its difference from S2; do NOT assert a monotone oracle-selection gap under class racing. The explicit two-class race \(p_G=1/2,w_G=1,w_{\mathrm{bad}}=.03\) has gap \(D_6=.0386195>D_7=.0385397\), although \(w_G>w_{\mathrm{bad}}\). The S1–S3 monotone-gap result belongs to the exchangeable-ranks reference only. State CLIP's additional orientation-success probabilities separately for ImageNet; do not transfer FFHQ's 58/58 calibration success as a universal guarantee.

**S2-C (promote, ~1–1.5 pages): high-K generation nonidentifiability.** The per-image Bernoulli success propensity \(p\) with unknown mixing measure \(\mu\) produces a four-run count distribution determined by polynomials of degree at most four:
\[
 \Pr(J=j)=\binom4j\int p^j(1-p)^{4-j}\mu(dp).
\]
Yet the population oracle target \(O_{64}=\int[1-(1-p)^{64}]\mu(dp)\) is not fixed by these moments. Give the explicit Gauss–Radau pair (classical quadrature, NOT a new theorem):
\[
 \mu_A={\rm Uniform}[0,1],\quad
 \mu_B=\frac19\delta_0+
 \frac{16+\sqrt6}{36}\delta_{(6-\sqrt6)/10}+
 \frac{16-\sqrt6}{36}\delta_{(6+\sqrt6)/10}.
\]
They match all 4-run count distributions exactly, but differ by **0.0957265 in O64**, and their infinite-K ceilings differ by **1/9**. This illustrates why model-conditional bootstrap intervals cannot be advertised as partial-identification bounds.

Then present the A8-only **data-specific population** C-T8 outer enclosures for O_raw(64): B22 **[0.8696,1]**, DEV80 **[0.6113,1]**. Explain that these result from five Bonferroni-adjusted Clopper–Pearson category intervals, feasible probability measures with masses summing to one, and a conservative continuum discretization correction: degree-four Bernstein count integrands have derivative magnitude <=4, while \(1-(1-p)^{64}\) has derivative <=64. On a p-grid of step \(h=10^{-4}\), expand count bands by \(2h\) and oracle extrema by \(32h\). These intervals concern an iid image-population target, **not the frozen fixed-image conditional-prefix U-statistics**. DEV80 is deliberately selected/enriched: a nominal 95% population coverage claim needs an additional iid-within-cohort sampling assumption; otherwise show it as descriptive sensitivity. This LP was explored **after FAST64 release**, despite using only A8 count data, and must not be described as a prospective success.

**S2-D (~1 page): prospective calibration and failure-case anatomy.** Print each immutable registration SHA and explain the pre-results coverage rule (observed realized-panel statistic inside [lo95,hi95]), including its parametric, non-distribution-free uncertainty scope. Keep these separate:
- **FFHQ FAST v1** failed: E0 coverage race 6/21 vs S2 15/21, but oracle generation only 6/21 raw and 4/21 orbit. Seven images never produce GOOD within 64 runs, six entirely INCONSISTENT; conditional-on-realized-class-composition (explicitly *post-hoc*) race predicts pooled K64 E0 .9526 versus .9389 observed, while S2 predicts .8773. S2's apparent unconditional advantage is error cancellation, not vindication of uniform ranks.
- **ImageNet D1 v2 E1** passed coverage for race/race_tau1/tail in 6/6 checks each, while S2 passed 0/6. Observed K16 E1=.650 vs primary race .6027 [.5197,.6825], but \(O_{\mathrm{orbit}}\) **missed**: .750 actual vs .8587 [.7849,.9292] forecast. Thirty images had no raw GOOD in 16, so the raw oracle was .700 vs .6832 forecast; only 5/100 were orbit-only. The E1 success must be reported **together with** the orbit-generation miss and the E1-oracle decomposition, not as proof the full model is correct.
- If later B16/F8 are released, append a **separately registered** v1-vs-v3 comparison; v3 uses the now-released FFHQ FAST64 difficulty covariate and is not an independent replication of FAST.

**S1 (supporting only, ~0.5–1 page): source-faithful solver/physics audit.** Retain operator definition, whether the likelihood uses signed vs absolute magnitude, which terminal iterate is returned (DAPS/SITCOM final zero-sigma Langevin iterate; FAST clamp(x0hat)), and the effective 50x/25x data-gradient weighting and FAST tolerance limitation. The corrected Gaussian posterior sample trace identity can remain as a supporting diagnostic, conditioned on local linear/Gaussian assumptions. Do not present the toy posterior as the exact native algorithm or classify all methods as purely Gaussian chains.

**Symmetry (~one paragraph, remark only, subject to planner approval).** If \(A(gx)=A(x)\), the likelihood is orbit invariant, and for positive prior densities,
\[
  \frac{\pi(gx\mid y)}{\pi(x\mid y)}
  =\frac{\pi(gx)}{\pi(x)}.
\]
This is classical and does not show that the diffusion solver is itself equivariant. State the operator/finite-grid boundary assumptions and put long pathwise-TV/KL inequalities in an auxiliary proof or omit them if they do not materially explain the observed selection discrepancies. **Do not edit the actual Sec. 4 or assert new symmetry novelty without a new confirmed planner instruction.**

## Proposed figures and a falsifiability checklist

A compact three-panel supplement display is sufficient: (A) within-image G/C and G/O residual rank competitions; (B) image-level never-GOOD and consistency failure counts with frozen oracle/selector prediction bands; (C) four-run moment-matching witness and data-specific LP enclosures, separated from fixed-image forecasts. Report E0/E1, oracle-generation and selection-loss curves on consistent denominators and note the baseline v2 E1 abstention versus its separately frozen addendum.

Before integration, verify no claim treats a conditionally evaluated selector as an unconditional prospective forecast, no number is reported with a different target cohort or panel nmax, no Gaussian surrogate identity is asserted for a native diffusion output without its assumptions, and no post-release sensitivity is promoted over the registered primary variant.
