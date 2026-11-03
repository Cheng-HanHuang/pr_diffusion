# TH1.3–TH1.4 verification and scope audit (2026-11-03)

**Status:** exact TH1.3 combinatorics PROVED; TH1.4 Gaussian formula PROVED; fresh seeded CPU verification PASS. This is a verification supplement to `TH1_WORKING_PROOFS_20261026.md`, not a new prediction, model fit, or revision to the final S1–S3 supplement.

## Frozen forecast / access gate

Read the **planner-branch** `docs/track_r/theory/PREDICTIONS_REGISTRY.md` before any experimental access. It registers v1 at theory commit `8ef5f94d6de8aab4c3cc25b17fa797db0de1560b` (2026-10-28 06:22:47 UTC, SHA-256 `c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753`); v2 and later freezes are also listed. The copy of the registry on the **theory branch** is historical/stale and still has blank entries; it is not authoritative. **No XP returns, PAC files, GT images, or confirmation holdout were accessed in this run.**

## TH1.3 proof

Let `X_1,...,X_n` be iid per-image runs, `1<=K<=n`, and let `T_K` be a symmetric measurable policy outcome on an unordered subset, with a consistent deterministic total tie order (e.g., score then original run index). Then
`U_K = (1/C(n,K)) sum_{|I|=K} T_K(X_I)`.
Linearity of expectation and iid exchangeability give
`E[U_K]=(1/C(n,K)) sum_I E[T_K(X_I)]=E[T_K(X_1,...,X_K)]`.
This is an unbiased U-statistic for the expected K-run policy outcome; no S1–S3 assumptions are needed. With globally ascending score ranks `r=1,...,n`, the argmin winner is rank `r` iff the subset contains it and its other `K-1` members come from the `n-r` worse-ranked runs, giving weight `C(n-r,K-1)/C(n,K)`. For an argmax policy with globally ascending ranks `j=1,...,n`, the weight is `C(j-1,K-1)/C(n,K)`. These weights sum to one by the hockey-stick identity. For `g` binary successful runs, `U_pass=1-C(n-g,K)/C(n,K)`, taking the binomial coefficient to be zero when `n-g<K`. This is the Chen et al. (2021) pass@k estimator.

**Scope:** the estimator is conditional on the observed n runs and unbiased over their iid generation, not independent across K or across images. It is not an unbiased extrapolator for `K>n`. Run-index tie-breaking must be defined on the full run pool and preserved in subsets; if the policy breaks ties by subset position, it is not the same policy. For a per-metric oracle, rank by that metric only; PSNR, SSIM and LPIPS maxima need not select the same run.

## TH1.4 proof and limitations

For iid bivariate normal per-run scores `(M1,M2)`, with `M2=mu2+sigma2*(rho*Z1+sqrt(1-rho^2)*Z2)`, the index maximizing M1 is measurable with respect to all Z1 values. Hence its independent Z2 term has mean zero and `E[M2_at_argmax_M1]=mu2+rho*sigma2*cK`, `cK=E[max_{i<=K}Z_i]`. Marginally the standardized M2 values are iid normal, so `E[max_i M2_i]=mu2+sigma2*cK`, yielding `Delta_K=(1-rho)*sigma2*cK>=0`. The identity holds for negative rho as well. For a lower-is-better metric (LPIPS), reverse its sign before defining the maxima. Without Gaussian marginals and linear conditional mean, this *quantitative* identity need not hold, although `max_i M2_i >= M2_at_argmax_M1` proves nonnegativity pointwise. If M2 is bounded, `Delta_K<=sup M2-inf M2`; otherwise there is no universal finite bound from rho alone. A Gaussian copula with non-Gaussian metric marginals is not enough for the displayed formula. Across-image heterogeneity requires image-conditional application and averaging, not a pooled correlation substituted into the formula. The per-metric oracle is an optimistic evaluation convention, not one deployable selected image.

## Fresh seeded CPU verification

Executed a self-contained JavaScript CPU harness with fixed 32-bit LCG seed **20261103** (no data files or external services). Across n=2..9, ten deterministic pseudo-random pools per n, scores drawn from four discrete values to force ties, and all K=1..n, compared the closed-form argmin and argmax order-statistic estimators against exhaustive subset enumeration: **880 comparisons, maximum absolute discrepancy 0** in double precision. A separate seeded Box–Muller Gaussian Monte Carlo (120,000 iid panels, K=8, rho=.55, sigma2=2.3) gave oracle inflation **1.4777420466**, against `(1-rho)*sigma2*sample_mean(max Z1)=1.4752509000`, discrepancy **0.0024911466**. The latter is a Monte Carlo sanity check, not a formal test of distributional assumptions or an exact evaluation of c8. (The expected c8 is about 1.424.)

## S1/S2 calibration-only audit

The prefreeze A8-only script `scripts/track_r/theory/audit_a8_rank_race.py` is already committed and enforces authorized K<=4 A8 SHA-256 `197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6`, 760 rows, 1,960 labels, family/cohort counts, and ranked-label consistency. Its pinned calibration checks include GOOD beating CONSISTENT **200 vs 5** in mixed class pairs and GOOD beating ORBIT **50 vs 19**, with **92/100** GOOD residual winners among mixed passing images against S2's **51.9167** expected. These are script-encoded earlier checks, **not freshly rerun against A8 data** in this run. They support a strong class-rank effect and reject treating S2 as an empirical identity; they do not prove S1's strict separation or universal stochastic ranking on new images. The pair counts within images are dependent, so naive independent-pair significance tests are inappropriate.

**Disposition:** Existing TH1.3/TH1.4 proofs and S1–S3 final text need no changes; retain November 6 freeze. TH2 remains a separate November 13 memo target.
