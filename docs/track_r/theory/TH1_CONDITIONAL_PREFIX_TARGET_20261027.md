# Track R TH1 prediction target: fixed observed run prefix, new seeds, and rank-conditioned exponential scores

**2026-10-27. Status: PROVED statistical identities, not yet full native K>4 forecast.** This distinction is essential for a valid October 29 prediction freeze. No XP returns/new indices have been accessed.

## 1. Two different estimands must not be silently conflated

A calibration file observes the *actual* labels and residual order of runs 0,...,n0-1 on each fixed image (n0=2 for DAPS B22, otherwise 4). The K-scaling continuation adds actual new run indices n0,...,nmax-1. The U-statistic evaluator then computes the average policy success over all K-subsets of the COMPLETE nmax-run pool.

A naive image-conditioned Dirichlet forecast that integrates an independent brand-new pool of K iid labels **does not condition on the known original run labels or their known rank order**. Merely using those labels to form a posterior for per-image class proportions is *insufficient* to predict the actual continuation protocol. At K=n0, the observed original pool's oracle status is fixed, whereas a fresh-K posterior predictive oracle value is fractional; this is a useful sanity-check counterexample.

Therefore the **primary v1 target must be conditional continuation**, not a new independent K-pool. Forecast both the exact rank-based and oracle U-statistics after generating only the unknown later runs, preserving the known initial class labels and their rank order. A fresh-pool posterior predictive model can be reported separately as a theoretical population reference.

## 2. Exponential Plackett--Luce scores conditional on their observed order

Conditional on an image's original labels L=(L0,...,L_{n0-1}), suppose their residual ranking is represented by independent exponential surrogate scores T_i~Exp(w_{Li}) with class rate w. Let the *observed* ascending rank order be i1,...,in0. For positive rates, **standard exponential-race memorylessness** gives the exact conditional joint distribution of sorted original scores:
\[
T_{i_k}=\sum_{j=1}^k \xi_j,\qquad
\xi_j\ \overset{\rm indep}{\sim}\mathrm{Exp}\left(
\sum_{\ell=j}^{n0}w_{L_{i_\ell}}\right).
\]
The spacings are independent of the observed permutation (that permutation has Plackett--Luce probability \(\prod_{j<n0}w_{L_{i_j}}/\sum_{\ell\ge j}w_{L_{i_\ell}}\)). Thus A8 ranks **suffice to condition** the old PL score distribution without knowing actual physical residual values. A8b would improve this by supplying real \(\rho\) gaps; the surrogate conditional law is explicitly model-dependent.

When w_I=0, I candidates have infinite synthetic scores and always appear after passing runs. The one observed FAST case with I outranking G violates that model and must be flagged and handled as a separate S1 exception; do not simply delete the initial row or silently overwrite its rank with the model's expectation.

## 3. Heavy-tail old-C conditioning is NOT a naive epsilon coin toss

Suppose a CONSISTENT candidate has latent subtype H in {ordinary, extreme}, with per-run prior extreme rate epsilon, and subtype-specific synthetic winning weight. For an original class-label sequence and observed rank permutation, the **posterior subtype configuration** obeys
\[
P(H=h\mid L,\mathrm{rank})\ \propto\
P(H=h\mid L)\,
P(\mathrm{rank}\mid L,H=h).
\]
The PL permutation likelihood is explicit. Since n0<=4, there are at most 2^4=16 possible subtype configurations, so exact enumeration is trivial. Draw a subtype configuration using those posterior weights, then use the exponential-spacing formula in section 2. Ignoring the observed ranks in choosing an old C's extreme status would incorrectly treat the strongest S2-rejection evidence as irrelevant.

A new future C receives an independent subtype drawn from epsilon; in the parent model, future class labels are drawn using each fixed image's posterior five-class probabilities.

## 4. Exact conditional subset-policy estimator after completing the hypothetical pool

Given one simulated complete nmax-run score/label pool with the known prefix retained, sort scores by the predetermined tie rule. At ascending residual rank r=1,...,nmax, the probability that this run is the minimum in a uniformly selected K-subset is
\[
w_r^{min}=\binom{nmax-r}{K-1}/\binom{nmax}{K}.
\]
Thus the exact subset-average E0/E1 policy value is obtained by summing the appropriate rank weights of successful candidates. For the oracle, with g GOOD candidates or q orbit-correct candidates in the completed pool,
\[
\widehat O(K)=1-\binom{nmax-g}{K}/\binom{nmax}{K}
\]
(and likewise for q). No subset enumeration is needed. Monte Carlo averaging these exact U-statistics across simulated unknown future pools gives the desired conditional continuation expectation, while panel-to-panel Monte Carlo variability characterizes **finite-run U-statistic noise**.

For image i with observed prefix \(\mathcal D_i\), the scientifically relevant forecast is
\[
E[\widehat{\mathrm{policy}}_{i,K}\mid
 \mathcal D_i,\ \hat\alpha_{\mathrm{cohort}},\ \hat w_{\mathrm{family}}],
\]
and the 180-image point forecast is 100:80 weighted across original image IDs, *not* an independently resampled-image target.

## 5. Assumptions and practical limitations

- The observed e0_rank_order is a complete order of n0 *real* residual values. Conditional spacing law concerns **synthetic** Exponential(w) winning scores, not physical residual units.
- The real score values are unknown until A8b; if A8b does not arrive by the announced gate, freeze with the rank-conditional surrogate and say so.
- Pairwise class-race weights were fitted from the same rank strings (composite likelihood) and used to condition old scores. Uncertainty must block-bootstrap old images by cohort, then re-evaluate the ORIGINAL fixed images.
- For S1's single FAST anomaly, choose and preregister an explicit robust exception model (or separate nonzero w_I sensitivity) before looking at K>4 outcomes. The strict S1 reference is still meaningful as a model reference but not a perfect fit.
- The ideal global-only orientation test is taken from the already-calibrated CLIP rule; the separate partial ORBIT remains unfixable by E1_CLIP, and the no-orbit-candidate failure is (1-q)^K for *fresh K samples*. With a FIXED PREFIX the exact conditional generation probability is zero if the prefix already contains G/O/P, and otherwise \(E[(1-q)^{K-n0}\mid\mathcal D_i]\) for the first-K continuation. For all-subset K-of-nmax, compute the exact conditional empirical oracle from each simulated completed panel; **do not attach the fresh-K formula to that different conditional estimand**.

## 6. Freeze discipline

Do not freeze the existing preliminary fitter's independent-fresh-K curves as though they describe the actual XP continuation. Implement conditional-prefix scoring as the primary forecast. Keep the original fresh-K formulas as a labeled reference and derive honest intervals for the simulated complete-panel U-statistics.

This correction improves scientific validity and does not depend on reading any XP return or K>4 outcome.
