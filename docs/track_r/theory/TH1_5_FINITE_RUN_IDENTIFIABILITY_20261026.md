# TH1.5: why two or four calibration runs cannot identify high-K behavior nonparametrically

**Date:** 2026-10-26. **Status:** PROVED elementary truncated-moment construction; numerics checked on synthetic distributions. **Scope:** mathematical risk statement for the October 31 FFHQ prediction freeze. No A8 / XP data or GT images used.

## 1. Exact observational indistinguishability at calibration depth n

Consider one image whose run success probability is \(p\in[0,1]\). Across the image population, \(p\sim\mu\) can vary. On each image we observe at most \(n\) independent success labels, giving \(G\mid p\sim\operatorname{Binomial}(n,p)\).

**Proposition (finite-run moment obstruction; classical truncated moment problem).** If two mixing distributions \(\mu_1,\mu_2\) have the same ordinary moments \(E[p^j]\) for \(j=0,\ldots,n\), then they induce **exactly the same distribution** of every \(n\)-run count \(G\), including with an arbitrarily large number of independently sampled images. But for \(K>n\), their \(K\)-run oracle success expectations \(1-E[(1-p)^K]\) may differ.

**Proof.** \(P(G=g)=\binom ng E[p^g(1-p)^{n-g}]\). The polynomial inside the expectation has degree n, so its value depends only on the first n moments of \(\mu\). For K>n, \(p\mapsto(1-p)^K\) is a polynomial of strictly larger degree, not determined in general by the truncated moment sequence.

This obstruction extends to all five run classes: the \(n\)-run multinomial count likelihood \(\frac{n!}{\prod_j C_j!}\prod_j\theta_j^{C_j}\) is a polynomial in the latent class probabilities with total degree n. Even infinitely many image records of exactly n trials would identify at most their mixed moments through that degree. The two-class case below is already a counterexample within the five-class model (assign all other class probabilities zero).

## 2. A fully explicit pair with matching every n=4 observable

Let \(P_5\) be the ordinary Legendre polynomial of degree 5. Define two densities on \(0\le p\le1\):
\[
 f_\pm(p)=1\pm P_5(2p-1),\qquad
 P_5(t)=(63t^5-70t^3+15t)/8.
\]
Because \(|P_5(t)|\le1\) for \(t\in[-1,1]\), both are nonnegative. Since \(P_5\) is orthogonal to polynomials of degree at most 4 on [-1,1], each f integrates to one and the first four moments are identical (in fact, those of Uniform[0,1]).

Consequently **no possible 4-run count table, even evaluated across infinitely many independent images, distinguishes \(f_+\) from \(f_-\)**.

For \(K\ge5\), Rodrigues' formula and repeated integration by parts give
\[
\begin{split}
I_{5,K}&=\int_0^1 (1-p)^K P_5(2p-1)\,dp\\
&=-\frac{K(K-1)(K-2)(K-3)(K-4)}
{(K+1)(K+2)(K+3)(K+4)(K+5)(K+6)}.
\end{split}
\]
Thus the two high-K oracle success probabilities differ by
\[
\left|\mathbb E_+[1-(1-p)^K]-\mathbb E_-[1-(1-p)^K]\right|
=2|I_{5,K}|.
\]
Their disparity is 0 at K<=4, about **1.951 percentage points at K=16**, **2.431 points at K=32** and **1.938 points at K=64**. This construction is deliberately simple; it is *not* a claim of an optimal identification interval.

For the DAPS B22 stratum with only n=2 runs, replace P5 with P3: \(f_\pm(p)=1\pm P_3(2p-1)\). The first two moments match exactly while K=8 oracle success differs by **5.657 percentage points**.

These effects exist **without measurement noise, imperfect S1/S2, image correlations, parameter estimation errors, or rare tail selection effects**. They are purely finite-run mixing-distribution ambiguity.

## 3. Distribution-free shape bounds that survive nonidentification

For any mixture of per-image probabilities \(p\in[0,1]\), define \(O_K=E[1-(1-p)^K]\). Then

1. \(O_K\) is nondecreasing with K;
2. \(O_{K+1}-O_K=E[p(1-p)^K]\) is nonincreasing in K;
3. Given the true \(O_n\), for any K>=n,
\[
 \boxed{O_n\le O_K\le 1-(1-O_n)^{K/n}}.
\]

**Proof of (3).** Let \(T=(1-p)^n\in[0,1]\). For K>=n, \(T^{K/n}\le T\), so \(E[T^{K/n}]\le E[T]=1-O_n\) (yielding the lower success bound). Jensen's inequality for the convex map \(t\mapsto t^{K/n}\) gives \(E[T^{K/n}]\ge E[T]^{K/n}\) (yielding the upper success bound). These bounds apply to *true population expectations*, not automatically to finite-sample estimates without interval adjustments.

For the S1--S2 class selector, \(E_0(K)=E[(p/s)(1-(1-s)^K)]\), with its summand zero when s=0. It is also nondecreasing and discretely concave in K. Likewise for \(E_1\) when repair probabilities are fixed. The monotone *oracle-minus-selector gaps* require the stronger S1--S2 structure.

## 4. Consequences for the preregistered prediction model

**Critical:** A Dirichlet-multinomial empirical Bayes fit to A8 per-image n<=4 counts is a structured *model-conditional extrapolator*. Its high-K point estimates and bootstrap credible/predictive intervals necessarily rely on the Dirichlet family and shared hyperparameters. They are not nonparametric identification intervals. A tight high-K interval cannot be justified from four-count data without structural assumptions.

The concentration prior can materially affect how rapidly per-image failure probabilities extrapolate. In the limit K large, some images' near-zero successful-class probability is especially important; a mixture with structural-zero image types and one without them may look identical over n=4 trials while having different asymptotes. Don't conflate model-conditional uncertainty with robustness to misspecification.

**Recommended pre-freeze protocol (no change to registered target):**
- Keep the five-class Dirichlet-multinomial model and seed as the primary predicted curve. Document model nonidentifiability clearly.
- Report a sensitivity study based only on allowed n<=4 data, e.g. a parsimonious two-type mixture fitted without seeing higher-K results; label it sensitivity, **not a second headline forecast chosen after the fact**.
- Keep separate B22 and DEV80 fits; a pooled forecast should preserve the known 100:80 cohort weights rather than assuming a single homogeneous Dirichlet prior across two deliberately different strata.
- Use the data's rank strings to test S1 and S2 whenever A8 arrives, but recognize that extreme residual tails are not identified from short rank strings.

## 5. Primary intellectual precedents

The mathematical phenomenon belongs to the classical **Hausdorff/truncated moment problem** and nonidentifiability of finite mixtures of binomials. No new moment-problem theorem is claimed. The Annual Review of Statistics and Its Application article *Finite Mixture Models* (2019) discusses conditions for identifiability of finite mixtures of binomial laws, citing Teicher's historical results (1960). The explicit shifted-Legendre construction and high-K numerical illustrations here specialize elementary orthogonality to Track R's four-run calibration budget.
