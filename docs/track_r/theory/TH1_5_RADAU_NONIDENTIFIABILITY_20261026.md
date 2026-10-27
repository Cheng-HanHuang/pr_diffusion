# TH1.5 strengthening: explicit Gauss–Radau nonidentifiability at K=64

**2026-10-26. PROVED**, elementary finite-moment/quadrature construction and synthetic CPU regression. **NOT** a result about the unknown real Track R calibration distribution. No A8 data, XP K>4 outcomes, PAC, GT images or confirmation305 were used.

This supplements, without replacing, TH1_5_FINITE_RUN_IDENTIFIABILITY_20261026.md. The earlier shifted-Legendre pair demonstrated modest high-K separation; the present Gauss–Radau construction exhibits **structural-zero mass and a substantial asymptotic separation while matching all admissible n-run count distributions exactly**.

## 1. What finite-n data can and cannot identify

Fix a family and an image cohort. Each image has a latent probability p of yielding GOOD in one independent run. Its n-run GOOD count G obeys
\[
G\mid p\sim {\rm Binomial}(n,p),\qquad p\sim\mu .
\]
The entire observable distribution of G is determined by the first n moments of mu:
\[
\Pr(G=g)=\binom ng\int p^g(1-p)^{n-g}\,\mu(dp).
\]
But the population oracle curve is
\[
O_K=\int[1-(1-p)^K]\,\mu(dp),\quad
O_\infty=\Pr_\mu(p>0).
\]
The second identity follows from dominated convergence and includes a *potential* structurally impossible-image fraction \(\Pr_\mu(p=0)\). Such a fraction must not be asserted in a real diffusion solver just from four failures; arbitrarily tiny positive p values may also be observationally indistinguishable at n<=4.

**General observation:** The finite-n counts determine only polynomial functionals through degree n. No matter how many independently drawn images are observed, they do not identify \(\Pr(p=0)\) or the K>n curve without structural assumptions on mu.

## 2. Completely explicit n=4 pair

Population A has \(p\sim{\rm Uniform}[0,1]\). Population B has exactly three image types:
\[
\begin{array}{c|c}
\text{latent per-run GOOD probability}\ p&
\text{image-population mass}\\
\hline
0&1/9\\
(6-\sqrt6)/10&(16+\sqrt6)/36\\
(6+\sqrt6)/10&(16-\sqrt6)/36
\end{array}
\]
These are the three-node left-endpoint **Gauss–Radau quadrature** points and weights for a uniform measure on [0,1]. Every polynomial of degree <=4 has exactly the same integral under A and B (this can also be checked directly for the five monomials \(1,p,\ldots,p^4\)). Thus:
\[
\Pr_A(G=g)=\Pr_B(G=g)=1/5,\qquad g=0,1,2,3,4.
\]
**Even an infinite cohort with four runs per image cannot distinguish A from B using GOOD counts.**

At larger K, however,
\[
O^{A}_K=\frac{K}{K+1},\qquad
O^{B}_K=1-\sum_{j=0}^2 w_j(1-p_j)^K.
\]
In particular,
\[
\begin{array}{r|rrr}
K& O^A_K&O^B_K& O^A_K-O^B_K\\
\hline
4&0.800000&0.800000&0\\
8&0.888889&0.873547&0.015342\\
16&0.941176&0.888430&0.052747\\
32&0.969697&0.888888&0.080808\\
64&0.984615&0.888889&0.095726
\end{array}
\]
Population A has no zero-success images, so \(O^A_\infty=1\).
Population B has mass 1/9 at p=0, so \(O^B_\infty=8/9\).
The two asymptotic oracles thus differ by **11.11 percentage points** despite identical four-run count distributions.

This construction is *not* claimed to be a sharp upper bound at K=64 or a plausible fitted description of Track R's FFHQ cohorts. It proves substantial distribution-free nonidentifiability in a concrete setting.

## 3. The n=2 DAPS/B22 case is even less identified

Population A remains Uniform[0,1]. Population B now has two image types:
\[
\Pr(p=0)=1/4,\qquad
\Pr(p=2/3)=3/4.
\]
The two populations match moments 0,1,2 and therefore produce exactly the same two-run count distribution:
\[
\Pr_A(G=g)=\Pr_B(G=g)=1/3,\quad g=0,1,2.
\]
Yet at K=8, oracle GOOD probabilities are approximately 0.888889 versus 0.749886, a **13.90-point** difference. As K grows, their asymptotic success limits differ by 25 points.

**Do not confuse these toy values with the observed DAPS/B22 rates.** They show what the calibration *design cannot identify in general*, not that the actual solver must have an impossible-image subpopulation of any particular size.

## 4. Consequences for the October 31 preregistration

The proposed five-type Dirichlet-multinomial empirical-Bayes model is meaningful as a **structurally regularized extrapolator**, but its uncertainty bands do not capture such an alternative latent-population law unless the alternative is represented within the model. A distribution with mass at p=0 violates a standard Dirichlet with strictly positive concentration parameters. Bootstrap resampling image rows does not by itself repair this model misspecification.

The scientific claim to retain is: (i) empirically observable curve shapes on K<=4; (ii) predictions conditional on a clearly stated population model, frozen before post-K4 evidence; (iii) transparent failure and success of extrapolation; (iv) a discussion of why this is intrinsically a difficult prediction problem. This *broadens* Track R's interpretation, instead of forcing an artificial novelty claim on the textbook quadrature.

For population mixtures under S1–S3,
\[
\lim_{K\to\infty}O_{\rm raw}(K)=\Pr(p>0),
\quad
\lim_{K\to\infty}O_{\rm orbit}(K)=\Pr(q>0),
\]
and
\[
\lim_{K\to\infty}E_1(K)
 =\mathbb E\!\left[\mathbf1\{s>0\}\,a/s\right],
\]
where \(a=(1-h)p+\tau_g o_g+\tau_p o_p\), \(s=p+o_g+o_p+c\).
The orbit oracle minus E1 ceiling is
\[
\mathbb E\!\left[\mathbf1\{q>0\}(1-a/s)\right],
\]
with the convention that the quotient is zero when s=0. This corrects any implicit assumption that *every* image must eventually generate a successful candidate.

## 5. Prior mathematics and provenance

The construction is standard **Gauss–Radau quadrature**, which fixes one endpoint and integrates polynomials up to degree \(2m-2\) with m nodes; see the deal.II QGaussRadau documentation and Gautschi's *Numerical Analysis*. The general nonidentifiability is an instance of the classical Hausdorff/truncated moment problem. **Neither the quadrature theorem nor this moment obstruction is a new general mathematical result.** The Track R-specific use is to audit how credible an extrapolation from 2/4 reconstruction attempts per FFHQ image can be.

**CPU receipt:** scripts/track_r/theory/check_th1_radau_obstruction.py verifies all low-order moments and every n-run binomial count probability, computes K<=256 oracle curves, and prints TH1_RADAU_OBSTRUCTION_PASS.
