# Track R TH1 extension: residual-score races, rare decoys, and controlled departures from S2

**Date:** 2026-10-26. **Status:** PROVED mathematical statements and CPU-only synthetic tests; **EMPIRICAL / UNTESTED** on DAPS, SITCOM and FAST-DIPS. **Data rule:** no XP returns, no GT files, no confirmation305, no PAC access. The Oct 31 predictions under the preregistered S1–S3 model remain the primary freeze; none of the exploratory alternatives here is fitted to embargoed data.

## 0. Novelty audit comes first

**Direct recent predecessor, discovered before preparing this extension:** Yano, Ishibashi & Oyamada, *Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems*, arXiv:2610.19362, first submitted **14 October 2026**, reports correct-vs-incorrect score-tail competition, non-monotone best-of-N success, Gaussian and general EVT regimes, and an early stopping analysis. Primary full text: https://arxiv.org/abs/2610.19362 ; https://arxiv.org/html/2610.19362 (availability varies). Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023, is an additional predecessor (https://proceedings.mlr.press/v202/gao23h.html). Stroebl, Kapoor & Narayanan (ICLR 2026) already discuss the imperfect-verifier saturation model. **We must not claim tail competition, overoptimization, or the elementary score-race formula as newly invented.**

What Track R can contribute is a **physics-specific, five-label, frozen-data test**: GOOD / globally rotated / partially rotated / consistent-nonorbit / inconsistent, exact Fourier-measurement orbit blindness, solver-induced score distributions, and GT-free residual/orientation selection. This retains and extends the paper's explanation rather than replacing it with generic BoN theory.

## 1. Exact residual-race formula without S1 or S2

Fix one image, observed y, solver and hyperparameters. Let independent runs generate categorical label J in {G, Og, Op, C, I}, with probabilities \(\pi_j\). Conditional on J=j, let the numerical residual R have CDF \(F_j\), absolutely continuous with density \(f_j\). Put \(H(t)=\sum_j\pi_j F_j(t)\). Assume residual ties have probability zero.

**Proposition (standard competing-minima identity; PROVED).**
For residual-argmin over K independent runs, the selected class probability is exactly
\[
W_j(K)=K\pi_j\int_{-\infty}^{\infty}[1-H(t)]^{K-1}\,dF_j(t).
\]
Consequently, E0(K)=W_G(K), and an orientation test with *class-conditional, K- and rank-independent* repair parameters has
\[
E_1(K)=(1-h)W_G(K)+\tau_g W_{Og}(K)+\tau_p W_{Op}(K).
\]
The GT oracles remain \(O_{raw}=1-(1-\pi_G)^K\) and
\(O_{orbit}=1-(1-\pi_G-\pi_{Og}-\pi_{Op})^K\), regardless of residual scores.

**Proof.** Designate one of K run indices. The infinitesimal probability it has class j and score in dt is \(\pi_j dF_j(t)\); the other K-1 scores independently exceed t with probability \([1-H(t)]^{K-1}\). Multiply by K disjoint possible winning indices and integrate. For post-test success, condition on the winner's class. The oracle probabilities depend only on run classes.

**S1--S2 specialization (PROVED).** If all I scores exceed all pass-class scores, and the four pass-class conditional residual distributions coincide with a common continuous F0, then the rank winner is uniform over passers and
\[
W_j(K)=[1-(1-s)^K]\pi_j/s\quad(j\ne I),
\quad s=1-\pi_I.
\]
Thus the registered TH1.2 model is a transparent *special case*, not a competing alternative to the exact score-race model.

**Interpretation:** pointwise Fourier identity \(A(gx)=A(x)\) guarantees paired x and gx have identical scores. It does NOT force separately generated GOOD and ORBIT runs to have the same conditional score CDF, because their off-orbit fluctuations and solver distributions may differ. S2 remains empirically testable, not a consequence of exact Fourier symmetry alone.

## 2. Explicit tiny false-positive tail that makes more samples worse

Consider a wholly synthetic model in which all K candidates pass a consistency criterion (so S1 is vacuous) but the true residual ranking has three strictly ordered supports:

- GOOD: mass p=0.700, residual Uniform[0.2,0.4].
- Rare consistent-but-wrong "extreme" decoy: mass r=0.005, residual Uniform[0,0.1], always outranking GOOD.
- Ordinary consistent-but-wrong decoy: mass h=0.295, residual Uniform[0.8,1.0], always losing to GOOD.

The success probability for a random GOOD-vs-CONSISTENT pairwise comparison is
\[
P(R_G<R_C)=\frac{h}{r+h}=\frac{0.295}{0.300}
=0.98333\ldots.
\]
Yet **exactly**
\[
E_0(K)=(1-r)^K-h^K=0.995^K-0.295^K.
\]
**Proof:** GOOD wins iff (a) none of K samples is a rare extreme decoy and (b) at least one sample is GOOD. The probability of (a) is \((1-r)^K\); the probability of (a) without (b) is \(h^K\). Subtract.

Numbers: E0(1)=0.700; E0(2)=0.903; E0(4)=0.972576; **E0(5)=0.973015 (maximal over integers)**; E0(16)=0.922931; E0(32)=0.851802; E0(64)=0.725566. Yet O_raw(64)=1-(0.3)^64, effectively 1. The initial gain and eventual collapse are **not a novel phenomenon**; Yano et al. 2026 and the reward-overoptimization literature already supply the general theory. Track R's actionable meaning is that **a small CONSISTENT class's extreme low-residual tail**, not merely the average number of CONSISTENT runs, may determine whether K=64 helps.

General three-stratum form: if rare residual-dominant bad mass is r>0, good mass p>0, and ordinary worse-than-good mass h=1-r-p>0, then
\[
E_0(K)=(1-r)^K-h^K.
\]
For a continuous K, its unique positive maximizer is
\[
K_\mathrm{peak}=
\frac{\log\left[\log(h)/\log(1-r)\right]}
     {\log\left[(1-r)/h\right]}.
\]
The integer maximum lies at an adjacent integer. At large K, E0 tends to zero despite O_raw tending to one. For r=0 the collapse disappears, as expected. This is a **stress test**, not a forecast inferred from A8.

## 3. Fully continuous lower-tail asymptotics (standard extreme-value regime)

Suppose each class with positive mass has a common lower residual endpoint t0, no atom there, and as t decreases to t0,
\(F_j(t)\sim a_j(t-t0)^{\alpha_j}\),
\(f_j(t)\sim a_j\alpha_j(t-t0)^{\alpha_j-1}\),
with \(a_j,\alpha_j>0\). Let
\(\alpha_*=\min_{\pi_j>0}\alpha_j\) and
\(C_*=\sum_{j:\alpha_j=\alpha_*}\pi_j a_j\).
Then **PROVED under the stated regular variation and density conditions**:
\[
\lim_{K\to\infty} W_j(K)=
\begin{cases}
\pi_j a_j/C_*,&\alpha_j=\alpha_*,\\
0,&\alpha_j>\alpha_*.
\end{cases}
\]
Sketch of proof: substitute \(t-t0=v(KC_*)^{-1/\alpha_*}\) into the exact integral in section 1. Then \([1-H(t)]^{K-1}\to e^{-v^{\alpha_*}}\); the leading-density terms integrate to \(\pi_j a_j/C_*\) when \(\alpha_j=\alpha_*\), while other classes contribute smaller order \(K^{1-\alpha_j/\alpha_*}\). Split the integral at a small fixed neighborhood of t0 to justify dominated convergence; the complement decays exponentially in K.

This is an application of known competing-tail extreme-value principles, **not** a new generic EVT theorem. In Fourier phase retrieval, exact orbit ties and non-orbit consistent decoys require attention: class residual distributions can have distinct endpoints or point masses, in which case use the full score-race formula and explicit tie rules instead of this continuous asymptotic specialization.

## 4. A finite-K model-discrepancy bound for the preregistered S1--S2 model

The general tail model is too flexible to identify from 2--4 runs per image. A useful alternative is to *bound how far* a residual selector can depart from the ideal five-class formula.

Let P be the four passing labels, with total probability s>0, and I the inconsistent label (mass u=1-s). Let \(F_0\) be a proposed continuous, common passing-residual distribution. For each passing class j let \(d_j=TV(F_j,F_0)\) as distributions of numerical residuals, and define
\[
d=\sum_{j\in P}\pi_j d_j.
\]
Let \(\epsilon=P(R_I<R_P)\), where \(R_I\) is sampled from the conditional I-score law and \(R_P\) is sampled from the passing-class mixture, independently (defined as 0 if u=0).

**Proposition (coupling bound; PROVED).** In the continuous-score, iid-run model, for every class-specific selected-label success criterion with success values in [0,1], the difference between its actual argmin performance and the S1--S2 ideal passing-class prediction is bounded by
\[
\boxed{
\left|E_\mathrm{actual}(K)-E_\mathrm{S1S2}(K)\right|
\le
\min\left(1,\ 1-(1-d)^K + K(K-1)us\epsilon\right).
}
\]
For K=1, the bound is conservative (both selectors are identical, with the same label).

**Proof.** Couple every passing-class score to an iid draw from F0 by an independent maximal coupling, conditional on its label. By the coupling lemma, for a random run a mismatch occurs with probability d, so the probability of any passing-score coupling mismatch among K runs is \(1-(1-d)^K\). For the true residual selector to choose an I-class run when a passer exists, an ordered I/P pair must occur with the former residual smaller. A union bound over K(K-1) ordered pairs gives \(K(K-1)us\epsilon\). Outside these two exceptional events, the actual selected passing index coincides with the ideal reference argmin, whose class law is the S1--S2 formula; if no passer exists, both select a failure label. Coupling success outcomes (for post-test probabilities, use shared independent uniform random variables conditioned on selected class) and the coupling inequality yield the bound.

This makes the frozen S1--S2 assumptions **quantitatively falsifiable**. But estimating per-image distribution-level TV distances or an ultra-rare \(\epsilon\) reliably from n<=4 is not feasible. Such bounds are for interpretation and sensitivity, not an excuse to create data-driven alternative v1 predictions after inspecting held-out results.

## 5. Identifiability warning (PROVED)

Suppose n independent calibration runs contain zero observed instances of a specific rare class with probability r. Then
\[
P(\mathrm{zero\ observed}\mid r)=(1-r)^n.
\]
The one-sided exact 95% upper confidence bound for r in this homogeneous-binomial toy is \(r\le1-0.05^{1/n}\). With n=4, this is about 0.527, meaning four per-image runs cannot rule out a modest rare-tail rate. Pooling is stronger only when the homogeneous-rate assumption is scientifically defensible; image heterogeneity changes the analysis. Also, observing the class CONSISTENT does not identify its **extreme residual tail** from class counts alone.

Therefore report the preregistered Dirichlet-mixture forecasts as **model-conditional extrapolations**, not robust distribution-free CIs about best-of-K. If post-freeze observations deviate, distinguish structural S2 failure, score tails, class-probability heterogeneity and independent-run errors.

## 6. Practical scientific positioning

**TH1 primary (unchanged):** exact physics-based group symmetry + posterior prior ratios; three-solver failure taxonomy; per-image five-class calibration at n<=4; prospectively registered K forecasts; exact unbiased K-subset estimators.

**TH1 exploratory/interpretive (this document):** score-race integrals, rare-tail stress test, quantitative S2-discrepancy bound, and eventual measurement of class-conditioned extreme residual ranks. All are mathematically inherited or straightforward specializations and do not replace the main physical contribution.

**TH2:** class-conditioned score tails alone do not identify *why* a solver lands in a wrong basin. The separate TH2 study should model how DAPS MCMC, SITCOM optimization and FAST-DIPS ADMM determine occupancy, with explicit hypotheses and verifiable source-level assumptions, rather than relabeling existing tail theory as a new inner-functional theorem.

## 7. Related primary sources

- Taro Yano, Yoichi Ishibashi, Masafumi Oyamada. **Tail Competition Explains Scaling in Best-of-N Sampling for Verifiable Problems**, arXiv:2610.19362, 14 October 2026. https://arxiv.org/abs/2610.19362
- Leo Gao, John Schulman, Jacob Hilton. **Scaling Laws for Reward Model Overoptimization**, ICML 2023. https://proceedings.mlr.press/v202/gao23h.html
- Stroebl, Kapoor, Narayanan, ICLR 2026 imperfect-verifier model (see THEORY_PROGRAM and existing theory literature digest).
- Hoeffding (1948) U-statistics; Fisher--Tippett--Gnedenko classical extreme-value theory.
