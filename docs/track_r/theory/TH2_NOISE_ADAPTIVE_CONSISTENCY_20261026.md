# TH2/TH1 bridge: noise-adaptive consistency, effective fit degrees of freedom, and exact-orbit blindness

**Date:** 2026-10-26. **Status:** PROVED statements for fixed-candidate Gaussian tests and linearized ridge/least-squares models; SYNTHETIC CPU numerical verification; **CONDITIONAL** application to the high-dimensional FFT nonlinear solvers. No PAC, FFHQ/GT image bytes, XP returns, or confirmation305 were used. Neither Stein optimism nor inverse-problem discrepancy principles are new; this document seeks a more physically precise failure explanation, not a new theorem in statistical estimation.

## 1. Three different uses of the phrase “measurement-consistent”

Track R uses \(y=A(x_\star)+\sigma Z\), \(Z\sim N(0,I_m)\), with **signed** y, a magnitude operator \(A\), \(m=3\cdot384^2=442{,}368\), and \(\sigma=0.05\). Our empirical CONSISTENT label means **a reconstruction fails the target orbit/Good25 evaluation and its realized normalized residual \(\rho(\nu)=\|A(\nu)-y\|/\sqrt m\) is below a specified threshold**. This is not the same as either:

1. *Exact inverse ambiguity*: \(A(\nu)=A(x_\star)\) for all admissible noise realizations. Every \(g\cdot x_\star\) has this property; other exact ambiguities may also occur.
2. *Statistical indistinguishability*: the probability distributions \(N(A(\nu),\sigma^2I_m)\) and \(N(A(x_\star),\sigma^2I_m)\) are close in total variation, or their KL divergence is small.
3. *Realized residual consistency*: both outputs lie within a broad observation-dependent threshold ball around the **same sampled y**. This criterion can hold even when the underlying generative distributions are highly distinguishable.

Separating these notions may be a valuable contribution to how the CONSISTENT class is interpreted. Do not write that every empirical CONSISTENT decoy is an exact second solution or an intrinsic non-identifiability of the observation. That inference would be false.

## 2. Fixed alternatives: exact Gaussian testing and noncentral residual law (PROVED)

Fix two candidates x and \(\nu\) **independently of the noise realization**, and put \(d=A(\nu)-A(x)\), \(\delta=\|d\|\). Under \(Y=A(x)+\sigma Z\),
\[
\frac{\|Y-A(x)\|^2}{\sigma^2}\sim\chi_m^2,\qquad
\frac{\|Y-A(\nu)\|^2}{\sigma^2}\sim\chi_m^2(\lambda),\quad
\lambda=\delta^2/\sigma^2.
\]
The two displayed statistics are correlated because they use the *same* noise vector. The exact paired score difference is
\[
\|Y-A(\nu)\|^2-\|Y-A(x)\|^2
=\delta^2-2\sigma\langle Z,d\rangle.
\]
For \(\delta>0\), likelihood/residual ranking identifies the correct fixed candidate with probability
\[
\Pr(\rho(x)<\rho(\nu))=\Phi(\delta/(2\sigma)).
\]
At \(\delta=0\), the scores are exactly tied for every realization y; the tie-breaking convention determines the selected orientation.

The KL divergence of the two *measurement distributions* is
\[
D_{\rm KL}(N(A(x),\sigma^2I)\|N(A(\nu),\sigma^2I))
=\delta^2/(2\sigma^2).
\]
For equal prior weights, their optimal Bayes classification error is \(\Phi(-\delta/(2\sigma))\). These are standard Gaussian-shift facts, not new PR-specific inference bounds.

**Scale demonstration:** With the actual Track R m and sigma, take a hypothetical fixed candidate with \(\delta=5\sigma=0.25\). Then the root-mean-square residual floor at x is approximately \(\sigma=0.05\), while the square root of the expected \(\rho(\nu)^2\) is
\[
\sigma\sqrt{1+25/m}\approx 0.05000141.
\]
The difference is minuscule at the normalized-residual scale, yet the *paired* residual ordering favors x with probability \(\Phi(2.5)\approx 99.379\%\), and the two generative measurement models have KL=12.5 nats. This is only a fixed-pair mathematical example; no assertion that such a candidate occurs in FFHQ or passes any empirically selected threshold.

**Lesson:** a broad residual-consistency label and high-confidence statistical discrimination can coexist. On exact Fourier orbits \(\delta=0\), by contrast, the measurement truly contains no orientation information.

## 3. The missing mechanism: a reconstruction chosen AFTER observing y can overfit measurement noise

A practical diffusion solver's returned reconstruction \(\hat x(y)\) depends on the noisy y. Therefore the fixed-candidate Gaussian testing formula does **not** apply to its residual ranking against the unknown x* without further conditional analysis. A simple linearized calculation makes the distinction exact.

Suppose near x* the forward map is modeled by \(A(x_\star+v)\approx a+Jv\) with \(a=A(x_\star)\), \(J\in\mathbb R^{m\times d}\) fixed, and \(y=a+\sigma Z\). Let rank(J)=r. Let \(P\) be the orthogonal projector onto range(J), and let the unrestricted linear least-squares fit have fitted measurement \(a+\sigma PZ\). Then:

**Proposition 1 (training residual optimism, PROVED).**
\[
\begin{aligned}
\|y-a\|^2-\|y-(a+\sigma PZ)\|^2
&=\|\sigma PZ\|^2\ge0\quad\text{a.s.},\\
\mathbb E\|y-(a+\sigma PZ)\|^2&=\sigma^2(m-r),\\
\mathbb E\|\sigma PZ\|^2&=\sigma^2r.
\end{aligned}
\]
If r>0, residual fitting strictly improves the observed y with probability one, even though the fitted measurement moves **away** from the clean truth a. The residual's exact squared distribution is \(\sigma^2\chi^2_{m-r}\). This is a classical projection/overfitting identity.

**Proof:** Decompose \(Z=PZ+(I-P)Z\) into orthogonal components. The in-sample residual is \(\sigma(I-P)Z\), whose norm squared is orthogonal to the fitted displacement \(\sigma PZ\). Orthogonal Gaussian components have chi-square degrees of freedom r and m-r.

For the actual dimensionalities \(d=3\cdot256^2=196{,}608\) and \(m=442{,}368\), rank \(r\le d\). The largest *possible* linearized least-squares residual reduction occurs if \(r=d\): the expected residual RMS becomes \(\sigma\sqrt{1-d/m}\approx0.03727\) instead of 0.05. This is a **mathematical rank-cap illustration**, not a measured Jacobian rank or realistic prediction for diffusion solvers; image-cube constraints, prior regularization, nonlinear operator geometry, and finite-step optimization may greatly reduce actual accessible fit directions.

**Proposition 2 (ridge regularization trades noise fit for clean signal risk, PROVED).**
For \(\lambda>0\) solve
\[
\hat v_\lambda=\arg\min_v\{\|Jv-\sigma Z\|^2+\lambda\|v\|^2\}.
\]
Let \(H=J(J^\top J+\lambda I)^{-1}J^\top\), a symmetric matrix with eigenvalues \(h_i\in[0,1)\), and define \(\hat a=a+\sigma HZ\). Then
\[
\begin{split}
\mathbb E[\rho(\hat x)^2]&=\frac{\sigma^2}{m}\operatorname{tr}(I-H)^2,\\
\mathbb E\|\hat a-a\|^2&=\sigma^2\operatorname{tr}(H^2),\\
\mathbb E\|y-a\|^2-\mathbb E\|y-\hat a\|^2
&=\sigma^2(2\operatorname{tr}H-\operatorname{tr}H^2),\\
\mathbb E\|Y'-\hat a\|^2&=\sigma^2(m+\operatorname{tr}H^2),
\end{split}
\]
where \(Y'=a+\sigma Z'\), with fresh independent \(Z'\), is a hypothetical replicated observation of the same clean x*. Consequently
\[
\boxed{\mathbb E\|Y'-\hat a\|^2-
       \mathbb E\|y-\hat a\|^2=2\sigma^2\operatorname{tr}H.}
\]
This is precisely the familiar Stein/SURE optimism identity for a linear smoother. No new method for selecting \(\lambda\), no re-noised acquisition, and no new solver is proposed.

**Proof:** Diagonalize H. Compute \(y-\hat a=\sigma(I-H)Z\), \(\hat a-a=\sigma HZ\), and \(Y'-\hat a=\sigma(Z'-HZ)\). Orthogonality/zero covariance of independent normal vectors yields the traces. Subtraction gives the last equation.

## 4. Why this matters specifically for Track R

**The physical measurement can be exact-orbit blind and still numerically permissive to noise-adaptive wrong reconstructions.** These are two different mechanisms:

- A strict **GROUP ambiguity** \(A(gx)=A(x)\) survives without noise or approximation. No measurement-only verifier can prefer one such member.
- A **CONSISTENT non-orbit decoy** may be an exact alternative, an approximate alternative, or a measurement-noise-adapted fitted output. A low residual is neither proof of exact ambiguity nor proof of posterior support. Its distribution can depend strongly on whether the solver minimizes a data penalty, performs MCMC corrections, or uses feasibility projections.
- The solver's use of y makes apparent residual accuracy optimistic. For the same noisy y, blindly choosing the smallest residual can reward fitting noise, as illustrated by the linearized result. This connects inverse-problem discrepancy principles and Stein optimism to the imperfect-verifier category already known from Stroebl.

The local-Jacobian theory is a **first-order diagnostic thought experiment**. The FFT magnitude derivative can be nonsmooth at Fourier zeros; the network priors are non-Gaussian; nonconvex optimization and multiple basins preclude directly applying the linear formula to the real output. One should not estimate a solver's effective degrees of freedom from its observed residual alone without accounting for bias, stochasticity, and model mismatch.

### A nonmethod audit that could genuinely differentiate the paper

Without changing the preregistered experiment, compare the already recorded y-residual distribution and failure labels across the three solvers. Ask whether CONSISTENT outputs have systematically smaller \(\rho\) than GOOD outputs (low-residual false-positive tail), or whether they simply pass the threshold while remaining worse than GOOD. The two cases make very different K-scaling predictions; the existing exact residual-race identity in TH1_SELECTION_TAILS_20261026.md already covers them. If no sufficiently rich residual ranks exist in A8, leave this for a **post-freeze exploratory** analysis; do not infer an answer from aggregate P4 alone.

## 5. Prior art and scope

- Stein (1981) unbiased estimation of risk and degrees of freedom; Efron (2004) prediction optimism.
- Morozov discrepancy principle; Mead & Hammerquist (2013), chi-square tests for nonlinear inverse problems, SIAM Journal on Matrix Analysis and Applications, DOI 10.1137/12088447X.
- DAPS, CVPR 2025; SITCOM, ICML 2025; FAST-DIPS, ICLR 2026; Stroebl et al., ICLR 2026.
- Exact Fourier ambiguity (Hayes and classical phase retrieval) and zero-flip toy are previously established, not claimed new.

**Novelty status:** NO claim of a new Gaussian statistics result or projection theorem. Potential differentiation is a precise **measurement-physics + noise-adaptive verification** interpretation, paired with the forthcoming cross-solver empirical failure and held-out K outcomes.
