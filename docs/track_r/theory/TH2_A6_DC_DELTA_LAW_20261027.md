# TH2 A6: exact D/C/Delta law for a locally tempered Gaussian posterior sample

**Date:** 2026-10-27. **Status:** PROVED for the stated local Gaussian posterior and flat identifiable subspace, not proved for deployed finite-step diffusion samplers. **Source restriction:** uses only the planner's pinned native weighting and authorized K<=4 N1c/A8 aggregates. No A5.5 results or XP outcomes accessed.

Let \(y=a+\sigma Z\), with Z standard normal in m measurement coordinates, and let P project onto the rank-r range of the local measurement Jacobian. For a posterior \(\kappa\)-weighted data term and a flat prior along the r identifiable directions, the fitted posterior-sample measurement displacement can be represented as
\[
d=A(\hat x)-a=\sigma PZ+\frac{\sigma}{\sqrt\kappa}P\xi,\quad Z,\xi \stackrel{\mathrm{iid}}{\sim}N(0,I_m).
\]
Define the A5.5 random statistics
\[
D=\|d\|^2/\sigma^2,\qquad
C=\langle y-a,d\rangle/\sigma^2,\qquad
\Delta=D-2C.
\]
Then, **exactly and pathwise**,
\[
\begin{aligned}
D&=\|PZ\|^2+2\kappa^{-1/2}\langle PZ,P\xi\rangle
       +\kappa^{-1}\|P\xi\|^2,\\
C&=\|PZ\|^2+\kappa^{-1/2}\langle PZ,P\xi\rangle,\\
\boxed{\Delta}&=\boxed{\kappa^{-1}\|P\xi\|^2-\|PZ\|^2}.
\end{aligned}
\]

**Proposition (proved).** With rank(P)=r>0,
\[
\begin{aligned}
\mathbb ED&=r(1+\kappa^{-1}),&
 \operatorname{Var}D&=2r(1+\kappa^{-1})^2,\\
\mathbb EC&=r,&
 \operatorname{Var}C&=r(2+\kappa^{-1}),\\
\mathbb E\Delta&=-r(1-\kappa^{-1}),&
 \operatorname{Var}\Delta&=2r(1+\kappa^{-2}).
\end{aligned}
\]
Moreover \(\Delta\stackrel{d}{=}\kappa^{-1}\chi_r^2-\widetilde\chi_r^2\) with independent chi-square variables, giving
\[
\Pr(\Delta<0)=F_{r,r}(\kappa),
\]
the CDF of an F distribution with degrees of freedom (r,r) at \(\kappa\). Thus it is 1/2 at \(\kappa=1\), greater than 1/2 for \(\kappa>1\), and tends to one for large r at fixed \(\kappa>1\). A simple (loose) one-sided risk bound is
\[
\Pr(\Delta\ge0)\le
\frac{2(1+\kappa^{-2})}{r(1-\kappa^{-1})^2}
\quad(\kappa>1).
\]

**Proof:** PZ and Pξ are independent standard Gaussian vectors on an r-dimensional subspace. Expand D,C and cancel the cross term in Delta. The chi-square means/variances give the formulas; for C, the independent-on-expectation covariance of \(\|PZ\|^2\) and \(\langle PZ,P\xi\rangle\) is zero. For Delta use independent chi-square laws and the symmetry of the F ratio. Chebyshev's inequality yields the stated upper bound.

**Comparison with actual runs:** The local model's D and C concern the candidate versus the clean forward signal a and overfitting one fixed noisy observation; A5.5 by class tests whether the dynamics display that behavior. DAPS and SITCOM have reported effective inner data weights κ≈50 and 25, respectively, but their stationary distribution is NOT demonstrated to be the local Gaussian posterior, and SITCOM has an extra Euler SSE-gradient correction. Their empirical \(\rho\) medians are not estimates of \(\mathbb E\rho^2\) or exact active rank. No numeric r is claimed independently of the toy.

**For FAST-DIPS:** If the returned denoiser displacement is almost independent of the measurement noise and is prior-dominated, then \(E[C]\approx0\), so \(E[\Delta]\approx E[D]\ge0\). But this is conditional; the actual denoiser depends on y through the trajectory, so the independence hypothesis cannot be presumed. The FAST inner projection epsilon is NOT the returned x0hat epsilon tolerance. A class-conditioned positive Delta is a testable hypothesis, not a proven FAST distributional property.

**Orientation leak in D/C:** For each fixed image with G and globally rotated O runs,
\[
\rho_O^2-\rho_G^2=
\frac{\sigma^2}{m}\left[(D_O-D_G)-2(C_O-C_G)\right].
\]
If O were exactly the 180-degree rotation of G, the two FFT magnitudes and hence residuals are identical: \(D_O=D_G,C_O=C_G\). Observed DAPS/FAST G preference in rank comparisons (DAPS 21/24; FAST 15/15) therefore reports *differences in the generated trajectories' off-orbit perturbations* and their alignment with the same measured noise, not measurement information that distinguishes exact orbit representatives. SITCOM 14/30 is consistent with equal G/O ranking. The local sample law does not by itself determine which method should have the larger orientation leak.

**Novelty accounting:** These Gaussian calculations, F-law and noise-fit identities are classical. The genuinely testable Track R contribution would be alignment (or a principled mismatch) between A5.5 class-resolved observables and this *pre-frozen, source-informed mechanism model* in the three published solvers.
