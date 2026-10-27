# Track R TH2 — over-weighted posterior samples, noise-floor anatomy, and paired residual ranks

**2026-10-27. Status:** PROVED exactly in linear-Gaussian models and for algebraic paired residual identities; *conditionally interpretable* for nonlinear Fourier/diffusion solvers, whose states and learned priors violate several assumptions. Numeric inputs below come **only from authorized N1c K<=4 summary and planner-provided source configuration**. No XP returns, new run indices, PAC, GT image bytes, or holdout access.

## 1. Posterior *draws* at exaggerated likelihood precision: exact solvable model

Let \(y=a+\sigma Z\), \(Z\sim N(0,I_m)\), where \(a=A(x_\star)\) is the clean signal and \(m=3\cdot384^2=442368\). Locally linearize \(A(x_\star+v)=a+Jv\). Suppose the local reference prior for \(v\) is Gaussian \(N(0,\Sigma_0)\), with \(\Sigma_0\succ0\), and let a *tempered data term* assign the artificial posterior density
\[
\pi_\kappa(v\mid y)\ \propto\
\exp\!\left[-\frac{\kappa}{2\sigma^2}\|Jv-(y-a)\|^2\right]
\exp\!\left[-\frac12 v^\top\Sigma_0^{-1}v\right].
\]
For \(\kappa=1\), the Gaussian likelihood has the statistically correct precision; \(\kappa>1\) over-weights data. **No assertion that finite-step DAPS/SITCOM exactly sample this posterior.** Their score corrections only motivate the toy effective precision.

Let
\[
\Sigma_\kappa=
(\Sigma_0^{-1}+\kappa J^\top J/\sigma^2)^{-1},\qquad
H_\kappa=\frac{\kappa}{\sigma^2}J\Sigma_\kappa J^\top .
\]
Then \(H_\kappa\) is symmetric positive semidefinite with eigenvalues in [0,1), and an independent posterior draw can be written
\[
\hat a_\kappa=a+\sigma H_\kappa Z+J\eta,\quad
\eta\sim N(0,\Sigma_\kappa),\quad\eta\perp Z.
\]
The returned measurement residual is
\[
y-\hat a_\kappa=\sigma(I-H_\kappa)Z-J\eta.
\]

**Proposition 1 (EXACT posterior-draw residual expectation).**
\[
\boxed{
\frac{\mathbb E\rho(\hat a_\kappa)^2}{\sigma^2}
=\frac1m\left[\operatorname{tr}(I-H_\kappa)^2+
\frac1\kappa\operatorname{tr}H_\kappa\right].
}
\]
**Proof:** The cross term vanishes because Z and η are independent; the first trace gives the squared error of the posterior mean. The identity \(J\Sigma_\kappa J^\top/\sigma^2=H_\kappa/\kappa\) evaluates the extra posterior sampling variance. This \(H_\kappa/\kappa\) term is precisely what is missing from a MAP-only residual calculation.

**Proposition 2 (flat active subspace: planner's formula verified exactly).**
If the prior is flat on r measurement-identifiable directions and J has rank r, \(H_\kappa=P\) becomes the orthogonal rank-r projector. Then
\[
\boxed{
\frac{m\rho^2}{\sigma^2}\ \overset{d}{=}\ 
\chi^2_{m-r}+\kappa^{-1}\chi^2_r,
}
\]
with independent chi-square variables, hence
\[
\boxed{
\frac{\mathbb E\rho^2}{\sigma^2}
=1-\frac rm(1-\kappa^{-1}),\qquad
\operatorname{Var}\left(\frac{\rho^2}{\sigma^2}\right)
=\frac2{m^2}\left[(m-r)+\frac r{\kappa^2}\right].
}
\]
**Proof:** \(y-\hat a_\kappa=\sigma(I-P)Z-(\sigma/\sqrt\kappa)P\xi\) with independent standard Gaussians; its components lie in orthogonal subspaces.

The planner's proposed expression is therefore **exact for posterior samples, not just MAP estimates**, under this locally linear and flat-prior idealization. It need not hold with a finite prior, a strongly nonlinear forward map, state-dependent sampling, or finite-step bias.

With a proper Gaussian prior, let \(h_i\) denote eigenvalues of \(H_\kappa\). Relative squared residual contribution along each fitted singular direction is
\[
(1-h_i)^2+h_i/\kappa.
\]
The effective shrinkage \(h_i=h_i(\kappa)\) changes when the data-term weight changes. For \(\kappa=1\) the contribution is \(1-h_i+h_i^2\), which need not equal one; calibration toward the physical noise scale is *not* a universal equality outside the flat identifiable limit.

## 2. What the authorized N1c K<=4 numbers do and do not imply

The planner and N1c give GOOD-run median \(\rho\), with \(\sigma=0.05\), approximately:
- DAPS B22: \(0.047306\) (0.94612 sigma), effective \(\kappa=50\) from native data-gradient weighting;
- SITCOM B22: \(0.04418\) (0.8836 sigma), effective \(\kappa=25\) from its inner SSE Langevin term, plus a separate Euler step;
- FAST B22: \(0.049827\) (0.99654 sigma); its terminal output is denoiser \(x0hat\) rather than the last projected \(x0y\).

Naively identifying a **median** of \(\rho\) with the square root of \(\mathbb E\rho^2\) and imposing the flat model gives *illustrative*, NOT ESTIMATED model-free, active dimension ratios:
\[
r/m\approx\frac{1-(\rho_{\mathrm{median}}/\sigma)^2}
{1-1/\kappa}
=
\begin{cases}
0.1070 &\text{DAPS (50)},\\
0.2284 &\text{SITCOM (25)}.
\end{cases}
\]
The associated r are roughly 47,300 and 101,000 dimensions; these are not measured Jacobian ranks. The image dimension is d=196608, so \(r/m\le d/m=4/9\), making these illustrations numerically possible.

**Crucial limitations:** (i) empirical medians pooled over images/runs do not equal the expectation in Proposition 2; (ii) the methods are not exact stationary Gaussian samplers; (iii) practical denoisers and nonconvex basins introduce strong mode dependence; (iv) SITCOM's unit SSE step means its effective κ cannot be identified from its Langevin factor alone; (v) the raw magnitude FFT is non-smooth at Fourier zeros. The observed below-sigma residuals support *noise fitting*, not a precise inverse-Jacobian rank or confirmed posterior temperature.

**Pre-frozen qualitative XP-K mechanism forecast:** reducing the data term to the noise-calibrated precision can increase the GOOD-run residual toward the physical sigma noise floor (DAPS, SITCOM), but the shift magnitude and other class rates are not determined by the toy. Whether it improves PSNR or changes C frequency is **not signed by this theory** without additional assumptions about score/prior fidelity.

## 3. Why class residual medians can nearly coincide but within-image rankings be decisive

For every run \(\hat x\), let \(d_{\hat x}=A(\hat x)-A(x_\star)\). Define
\[
D_{\hat x}=\|d_{\hat x}\|^2/\sigma^2,\qquad
C_{\hat x}=\langle y-A(x_\star),\,d_{\hat x}\rangle/\sigma^2.
\]
These are the planner's preregistered A5.5 quantities. **Algebraic identity, without independence assumptions:**
\[
\frac{\|y-A(\hat x)\|^2-\|y-A(x_\star)\|^2}{\sigma^2}
=D_{\hat x}-2C_{\hat x}.
\]
Thus the normalized RMS residual difference for a candidate versus x★ is \(\sigma^2(D-2C)/m\) in **squared** residual units. For two runs G and C on the same image with the same y, the exact paired ordering is determined by
\[
\boxed{
\Delta_{C,G}=(D_C-D_G)-2(C_C-C_G);
\qquad\rho(C)<\rho(G)\iff\Delta_{C,G}<0.
}
\]
The large observation-noise term \(\|y-A(x_\star)\|^2\) cancels, so comparing *marginal* across-image residual medians is a much weaker diagnostic than the **paired, within-image** score difference.

**Proposition 3 (common-noise cancellation in a local measurement subspace).**
If all compared candidate measurement deviations \(d_j/\sigma\) lie in a fixed r-dimensional subspace V of \(\mathbb R^m\), with orthogonal projection P, then
\[
\|y-(a+d_j)\|^2/\sigma^2
=
\|(I-P)Z\|^2+\|PZ-d_j/\sigma\|^2.
\]
The first term is exactly shared across *all* compared candidates on the same observation and cancels from every rank comparison. It has mean m-r, which can dominate the absolute normalized residual when r≪m. Therefore nearly identical \(\rho\)-medians can coexist with highly class-ordered paired ranks.

**Proof:** Orthogonally split \(Z=PZ+(I-P)Z\); subtract \(d_j/\sigma\in V\); invoke Pythagoras. No Gaussian or independence assumption is needed for the identity, though Gaussianity determines the common term's chi-square distribution.

For *fixed, noise-independent* candidates, the pairwise rank probability specializes to \(\Phi(\delta/(2\sigma))\) when one candidate equals x★ and the other has \(\|d\|=\delta\). **For actual solver outputs \(\hat x(y)\)**, d and Z are dependent, so neither fixed-pair normality nor the above chi-square split automatically determines the class rank rates. That is why A5.5 is the direct diagnostic.

## 4. Relation to orientation and class-order leak (testable hypothesis, not a theorem)

For any fixed candidate x and its rotated member gx, **the realized residuals are equal exactly**: \(\rho(gx)=\rho(x)\) for every noise realization. Thus the observed DAPS/FAST preference for separately generated GOOD over separately generated globally ORBIT candidates (and SITCOM's near-equal rank rates) cannot come from the Fourier operator directly. It must depend on how the solver generates those candidate states, their within-mode non-orbit perturbations, or which conditional trajectories the learned prior favors.

In a local linearization, \(J_{gx}=J_xg^{-1}\); these Jacobians have the same image/measurement subspace under exact group action. If an orientation-independent Gaussian prior and equivariant kernel are used, within-orbit residual score distributions coincide. A learned orientation-biased prior can make their effective shrinkage/effective perturbation distributions differ. The residual rank leak is therefore **indirect prior/solver information**, not measurement information about the identity of an exact orbit pair.

**No universal monotone link between κ and G-versus-O rank preference is proven.** Increasing κ could wash out differences in output residuals if it drives all states toward the same measurement manifold, but algorithmic basin transitions, denoiser biases and terminal maps can reverse that intuition. This caveat is important for any TH2 mechanism JSON.

## 5. What to freeze as TH2 mechanism forecasts without pretending to identify everything

A useful supplemental preregistration would record:

1. The **flat active-subspace reference** formula for \(\mathbb E[\rho^2]/\sigma^2\) at κ=50,25 and at κ=1 with clearly stated rank-r sensitivity, including residual concentration variance.
2. **Signed qualitative prediction** only for DAPS/SITCOM GOOD-run ρ under a parameter-only move toward κ=1: tendency toward σ if active dimension/prior response is held fixed; flag SITCOM additional Euler correction and full solver adaptation as confounders.
3. The exact \(\Delta_{C,G}=D_C-D_G-2(C_C-C_G)\) paired score criterion to examine A5.5.
4. **Unidentified/abstain** for changes in CONSISTENT share, PSNR, orientation leak, and FAST ε/noise unless one can derive a parameter-faithful sign. Pre-registration can specify null/unidentified rather than fabricate precise predictions.

XP-K and XP-G outcomes remain embargoed. No new native solver experiment is recommended here beyond those already preregistered by the planner.

## 6. Prior art and mathematical classification

The posterior covariance/traces and Gaussian chi-square projection laws are textbook Bayesian linear inverse problems; the noise-adaptive residual optimism is closely related to Stein's SURE, Morozov's discrepancy principle and inverse-problem regularization. Track R's potential addition is **connecting the correct sample residual formula, three known implementation weights, the shared-noise paired-rank identity, and the empirically labeled failure categories**, not claiming the matrix identities are original. Distinguish conditional toy predictions from validated native-solver conclusions throughout.
