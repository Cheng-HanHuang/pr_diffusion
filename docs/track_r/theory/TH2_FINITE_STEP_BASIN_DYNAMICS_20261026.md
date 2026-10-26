# TH2: finite-step basin dynamics and the role of inner stochasticity

**Date:** 2026-10-26. **Status:** PROVED affine recurrences; synthetic NUMERICAL checks; application to actual DAPS, SITCOM-ODE, FAST-DIPS is CONDITIONAL. No PAC, XP, confirmation305, GT images, or solver changes.

## Scientific aim and prior knowledge

On the exact zero-flip Fourier pair z=(1,5,6)/7, v=(2,7,3)/7, separated by Euclidean distance d=sqrt(14)/7, model motion along the one-dimensional signed direction joining the candidates. The real Fourier operator remains nonconvex; the following quadratic well is only a *local idealization*, not an exact reduction of the original phase-retrieval solver.

Ornstein--Uhlenbeck Langevin recurrences, proximal maps, and the difference between optimization and sampling are classical. See Ehrhardt--Kuger--Schönlieb, *Proximal Langevin Sampling with Inexact Proximal Mapping*, SIAM Journal on Imaging Sciences 2024 (DOI 10.1137/23M1593565); Laumont--Dong--Andersen, *Sampling strategies in Bayesian inversion: A study of RTO and Langevin methods*, Inverse Problems and Imaging 2025 (DOI 10.3934/ipi.2024046). The contribution here is an explicit limitation-aware bridge to the frozen solver comparison, not a new OU theorem.

## Proposition 1: deterministic stable local refinement preserves the initial basin

Let a local energy in the positive basin be U(t)=kappa*(t-b)^2/2 with b>0, kappa>0, and t0>0. Explicit gradient descent with 0<eta*kappa<=1 gives

    t_(j+1)=(1-eta*kappa)*t_j+eta*kappa*b

and therefore t_j=b+(1-eta*kappa)^j*(t0-b)>0. A proximal update with lambda>0 yields

    t_(j+1)=(t_j+lambda*kappa*b)/(1+lambda*kappa)>0.

**Proof:** Each map is a convex combination of two positive quantities. Hence, in this locally quadratic and step-size-stable toy, deterministic correction alone cannot generate the missing opposite basin. This statement does not apply to arbitrary nonconvex optimization or large/unstable steps.

## Proposition 2: exact finite-step Langevin endpoint law

Consider the globally affine recursion

    T_(j+1)=b+a*(T_j-b)+sqrt(2*eta*Temp)*xi_j,

where a=1-eta*kappa in (0,1), Temp>0, and xi_j are iid N(0,1). Starting from deterministic T0=t0,

    T_J ~ N(mu_J, V_J),
    mu_J=b+a^J*(t0-b),
    V_J=2*eta*Temp * (1-a^(2J))/(1-a^2).

For mu_J>0, the probability of a negative **endpoint sign** is exactly

    P(T_J<0)=Phi(-mu_J/sqrt(V_J)).

**Proof:** Unroll the affine recurrence; the noise is a weighted sum of independent Gaussians. The endpoint event is not a first-passage probability, and the globally affine extension does not switch its restoring force after crossing the chosen basin boundary. It cannot be used directly to compute transitions between the true two stable modes of a double-well potential.

## Proposition 3: inner versus outer noise — both a distinction and an equivalence

The deterministic J-step mean is mu_J. A single outer re-noising step produces T_outer=mu_J+sigma_out*xi, with sign-crossing probability Phi(-mu_J/sigma_out).

If sigma_out^2 equals only one inner Langevin innovation variance, 2*eta*Temp, then V_J>=sigma_out^2 for J>=1, with strict inequality for J>1 and a>0. Thus at the same positive mean, inner-noised dynamics produce more negative endpoints.

**Negative result:** If sigma_out^2 is instead matched to V_J, then T_outer and T_J have *identical endpoint laws* in this affine model. The temporal location of Gaussian noise, by itself, cannot distinguish endpoint occupancy here at matched drift and covariance. Genuine mechanistic differentiation requires two-well/nonlinear forces, noncommuting step maps, constraint projections, or unequal noise budgets. This is an important rejection criterion for a potentially uninformative TH2 toy.

## Executed synthetic numbers

Using the exact zero-flip distance, b=d/2≈0.267261, t0=b, kappa=2, eta=0.1, Temp=0.2, and J=8, we obtain a=0.8 and V_J≈0.107984. The analytic Langevin endpoint crossing probability is 0.208020; seeded Monte Carlo with 150,000 paths gives 0.209453.

With one outer innovation of variance 0.04, the analytic crossing probability is 0.090725 and seeded Monte Carlo gives 0.091140. When the outer variance is instead set equal to V_J, the endpoint laws coincide exactly. These are synthetic mechanism checks, **not** empirical success or failure rates of the three production solvers.

## Code-grounded, conditional solver mapping

- DAPS uses inner MCMC/Langevin innovations as well as forward re-noising; the native pinned source was inspected in the solver equivariance audit. Its nonlinear data term and denoiser tether are not identical to the affine toy.
- The pinned SITCOM-ODE benchmark ALSO uses inner SGD/Langevin stochasticity. The separate local SITCOM Python code has deterministic Adam/SGD steps, but it should not be conflated with the native pinned phase-retrieval experiment. It is scientifically inaccurate to claim the actual SITCOM-ODE experiments are pure deterministic basin selection.
- FAST-DIPS pixel ADMM x/projection/dual correction is deterministic for a fixed denoiser anchor and injects outer Gaussian re-noising, with a terminal denoiser readout. These are source-level facts, not yet an on-PAC wrapper audit.

**Decision:** pursue a genuine two-well finite-step kernel or a native-code-faithful restricted dynamical model in TH2 next. Do not claim the elementary OU/proximal comparison as new or deduce a solver-specific ranking from it. Preserve TH1 as the CVPR-critical theory.

**Source details:** see TH1_SOLVER_EQUIVARIANCE_CHECKS.md and TH2_FOURIER_FIBER_FUNCTIONALS_20261026.md on the same branch.
