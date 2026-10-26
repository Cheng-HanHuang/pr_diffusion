# TH2 early feasibility study: sampling, mode choice, and anchor projection on one exact Fourier fiber

**2026-10-26 | CPU-only synthetic work, proof plus numerics.** No PAC, GT images, XP results, or protected data. TH2 is exploratory and does **not** alter the Oct 31 prediction model or assert the native solvers reduce to these toy rules.

## 1. Why this toy is worth doing even though its ingredients are known

The zero-flip ambiguity is classical (Beinert--Plonka, *Ambiguities in one-dimensional discrete phase retrieval from Fourier magnitudes*). Sampling, MAP optimization, and proximal projections are likewise established. The underexplored question for Track R is **how different conditional functionals turn an indistinguishable Fourier fiber into different occupancy probabilities**, and whether these distinctions help explain the actual solver-specific failure taxonomy.

Let `z=(1,5,6)/7` and `v=(2,7,3)/7`. These are nonnegative signals, not reversal pairs, with equal Fourier magnitudes for all Fourier frequencies. Their Euclidean distance is exactly `d=sqrt(14)/7`. Work on the restricted two-candidate fiber F={z,v}. This is not claimed to be the entire fiber of the original inverse problem.

## 2. Three explicitly different idealized choices

**A. Exact conditional posterior draw.** Assign prior conditional mass `w` to z and `1-w` to v. Since the likelihoods are equal, the posterior conditional masses are exactly the same. An exact posterior sample chooses z with probability w. For w=0.8, that probability is 0.8. This represents an *ideal exact conditional sampler*, not practical DAPS.

**B. Discrete conditional MAP.** Under 0--1 decision loss on F, a MAP rule chooses z if w>1/2 and v if w<1/2. For w=0.8, it chooses z with probability one (assuming no exogenous tie randomization). Thus MAP and sampling disagree even though both respect the same correct posterior. With continuous mixtures, MAP additionally depends on peak density/width, not just integrated mode masses. This is a toy optimization comparator, **not** a theorem that SITCOM is posterior MAP.

**C. Nearest-anchor feasible projection.** Suppose a denoiser produces an anchor R. Define the feasible projection `Pi_F(R)=argmin_{a in F} ||R-a||^2`, using any fixed rule at ties (ties have probability zero for continuous anchors). Let

    R = Z + sigma * epsilon  with probability lambda,
    R = V + sigma * epsilon  with probability 1-lambda,

where `epsilon ~ N(0,I_3)`, `sigma>0`, and Z=z, V=v. Then, writing `H=Phi(d/(2 sigma))`,

    P(Pi_F(R)=z) = lambda * H + (1-lambda) * (1-H).

**Proof:** If R=z+sigma epsilon, choosing z is equivalent to

    ||R-z||^2 < ||R-v||^2
    <=> 2 sigma <epsilon,v-z> < ||v-z||^2.

The normal projection has variance `||v-z||^2`, yielding probability `Phi(d/(2sigma))=H`. Centering R at v swaps the probability to `1-H`. Average with weights lambda and 1-lambda.

Thus a projection based on an anchor can favor z **despite total measurement blindness**, but that preference comes from the anchor distribution and projection geometry, not the measurement score. This isolates a mechanism in an *idealization of the feasibility-correction ingredient* of FAST-DIPS; native ADMM with soft iterations is not literally a two-point Euclidean projection.

## 3. Numerical outcome (executed)

For `sigma=0.25`, `lambda=0.30`, and `d=sqrt(14)/7=0.53452248`,

- `H=Phi(d/(2sigma))=0.8574753`;
- predicted anchor-projection success = **0.3570099**;
- Monte Carlo with 200,000 seeded anchors = **0.359085**.

Compare this to exact posterior sampling success 0.8 and conditional discrete MAP success 1 for w=0.8. Different output frequencies arise **without any difference in Fourier residual**, from three distinct selection functionals.

The numerical values are entirely synthetic and should not be quoted as expected solver performance. They demonstrate tractability of the mechanism study. The limiting cases are interpretable: as `sigma->0`, projection occupancy tends to lambda; as `sigma->infinity`, it tends to 1/2 regardless of lambda. This is a law for the two-point feasible set, not for natural-image phase retrieval.

## 4. What is worth proving or measuring next

1. Add a third non-orbit consistent decoy to the finite fiber, distinguishing generation of correct-orbit candidates from mass lost to a wrong but feasible component. This generalizes the exact two-point toy without committing to new methods.
2. Study the effect of continuous mode widths: a low-integrated-mass narrow mode can dominate MAP but not posterior samples. This is a classic phenomenon but offers a useful quantitative bridge for inner functional differences.
3. Analyze whether the SITCOM-ODE gradient/SGD-Langevin correction, DAPS MCMC, and FAST-DIPS ADMM have tractable stationary or finite-step distributions on an idealized 1D/quadratic-plus-multimodal objective.
4. Keep an explicit gap between toy and benchmark: the actual learned score, finite steps, re-noising, and the FFT's large nonconvex ambiguity set can change all quantitative predictions. No empirical family ordering is claimed before post-freeze validation.

## 5. Literature positioning and decision

A *new zero-flip theorem* is not a viable novelty claim; the one-dimensional ambiguity classification is old. A *new statement that sampling differs from MAP* is likewise not viable. The promising TH2 question is whether one can identify **distinct inner-functional basin occupation laws** and validate them against the three published inverse solvers, not simply reproduce these textbook phenomena.

**Current decision:** GO for a limited mathematical feasibility study; NO-GO for claiming a standalone theorem/paper until a nontrivial solver-faithful mechanism or a predictive empirical link is established. Keep TH1 as the CVPR-critical theory.

**Primary references:** Beinert & Plonka (2015) 1-D ambiguity survey; DAPS (CVPR 2025), SITCOM (ICML 2025), FAST-DIPS (ICLR 2026); prior Track R TH1 working proofs and native-solver code audit.
