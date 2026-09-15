# B26 literature and estimator boundary

B26 is a project-specific diagnosis/feasibility stage. Generic resampling, likelihood averaging, mixture guidance, and particle methods are prior art and are not novelty claims.

## Twisted Diffusion Sampler (TDS)

Primary source: Wu, Trippe, Naesseth, Blei, and Cunningham, *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models*, NeurIPS 2023 / arXiv:2306.17775.

Relevant established facts:

- TDS formulates conditional diffusion sampling as sequential Monte Carlo with weighted particles.
- It uses heuristic/approximate twisting information to improve proposals while retaining an explicit importance-weight correction.
- The SMC target is defined jointly over the diffusion trajectory; proposal and weighting functions together determine the intermediate targets.
- Asymptotic exactness follows from the SMC construction and its weights, not merely from particle exchangeability or multinomial resampling.

B26 boundary: a future NP method cannot claim posterior correctness because it replaces hard argmin by categorical likelihood weighting. Historical NP's re-noising/denoising candidates define a history-dependent proposal and incumbent reuse makes candidate laws nonexchangeable. The desired target/proposal ratio must be derived or the method must be labeled heuristic.

## MGDM

Primary source: Janati et al., *A Mixture-Based Framework for Guiding Diffusion Models*, ICML 2025 / PMLR 267 / arXiv:2502.03332.

Relevant established facts:

- Intermediate likelihoods/posteriors in diffusion inverse problems are generally intractable.
- MGDM introduces mixture approximations of intermediate distributions and a practical Gibbs-based inference construction.
- The work explicitly addresses multimodal intermediate structure beyond a single denoised-point approximation.

B26 boundary: estimating an intermediate likelihood by integrating over multiple plausible clean states, representing a conditional as a mixture, or using sampling/resampling to preserve multimodality is not novel by itself.

## B25 boundaries carried forward

B25 already documented overlap with DAPS, SITCOM, NCS, DDfire/FIRE, MGDM, and finite-sample posterior-sampler failure analyses. In particular, inaccurate intermediate likelihoods and finite/discrete-prior diagnostics are established topics. B26 does not claim a new theorem from its finite-support experiment.

## Defensible B26 contribution boundary

The potentially useful B26 evidence is narrower:

1. an exact native-path ablation of the historical NP observation clamp on the already frozen DEV80 measurements while proposal generation, hard selection, projection, roots, and compute remain fixed;
2. a replay-verified separation of raw signed observation likelihood/scoring from the nonnegative magnitude target required by amplitude projection;
3. a fixed-grid measurement of inner Monte Carlo likelihood-estimation error versus outer finite-proposal error under the exact B25 finite-support model, with varying truths/observations and both posterior-fidelity and reconstruction-risk outcomes;
4. an explicit cost/proposal-target audit showing what a learned-chain continuation estimator would actually estimate and how expensive it would be in native NP.

Any future weighted-NP method requires a separate authorization and should be compared against TDS/MGDM at the level of target distribution, proposal correction, mixture/particle structure, and compute—not merely at the level of implementation vocabulary.
