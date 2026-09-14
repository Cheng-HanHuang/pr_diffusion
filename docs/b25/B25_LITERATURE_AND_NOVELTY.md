# B25 literature and novelty boundaries

This note was written before B25 scientific execution. It separates prior results from B25 derivations/diagnostics.

## DAPS

**Source/version:** Zhang et al., *Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing*, arXiv:2407.01521v3 (15 Aug 2025), https://arxiv.org/html/2407.01521v3

Established in the paper:

- DAPS targets Bayesian posterior sampling and emphasizes that nonlinear inverse problems such as phase retrieval can have multiple measurement-consistent modes.
- Its decoupled annealing step samples an approximation to `p(x0 | x_t, y)` and then re-noises, rather than imposing only a local reverse-SDE correction.
- The paper writes `p(x0|x_t,y) proportional to p(x0|x_t) p(y|x0)` and uses a Gaussian approximation to `p(x0|x_t)` in its practical method.
- The noisy likelihood at intermediate diffusion time is generally intractable; DAPS contrasts its construction with approximations such as evaluating the likelihood at a conditional mean.
- The phase-retrieval appendix explicitly discusses multiple disjoint modes with exactly the same measurement, oversampling, and 180-degree-rotation outcomes.

B25 boundary: exact finite-support posterior/reverse-transition enumeration is a diagnostic benchmark, not a claim to replace DAPS or to reproduce DAPS natively.

## SITCOM

**Source/version:** Alkhouri et al., *SITCOM: Step-wise Triple-Consistent Diffusion Sampling For Inverse Problems*, arXiv:2410.04479v2, https://arxiv.org/html/2410.04479v2

Established in the paper:

- SITCOM uses step-wise consistency constraints and an inner optimization/data-consistency procedure.
- For phase retrieval, the paper reports best-of-four independent runs and discusses the problem's multiple measurement-equivalent modes.
- The phase-retrieval appendix states that the forward model follows the DPS setup and reports SITCOM-ODE as empirically more stable on that task.
- The published ablation uses a nonzero regularization parameter for phase retrieval.

B25 boundary: best-of-four is historical protocol matching and is not B25 novelty. B25 does not alter or rerun SITCOM.

## Noise Combination Sampling (NCS)

**Source/version:** *Noise is All You Need: Solving Linear Inverse Problems by Noise Combination Sampling with Diffusion Models*, arXiv:2510.23633v2, especially Appendices G–H, https://arxiv.org/html/2510.23633v2

Established in the paper:

- NCS constructs measurement/useful-direction-dependent combinations of Gaussian noise proposals.
- Appendix G analyzes the synthesized-noise distribution. For the optimized construction, the component in a preferred direction can have a chi-distributed amplitude, nonzero mean, and modified rank-one covariance while the orthogonal component remains Gaussian.
- The appendix explicitly notes that apparently near-Gaussian high-dimensional noise can still have a structured directional deviation.
- The nonlinear-task appendix reports that NCS does **not** give consistent phase-retrieval improvements; some NCS variants are comparable to or worse than their baselines.

B25 boundary: demonstrating that selected/synthesized noise is non-Gaussian, directionally biased, or covariance-modified is not by itself novel and is not by itself evidence of an error. B25's distinct question is whether hard measurement-score selection induces a transition bias of the wrong small-step order relative to an explicitly defined conditional likelihood tilt, and whether that mechanism survives an exact-prior diagnostic.

## MGDM

**Source/version:** Janati et al., *A Mixture-Based Framework for Guiding Diffusion Models*, ICML 2025 / PMLR 267, arXiv:2502.03332, https://proceedings.mlr.press/v267/janati25a.html

Established in the paper:

- Intermediate posterior likelihoods are intractable in diffusion inverse problems.
- MGDM proposes a mixture approximation for intermediate distributions and a Gibbs-sampling-based practical method.
- The work is explicitly about representing multimodal/intermediate posterior structure more faithfully than a single point approximation.

B25 boundary: finite-support mixture likelihood enumeration, generic mixture guidance, Gibbs/importance ideas, or the observation that a conditional mean can erase multimodality are not novelty claims for B25.

## DDfire / FIRE

**Source/version:** Bendel et al., *Solving Inverse Problems using Diffusion with Iterative Colored Renoising*, arXiv:2501.17468v4 (27 Aug 2025), https://arxiv.org/html/2501.17468v4

Established in the paper:

- The method focuses on the distribution of denoiser-input error during inverse-problem inference.
- FIRE/DDfire adds designed colored Gaussian noise so the denoiser input error is approximately white, matching the AWGN corruption on which the denoiser was trained.
- The paper extends the construction to generalized-linear problems including phase retrieval and reports variance/error diagnostics.

B25 boundary: distribution mismatch at the denoiser input and deliberate noise shaping are established research topics. B25 may cite DDfire when interpreting native NP proposal-distribution changes, but cannot claim the generic insight that inference-time error distributions matter.

## Finite-sample failure analysis

**Source/version:** Burns & Fridovich-Keil, *When, why, and how do diffusion posterior samplers fail? A finite-sample lens*, arXiv:2605.30330v1 (28 May 2026), https://arxiv.org/html/2605.30330v1

Established in the paper:

- Intermediate likelihood approximations can under- or over-estimate posterior spread and propagate to terminal posterior errors.
- Reported consequences include inaccurate posterior-mode weights, premature commitment, prior-mode hallucination, and likelihood-consistent modes unsupported by the prior.
- These effects need neither a nonlinear measurement model nor a multimodal terminal posterior; a multimodal prior plus inaccurate intermediate approximation can suffice.
- The paper explicitly uses finite/discrete priors as an analytic diagnostic lens.

B25 boundary: the generic thesis that approximate intermediate likelihoods can distort mode probabilities, and finite-support/discrete-prior diagnostics themselves, are already established. B25 Experiment 2 is therefore a project-specific mechanism test contrasting hard finite-candidate selection, exact-intermediate likelihood weighting, and the denoised-point likelihood approximation under the particular selection question motivated by historical NP.

## B25's defensible contribution boundary

Before results, the potentially useful B25 contribution is **diagnostic specificity**, not a broad new posterior-sampling principle:

1. Recover the exact historical NP proposal/reuse/selection/projection semantics rather than analyze a generic noise-picker.
2. Test whether hard score-based finite proposal selection has a stronger small-step directional effect than a coherent likelihood tilt in an exactly checkable model.
3. Separate finite-candidate selection error from intermediate-likelihood approximation error using the same exact finite-support prior.
4. Quantify how much of the existing FFHQ DEV80 failure tail is explainable by the exact per-channel 180-degree Fourier-magnitude symmetry already implicit in the pinned operator.
5. Verify whether solver-specific measurement preprocessing creates a real additional discrepancy in the historical implementation.

None of these results should be described as a new theorem unless a genuinely new theorem is separately proved. The standard order-statistic density, finite-support Bayes enumeration, generic importance resampling, generic mixture guidance, and Fourier reversal ambiguity are background tools.
