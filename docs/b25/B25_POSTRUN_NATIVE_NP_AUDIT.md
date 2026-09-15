# B25 post-run native NP audit

## Scope

This document records the final source-level interpretation after the completed CPU-only B25 run. It does not modify historical B24 code or outputs and does not authorize a new reconstruction run.

Pre-run scientific commit:

`32453db6445acec4fc19a4a928142a412d67f1ae`

PAC run:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

## Native NP proposal-selection semantics

Historical NP is not the iid toy process used in B25 Experiment 1. In the native branching path, candidate index 0 may reuse the previously selected noise when `eps_prev` exists and `K>1`, while the remaining candidates are freshly sampled. Candidates are denoised, ranked by the frozen measurement-derived score, and the hard minimum is propagated together with its state/noise lineage.

Therefore the exact iid order-statistic formula used in Experiment 1 is a mechanism diagnostic, not a theorem about native NP. Any correction that preserves incumbent reuse must account for candidate-specific proposal distributions. The cleaner prospective implementation is to use exchangeable candidates from the same proposal kernel; otherwise an explicit target/proposal importance ratio is required.

## Observation-model audit

The accepted local DAPS and SITCOM phase-retrieval operators both map a candidate image to a nonnegative Fourier magnitude. Their main Gaussian observation losses compare that prediction directly with the supplied measurement by a squared residual:

`||A(x) - y||_2^2`.

The locked B24 measurement is supplied in signed noisy form. SITCOM records `measurement_preprocessing='none'`. DAPS's main likelihood likewise receives the supplied signed measurement. DAPS also has a separate local B20 low-frequency amplitude-projection helper that applies `abs(measurement)`, but that transformation is confined to the auxiliary projection path and is not the main DAPS likelihood.

Historical B24 DEV80 NP instead verifies the locked raw measurement and then executes:

`measurement_np = measurement_raw.clamp_min(0.0)`.

B25 measured the numerical consequence over the 80 development observations:

- all `80/80` contain negative stored amplitudes;
- total negative elements: `12,848,209`;
- raw-to-clamped relative L2 change: mean `0.08511295`, median `0.08222852`, minimum `0.05595935`, maximum `0.14793063`.

This establishes a genuine observation/preprocessing mismatch. It does **not** by itself prove that clipping caused any particular historical failure.

## Mechanism results relevant to correction

Experiment 1 passed all analytic engineering checks. For `K>1`, every prospectively frozen hard-min fit landed in the supportive `h^0.5` band, while every finite-likelihood-weighted fit landed in the supportive `h^1` band. The controlled evidence therefore supports the claim that hard best-of-K conditioning can introduce a leading-order selection displacement that differs from likelihood weighting.

Experiment 2 gives the stronger multimodal warning. At `K=8`:

- `ambiguity_unequal`: exact-intermediate weighted TV `0.00475`, hard TV `0.26242`, denoised-point weighted TV `0.24250`;
- `near_ambiguity_equal`: exact-intermediate weighted TV `0.03401`, hard TV `0.22087`, denoised-point weighted TV `0.24467`;
- `near_ambiguity_unequal`: exact-intermediate weighted TV `0.00567`, hard TV `0.24983`, denoised-point weighted TV `0.22762`.

Thus replacing hard argmin by a softmax of the existing denoised-point score is not an adequate correction in general. The relevant conditional quantity is the intermediate likelihood `p(y | z_t)`, not merely `p(y | E[X|z_t])`.

## Precisely defined correction returned to the planner

The recommended next scientific mechanism is **a precisely defined NP conditional-selection correction** with the following invariants:

1. Preserve the raw signed locked observation `y_raw` for the Gaussian observation likelihood used to condition/rank proposals.
2. Define `y_plus = clamp_min(y_raw, 0)` only for an amplitude projection/proximal operation that mathematically requires a nonnegative magnitude target. Do not silently substitute `y_plus` into the observation likelihood.
3. Do not use deterministic hard `argmin` as the conditional-selection rule.
4. Prefer exchangeable candidates from one proposal kernel. If native incumbent reuse is retained, compute candidate-specific proposal corrections rather than treating the proposals as iid.
5. For candidate `z_t^(j)`, use normalized stochastic resampling weights proportional to an estimator of the intermediate observation likelihood `p(y_raw | z_t^(j))`, with any required target/proposal correction included.
6. Do not define that estimator solely as the measurement likelihood at one denoised posterior-mean point. The estimator must preserve multimodal uncertainty sufficiently to avoid the mode-weight pathologies demonstrated by Experiment 2.
7. When a candidate is resampled, propagate its state/noise lineage consistently.

This is a prospective correction specification, not authorization to run it. B24 remains closed under `STOP_B24_METHOD_REFINEMENT`.
