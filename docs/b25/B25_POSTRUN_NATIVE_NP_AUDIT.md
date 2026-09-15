# B25 post-run native NP audit

## Scope

This document records the final source-level interpretation after the completed CPU-only B25 run and the planner's reporting review. It does not modify historical B24 code or outputs and does not authorize a new reconstruction run.

Pre-run scientific commit:

`32453db6445acec4fc19a4a928142a412d67f1ae`

PAC run:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

B25 supports a conditional-inference hypothesis worth investigating. It does **not** establish a better FFHQ reconstruction method.

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

This establishes a genuine and material observation/preprocessing change. It does **not** establish that clipping caused a historical failure or that using the raw signed observation will improve reconstruction quality. That requires an isolated ablation.

## Mechanism results relevant to a future correction

Experiment 1 passed all analytic engineering checks. For `K>1`, every prospectively frozen hard-min fit landed in the supportive `h^0.5` band, while every finite-likelihood-weighted fit landed in the supportive `h^1` band. This supports the narrower statement that hard best-of-K conditioning changes the controlled small-step behavior relative to likelihood weighting. Matching `h` scaling does **not** establish correct posterior dynamics.

Experiment 2 must be interpreted with posterior fidelity separated from one-truth reconstruction accuracy. At `K=8`:

| Prior family | hard TV | exact-intermediate weighted TV | denoised-point weighted TV | hard truth-MSE | exact-intermediate weighted truth-MSE | denoised-point weighted truth-MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| distinguishable equal | **0.00000** | 0.06213 | 0.00004 | **0.000000** | 0.004128 | 0.000002 |
| exact ambiguity, unequal weights | 0.26242* | **0.00475** | 0.24250 | 0.049744* | 0.024272 | **0.000728** |
| near ambiguity, equal weights | 0.22087 | **0.03401** | 0.24467 | **0.000000** | 0.024769 | 0.045241 |
| near ambiguity, unequal weights | 0.24983 | **0.00567** | 0.22762 | **0.000000** | 0.024829 | 0.002158 |

The near-ambiguity cases support the conditional-inference hypothesis: exact-intermediate likelihood weighting is much closer to the exact posterior probabilities. They do **not** show better reconstruction of the one frozen truth; hard selection has lower truth-MSE in both near-ambiguity rows. The distinguishable family prevents a universal claim that weighting wins.

`*` The exact-ambiguity hard-selection result is excluded from evidence for an intrinsic hard-selection defect until tie handling is studied. The theoretically identical likelihoods differ at floating-point precision, and hard `argmax`/`argmin` can convert that numerical difference into a systematic preference.

Thus replacing hard argmin by a softmax of the existing denoised-point score is not an adequate general correction, but B25 also does not supply an implementable pretrained-model estimator of the required intermediate likelihood `p(y|z_t)`.

## Precisely defined investigation returned to the planner

The recommended next scientific direction remains **a precisely defined NP conditional-selection correction**, interpreted as one focused prospective investigation rather than an established method.

The unresolved question is:

> **Can we estimate the intermediate conditional likelihood well enough to improve reconstruction at a fixed computational budget?**

Before a new FFHQ experiment, the planner must specify:

1. an implementable estimator of `p(y_raw | z_t)` for the pretrained-model setting;
2. the proposal distribution generating the candidate states;
3. the finite-proposal approximation and normalized stochastic resampling rule;
4. lineage handling for each resampled state/noise pair;
5. proposal-density correction if incumbent reuse or otherwise nonexchangeable proposals remain;
6. full compute accounting, including denoiser, likelihood-estimator/selector, projection, FFT/custom-operator, and branch overhead.

The observation-model and selection-rule mechanisms must be isolated against frozen NP with at least:

- frozen historical NP;
- raw-measurement correction only;
- selection-rule/intermediate-likelihood correction only;
- combined raw-measurement + selection correction.

Fresh2 remains an important efficiency comparator. Without this ablation, a gain could be attributed to the wrong mechanism.

For the prospective correction itself, retain these design invariants unless the planner explicitly revises them:

1. Preserve the raw signed locked observation `y_raw` for the Gaussian observation likelihood.
2. Define `y_plus = clamp_min(y_raw, 0)` only for amplitude projection/proximal operations that mathematically require a nonnegative magnitude target; do not silently substitute `y_plus` into the observation likelihood.
3. Do not use deterministic hard `argmin` as a posterior-conditioning substitute.
4. Prefer exchangeable candidates from one proposal kernel; otherwise include the required candidate-specific proposal correction.
5. Weight candidate `z_t^(j)` using the specified estimator of the intermediate likelihood, not merely the measurement likelihood at one denoised posterior-mean point.
6. Propagate the resampled candidate's state/noise lineage consistently.

This is a prospective investigation specification, not authorization to run it. B24 remains closed, confirmation remains locked, and no new GPU stage is authorized by the B25 review.
