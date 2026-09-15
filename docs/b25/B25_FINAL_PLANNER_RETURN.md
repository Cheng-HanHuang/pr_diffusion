# B25 final planner return

## Executor status

B25 is complete and returns control to the scientific planner. This return is archival/reporting only and does not authorize another experiment.

The single final recommendation is:

**a precisely defined NP conditional-selection correction**

B24 remains closed under `STOP_B24_METHOD_REFINEMENT`.

## PR / branch identity

- repository: `Cheng-HanHuang/pr_diffusion`
- draft PR: `#39 — [B25] noise-selection bias and phase-retrieval failure mechanisms`
- branch: `codex/b25-noise-selection-mechanisms`
- base: `codex/b24-bestof4-failure-sweep`
- immutable B24 start: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- pre-run B25 scientific commit: `32453db6445acec4fc19a4a928142a412d67f1ae`
- PR remains draft, open, and unmerged

## Completed execution

PAC run:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

The worker completed PASS. Scientific work was CPU-only with `CUDA_VISIBLE_DEVICES=""` and stayed within the frozen resource envelope:

- aggregate scientific wall time: `159.6690 s`;
- maximum observed RSS: `0.5504 GiB`;
- GPU work: false;
- pretrained-model inference: false;
- new FFHQ measurement generation: false;
- new FFHQ reconstruction generation: false;
- confirmation305 payload access: false.

## Experiment 1 decision

The frozen iid Gaussian proposal study passed all analytic order-statistic checks. All four `K>1` hard-selection fits supported the prospectively frozen `h^0.5` band, while all four likelihood-weighted fits supported the `h^1` band.

Observed exponents:

- linear `K=4`: hard `0.4916`, weighted `1.0325`;
- linear `K=8`: hard `0.4932`, weighted `1.0600`;
- quadratic `K=4`: hard `0.4752`, weighted `0.9833`;
- quadratic `K=8`: hard `0.4690`, weighted `0.9725`.

Interpretation: hard best-of-K selection can introduce a leading-order conditional-selection displacement unlike likelihood weighting. This remains a controlled mechanism result, not a direct theorem for historical NP because native NP includes incumbent reuse.

## Experiment 2 decision

The exact finite-support reference checks all passed. Multimodal cases decisively separate exact-intermediate likelihood weighting from both hard selection and denoised-point likelihood weighting.

At `K=8`:

- ambiguity unequal TV: exact-weighted `0.00475`, hard `0.26242`, denoised-point weighted `0.24250`;
- near ambiguity equal TV: exact-weighted `0.03401`, hard `0.22087`, denoised-point weighted `0.24467`;
- near ambiguity unequal TV: exact-weighted `0.00567`, hard `0.24983`, denoised-point weighted `0.22762`.

Therefore the correction must target the intermediate observation likelihood `p(y|z_t)` rather than simply soften the historical denoised-point score.

## Experiment 3 decision

The offline eight-mask symmetry oracle is valid under the tested channelwise Fourier-magnitude operator and reveals several isolated orientation failures. It is not the common B24 failure mechanism.

Across all DEV80:

- DAPS: `1` Good25 symmetry recovery;
- SITCOM: `3` Good25 symmetry recoveries;
- NP4: `0` Good25 symmetry recoveries.

On the frozen 10-image shared DAPS4/SITCOM4 failure subset:

- DAPS: `0` recoveries;
- SITCOM: `0` recoveries;
- NP4: `0` recoveries.

Symmetry-handling investigation is therefore not selected as the B25 recommendation.

## Experiment 4 decision

The observation/preprocessing discrepancy is source-verified and numerically nontrivial.

DAPS and SITCOM both predict nonnegative Fourier magnitude but evaluate their main Gaussian squared residual against the supplied signed noisy measurement. DAPS's local B20 low-frequency amplitude projection separately uses `abs(measurement)` only in that auxiliary projection path.

Historical B24 DEV80 NP instead verifies the locked raw measurement and then uses:

`measurement_np = measurement_raw.clamp_min(0.0)`.

All `80/80` DEV observations contain negative values. The raw-to-clamped relative L2 change has median `0.08223` and range `0.05596` to `0.14793`.

This establishes a real observation-model mismatch but does not prove clipping alone caused any historical failure.

## Precisely defined returned correction

The planner should treat the following as the B25-supported prospective NP correction specification:

1. Keep the locked signed observation `y_raw` for Gaussian observation likelihood evaluation.
2. Use `y_plus = clamp_min(y_raw,0)` only inside amplitude projection/proximal operations that require a feasible nonnegative magnitude target.
3. Replace deterministic hard argmin conditioning with normalized stochastic resampling.
4. Use exchangeable proposal draws from one proposal kernel whenever possible. If incumbent reuse is retained, include candidate-specific proposal-density correction rather than assuming iid proposals.
5. Weight candidate `z_t^(j)` by an estimator of the intermediate likelihood `p(y_raw|z_t^(j))`, together with any required target/proposal importance factor.
6. Do not approximate that likelihood solely by evaluating the measurement model at one denoised posterior-mean point; the estimator must preserve enough multimodal uncertainty to avoid the mode-weight failures demonstrated in Experiment 2.
7. Propagate the resampled candidate's state/noise lineage consistently.

This is a correction target for planner review. It is not authorization to implement or run it.

## Evidence capsule

Archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`

Verified SHA-256:

`5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`

The archive has 26 safe members. Its `SHA256SUMS.txt` contains 25 run-file digests and all 25 were independently verified after archive creation. The added post-run source audit has SHA-256:

`1d4ebc132d7e4b92e7a9f7be14bec20ef6715a438b18a34096836fb58c6d7d0c`

## Final executor return

B25 has answered the authorized mechanism questions sufficiently to return a concrete NP correction target. No confirmation execution, GPU work, new FFHQ work, or further B24 refinement is required or authorized. Return control to the planner with PR #39 left draft/open and the recommendation above frozen.
