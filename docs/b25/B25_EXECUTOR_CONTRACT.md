# B25 executor contract

## Identity and authorization

- Project: `Cheng-HanHuang/pr_diffusion`
- Stage: `B25 — noise-selection bias and phase-retrieval failure mechanisms`
- Immutable starting commit: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- Branch: `codex/b25-noise-selection-mechanisms`
- Draft PR base: `codex/b24-bestof4-failure-sweep`
- Signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- B24 remains closed under `STOP_B24_METHOD_REFINEMENT`.

B25.0 and B25.1 are authorized. The scientific work is CPU-only. No pretrained-model inference, GPU use, new FFHQ measurement generation, new FFHQ reconstruction, confirmation305 payload access, C1-only exposure, or B24 selector/schedule tuning is permitted.

## Resource envelope

Every scientific subprocess must run with `CUDA_VISIBLE_DEVICES=""`. Use no more than four CPU workers, 16 GiB RAM, and four aggregate wall-clock hours. Use an explicitly inventoried interpreter. Frozen parent environments are read-only.

Long runs must use the committed B25 launcher and write log, PID, status, and resource accounting under the B25 output root. Failure must leave the shell open and the partial evidence intact; B25 launchers therefore do not use `set -e`, `set -u`, `pipefail`, shell replacement, `exit`, or broad process-killing commands.

## Pre-run freeze

Scientific execution is allowed only when all of the following are true:

1. The B25 PAC worktree exists and is on `codex/b25-noise-selection-mechanisms`.
2. Its HEAD equals the pushed pre-run commit recorded in `PRE_RUN_IDENTITY.json`.
3. The worktree is clean.
4. The immutable B24 remote tip still equals `ed162c2f97430804fddb5d9a0bfec7abde201ca0`.
5. The B25 spec is exactly `configs/b25/b25_cpu_spec.json` from that pre-run commit.
6. DEV80 and confirmation305 IDs are proven disjoint from the frozen role registry before any DEV payload is opened.
7. The accepted DEV80 manifest/closeout identities and targeted external-source provenance checks pass.

Numerical choices are not to be changed after looking at B25 scientific outputs. A correction for an implementation error must preserve the failed attempt and document the correction.

## Experiment 1 — selection distribution

Use the independent Gaussian proposal model in the frozen config. For iid proposals `e_1,...,e_K ~ q` and continuous score `S`, selecting the minimum gives

`q_pick(e) = K q(e) [1-F_S(S(e))]^(K-1)`.

For the linear energy, the score-direction projection is the minimum of `K` standard normals, giving an analytic order-statistic reference. For the small-step transition

`x' = x + b h + sqrt(h) e`,

hard minimum selection with `K>1` is predicted to create a leading `O(sqrt(h))` directional displacement, while likelihood weighting with `L(x')` is predicted to create `O(h)` displacement. This is a hypothesis test about scaling and distributional change, not a claim that non-Gaussian selected noise is intrinsically defective.

The quadratic energy is also analytically checkable because multiplying the Gaussian proposal density by `exp(-||x'-c||^2/2)` yields a Gaussian target. Report means, covariance summaries, directional projections, fixed-bin projection distributions, uncertainty, and fitted small-step exponents.

## Experiment 2 — exact finite-support prior

Use only synthetic nonnegative `6x6` grayscale templates. The measurement is the flattened orthonormal 2D Fourier magnitude with additive Gaussian measurement noise. The forward noising process is the coherent VP Markov chain induced by the frozen clean-to-noise alpha schedule.

The terminal posterior is enumerated exactly:

`P(i|y) proportional to pi_i p(y|x_i)`.

At intermediate state `z_t`, compute

`P(i|z_t)` by finite-support Bayes and
`p(y|z_t) = sum_i p(y|x_i) P(i|z_t)`.

Do **not** replace the latter by the measurement likelihood at `E[X|z_t]` except in the explicitly labeled denoised-point approximation arm.

The exact reverse transition is a finite mixture over template index and the Gaussian VP bridge. The unconditional reverse kernel is the proposal for all finite-candidate methods, so the exact-intermediate likelihood is the appropriate importance factor. Finite-candidate weighted resampling is an approximation, not an exact conditional transition.

Frozen families cover distinguishable templates, an exact reversal ambiguity, and near ambiguity under positive noise. Exact tests must cover posterior normalization, independent Bayes enumeration, the large-noise prior limit, exact ambiguity/prior-odds preservation, and nonnegative bridge variances.

## Experiment 3 — DEV80 symmetry diagnostic

Before any tensor is opened, read only the frozen role registry and IDs, prove that the 80 DEV IDs are disjoint from the 305 confirmation IDs, and create an allowlist. Then follow only manifest-recorded DEV paths.

For each DEV image, inspect at most four existing DAPS terminals, four existing SITCOM terminals, and four `NP4_INDEPENDENT` terminals. Never regenerate missing candidates. Fresh2 is a pointer to an existing DAPS candidate only and is not counted as a new terminal.

Use the common B24 saved/quantized RGB8 raw-orientation representation. The tested symmetry family consists of 8 transformations: each RGB channel is independently either unchanged or spatially reversed by 180 degrees. First verify the transformation algebra on synthetic inputs under the exact channelwise padded centered-FFT magnitude implementation and the frozen scale-aware tolerances.

For each terminal report original PSNR/Good25, original measurement residual, transformed measurement discrepancy, GT-assisted best symmetry PSNR, and maximizing mask. Aggregate all DEV80 rows, frozen screening strata, and the corrected B24 shared-failure subset defined by fresh DAPS4 and SITCOM4 oracle records. The symmetry oracle is an offline upper bound, not a deployable selector or primary metric.

## Experiment 4 — likelihood/preprocessing audit

Audit source and existing artifacts only. Record amplitude/intensity choice, input range, padding, FFT normalization, noise assumption, clipping/absolute-value preprocessing, projection-only transformations, residual normalization, and tolerance choices for DAPS, SITCOM, and NP.

One fact is already established from the frozen repository: the B24 DEV80 NP wrapper verifies the raw measurement hash and then uses `measurement_raw.clamp_min(0.0)` in memory. The SITCOM B22/B24 wrapper records `measurement_preprocessing='none'` and passes the hashed raw measurement to the official operator. Whether DAPS behaves differently must be verified against the pinned accepted local source; do not assume a discrepancy.

Synthetic CPU examples may demonstrate the numerical consequence of a verified preprocessing difference. Historical code and reconstructions remain unchanged.

## Evidence

Small JSON/CSV/Markdown summaries, exact configs, tests, validators, checksums, and resource accounting belong in the repository. Full logs and existing/referenced tensors remain on PAC. The final capsule must be a timestamped `.tar.gz` with `.tar.gz.sha256`, safe member paths, and verified internal SHA-256 sums.

The final recommendation must be exactly one of: a precisely defined NP conditional-selection correction; a precisely defined symmetry-handling investigation; another mechanism directly supported by diagnostics and requiring planner review; or stop because neither mechanism is supported. No positive recommendation is required.
