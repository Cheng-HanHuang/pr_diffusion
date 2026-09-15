# B25 native NP audit

## Accepted parent identity

The historical parent used by B24 is the B23 frozen `NP-1` parent in `configs/b23/np1_frozen.yaml`.

Frozen parent semantics:

- native family: DiffFPR-guided noise proposal and ranking;
- policy entrypoint: `scripts/pr_external_difffpr_np_guided_lf_s2_selector.py`;
- base proposal/ranking implementation: `scripts/pr_external_difffpr_np_benchmark.py`;
- 1000 scheduler timesteps;
- soft candidate count `K=5` before projection;
- hard candidate count `K=1` after projection;
- projection begins at transition 300;
- score mode `lf`;
- score radius `0.6`;
- projection radius `0.2`, schedule `300:0.2`;
- terminal selection statistic: mean post-projection winner low-frequency measurement MSE.

B24 DEV80 wraps this parent through:

`run_b24_3_dev80_np.py -> run_b24_3_epp321_refinement.py -> run_b24_3_np_branching.py -> pr_external_difffpr_np_guided_lf_s2_selector.py -> pr_external_difffpr_np_benchmark.py`.

B25 does not substitute a generic or similarly named “noise picking” script for this chain.

## Native proposal semantics

For one NP branch at a transition, the B24 candidate-set implementation first restores that branch's saved CPU/CUDA RNG state. Let the current denoised state be `x0_hat`. If the transition is at or after the frozen projection start, low-frequency magnitude projection is applied to `x0_hat` before candidate construction.

At the next diffusion timestep, each candidate has

`x_t_candidate = sqrt(alpha_bar) x0_hat + sqrt(1-alpha_bar) eps_candidate`.

The neural denoiser is evaluated at that noisy candidate, producing a denoised candidate `x0_candidate`. The measurement-dependent candidate score is the low-frequency oversampled Fourier-magnitude discrepancy of this **denoised candidate**, not a score of raw Gaussian noise alone.

The candidate with minimum score is propagated. The propagated native state includes both the winning `x0_candidate` and the winning `eps_candidate`.

## Reused versus independent noise

Historical NP is not the iid-candidate toy model used first in B25 Experiment 1.

When `K>1` and the branch already has a previous winning noise `eps_prev`, candidate index 0 reuses `eps_prev`. The remaining candidates are freshly drawn Gaussian noise tensors. At the first transition, or whenever no `eps_prev` exists, candidates are fresh Gaussian draws.

Therefore the historical proposal set can contain an incumbent-correlated proposal plus fresh proposals. Selection also changes the future state and the `eps_prev` carried into the next transition. This makes a one-step independent order-statistic calculation informative about a mechanism but not a direct theorem about the complete NP trajectory.

## Proposal count and B24 schedules

The NP-1 parent uses `K=5` for transitions before 300 and `K=1` thereafter. B24's `NP4_INDEPENDENT` runs four independent NP-1 roots, giving 8796 proposal UNet evaluations plus four initial evaluations, or 8800 total UNet evaluations per image.

B24 EPP/PE3 descendants changed population retention and allocation but did not change the native proposal-scoring semantics inside a branch. Those B24 refinement arms failed their advancement gate and remain closed. B25 does not tune them.

## Selection score and measurement dependence

The native score uses a radius-0.6 low-frequency mask on the channelwise oversampled Fourier magnitude. The winner is `argmin` score with stable candidate-index tie breaking. Runtime selection is measurement-dependent and clean-free; ground truth is used only for offline metrics/oracles.

The B24 DEV80 wrapper verifies the locked raw measurement tensor against its recorded hash and schema, then constructs

`measurement_np = measurement_raw.clamp_min(0.0)`.

That clamped tensor is passed to the NP scoring/projection context. The raw artifact is not overwritten.

This preprocessing fact is a B25.4 audit target because the historical measurement model adds Gaussian noise to Fourier amplitudes, which can generate negative stored values. B25 must compare the exact DAPS/SITCOM data paths before deciding whether this constitutes a cross-solver discrepancy or has material consequences.

## Projection placement

Projection is late, beginning at transition 300. Before candidate generation at such a transition, NP applies one Gerchberg–Saxton-style low-frequency magnitude replacement to the current denoised state using projection radius 0.2. Candidate scoring then continues on denoised proposals using score radius 0.6.

Thus B25's simplified small-step model omits at least four native ingredients:

1. denoiser transformation of each proposed noisy state;
2. reuse of the previously selected noise as one proposal;
3. dependence accumulated through prior selections;
4. late nonlinear low-frequency projection.

Any applicability claim from B25 Experiment 1 must state these omissions.

## RNG behavior

Each branch snapshots/restores CPU and CUDA RNG state. Independent NP4 roots use frozen domain-derived root seeds. Population forks in B24 use explicit domain-separated seeds. Candidate sampling advances the branch RNG and the winning branch carries the resulting RNG snapshot forward.

Experiment 1 uses separate named NumPy streams and is not a replay of this CUDA RNG process.

## B24 outcome boundary

B24's final binding decision is `STOP_B24_METHOD_REFINEMENT`. On DEV80, NP4 had 55/80 Good25; EPP321 had 50/80; PE3-score had 52/80; PE3-random had 53/80. Neither PE3 arm passed its frozen advancement gate. B25 treats these as historical outcomes, not evidence that the B25 hypotheses are already established.

## Applicability statement to carry into the final report

If the independent-proposal toy experiment shows hard selection has an `O(sqrt(h))` directional effect while likelihood weighting has an `O(h)` effect, the supported conclusion is limited to this: **hard measurement-dependent finite proposal selection can impose a parametrically stronger local drift than a smooth likelihood tilt in an independent small-step model.**

Whether that mechanism materially explains historical NP failures must be assessed jointly with the exact-prior benchmark, native preprocessing audit, and DEV80 diagnostics. The B25 result must not be promoted to a theorem about native NP merely from Experiment 1.
