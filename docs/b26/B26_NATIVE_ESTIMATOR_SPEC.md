# B26 native intermediate-likelihood estimator specification

## Quantity of interest

For a noisy candidate state `z_t`, the ideal prior-conditional observation likelihood is

`L_true(z_t) = E[p(y_raw | X_0) | Z_t=z_t]`,

where the conditional law is induced by the true data prior and the forward noising process. A pretrained diffusion model instead provides a learned reverse-chain conditional law. A realizable continuation estimator therefore targets

`L_theta,R(z_t) = E_{X_0 ~ P_theta,R(.|z_t)}[p(y_raw|X_0)]`,

where `P_theta,R` is the explicitly chosen learned reverse kernel/discretization. These are not assumed equal.

For M independent measurement-free continuations `X_0^(m)` from the same candidate,

`L_hat_M(z_t) = M^{-1} sum_m p(y_raw | X_0^(m))`.

Likelihood values must be averaged arithmetically via log-sum-exp. The exponential of the mean log-likelihood estimates a different quantity and is prohibited.

## Concrete future FFHQ feasibility design

This is a cost/design specification only; B26 does not execute it.

- Candidate state: the native NP re-noised state immediately before the UNet evaluation used to form each candidate denoised `x0_cand`.
- Learned continuation kernel: the same frozen DiffFPR FFHQ UNet and scheduler family as native NP, run **without measurement scoring, projection, or measurement guidance** inside the continuation. Before any pilot, a source-level gate must identify and test the scheduler's native stochastic reverse transition from an arbitrary current timestep; if the accepted scheduler cannot supply that kernel without changing model semantics, the weighted pilot is blocked rather than silently substituting DDIM/DDPM code.
- Respacing: 8 monotonically decreasing reverse transitions from the candidate's current diffusion time to the terminal model time, chosen deterministically by nearest indices on the frozen 1000-step native timetable and including the terminal endpoint. Duplicate rounded indices are disallowed.
- Inner sample count: M=4 independent continuation lineages/candidate.
- Output representation: terminal model-range RGB tensor, converted to the same `[0,1]`, symmetric-pad-64, centered orthonormal FFT magnitude representation as the frozen NP observation operator.
- Observation likelihood: signed locked `y_raw`, Gaussian amplitude model with frozen `sigma_y=0.05`; use the same historical low-frequency mask only if the outer weighting target is explicitly defined as that masked pseudo-likelihood. A true full observation likelihood uses the full stored measurement and must not be relabeled as the historical LF selector.
- Projection: no projection inside inner continuations. Projection remains an outer NP operation and uses `y_plus` only.
- Inner RNG: independent namespace from proposal, outer categorical, and historical branch RNG; inner states are integration auxiliaries and never become uncounted terminal candidates.

### Model-call cost

Historical NP uses 300 pre-projection transitions with K=5 candidate evaluations, then 699 post-projection transitions with K=1, plus one initial UNet call: 2,200 calls/root.

If an intermediate-likelihood estimate is evaluated for every pre-projection candidate, the concrete M=4, R=8 continuation design adds

`300 * 5 * 4 * 8 = 48,000`

UNet calls/root, before any batching benefit. Total becomes `50,200` calls/root = `22.818...` historical NP1 work-FRE. Four roots require 200,800 calls/image; DEV80 would require 16,064,000 calls. This count is the scientific work count even if GPU batching reduces wall time.

This cost is intentionally exposed before any weighted-FFHQ authorization. A future pilot should not hide it by calling continuations 'selector overhead'.

## Proposal-tilted target warning

Historical NP's candidate mechanism is not automatically the learned unconditional reverse kernel. At transition i, it starts from the current selected `x0_hat`, optionally projects it, re-noises as

`z = sqrt(alpha) x0_hat + sqrt(1-alpha) epsilon`,

then maps `z` through one UNet/Tweedie denoising evaluation to a candidate `x0_cand`. For K>1, candidate 0 may reuse the incumbent selected noise while the others draw fresh Gaussian noise. Thus each candidate is a push-forward from a history-dependent proposal `q_i`, and candidate 0 can have a different proposal law.

If candidates `u_j` are sampled from proposal laws `q_j` and are resampled only proportional to an intermediate likelihood estimate, the selected distribution is generally proposal-tilted: it approximates a finite-candidate version of `q(u)L(u)`, not an arbitrary desired conditional kernel. Exchangeability removes the candidate-index asymmetry but does **not** prove `q` equals the desired learned reverse kernel.

For a stated target kernel density `p_target(u | history)`, a principled importance weight requires, where densities are available,

`w_j ∝ L_hat(u_j) * p_target(u_j|history) / q_j(u_j|history)`.

If the push-forward proposal density is unavailable, no posterior-exactness claim is permitted. A weighted-NP pilot may still be studied as a heuristic, but its target must be labeled accordingly.

## Error decomposition

Any eventual result must distinguish:

1. **prior/model error:** `P_theta,R(X0|z_t)` differs from the true-prior conditional;
2. **discretization/respacing error:** the 8-step continuation differs from the model's full learned reverse chain;
3. **finite Monte Carlo error:** M=4 approximation of `L_theta,R`;
4. **outer finite-candidate error:** K=5 resampling approximation;
5. **proposal-target error:** historical NP proposal differs from the desired reverse kernel;
6. **observation approximation:** masked LF pseudo-likelihood versus full Gaussian observation likelihood.

B26.2 measures items 3 and 4 in an exact finite-support setting where the inner conditional distribution is analytically sampleable. It does not validate the FFHQ continuation kernel above.
