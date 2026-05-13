# Current experiment plan: reliable phase retrieval solver after FFHQ NP/SITCOM study

Updated: 2026-05-12

## Guiding principle

The next phase should optimize for a single solver that reliably produces good reconstructions. Avoid post-hoc two-way oracle selection unless there is a mathematically justified selector or complementarity theorem.

Primary evaluation should include:

1. all-run mean and median;
2. seed-level failure counts;
3. image-level best-of-1 / best-of-2 / best-of-4 curves;
4. per-image variance across seeds;
5. failure cases and not only averages;
6. raw metrics whenever comparing to methods that do not ambiguity-align.

## Direction A: improve candidate score as one algorithm

Current finding:

- S1 low-frequency score is fragile.
- S2 `prev_l2` has useful signal, especially lambda around `0.01`, but moves failure cases.
- A fixed global lambda is not reliable enough.

### A1. Timestep-dependent S2

Try:

```text
score_i = LF_score + lambda(i) * previous_state_penalty
```

where lambda decays over the trajectory:

```text
lambda(i) = lambda0 * max(0, 1 - i / proj_start)
```

or only applies before projection:

```text
lambda(i) = lambda0 for i < proj_start
lambda(i) = 0 after proj_start
```

Motivation:

- Early branch choices affect global basin.
- Late regularization can over-constrain details.
- Previous fixed lambda sometimes created late/trajectory failures.

Suggested first grid:

```text
lambda0 in {0.005, 0.01, 0.02}
schedule in {pre_projection_only, linear_decay_to_proj_start}
sigma_y = 0.05
25 FFHQ images
2 seeds first
```

Promote only if:

```text
mean PSNR > 28.8
min PSNR > 25
below20 = 0
below25 = 0
```

### A2. Adaptive lambda from score uncertainty

Idea:

Use `prev_l2` penalty only when the LF scores are close or unreliable.

Example:

```text
if (best_lf - second_best_lf) / median_lf is small:
    use prev_l2 regularization
else:
    trust LF score
```

This is a single rule, not ground-truth selection.

Rationale:

- When one candidate clearly wins in measurement score, regularization may be unnecessary.
- When candidates are close, the LF score is noisy and stability should matter.

### A3. Candidate-bank / noise-memory selection

User-proposed direction:

At each timestep, keep the winning noise directions from the previous `k` steps and sample only:

```text
new_candidates = candidate_num - k
```

fresh candidates. The old noise directions must be rescaled appropriately to the current timestep.

Motivation:

- Some timesteps may contain no good fresh candidate.
- A previously successful noise direction may remain useful locally.
- This may reduce catastrophic branch switches.

First version:

```text
soft=5, hard=1
memory_k in {1,2}
reuse previous winning eps and maybe previous second-best eps
remaining candidates sampled fresh
score mode = LF or LF + weak prev_l2
```

Key implementation concern:

- `eps_prev` is currently kept as one candidate when `num_candidates > 1`.
- Extend this to a small `eps_memory` queue.
- At each step, include memory candidates first, then fill fresh candidates.
- Keep memory only if candidate is not stale or if it remains competitive.

Evaluation:

```text
sigma_y = 0.05
25 images
2 seeds
compare all-run and best-of-2 failure counts
```

## Direction B: NP inside SITCOM-ODE

Goal:

Build a single solver, not an oracle between NP and SITCOM.

Proposed algorithm:

```text
Early timesteps:
    use NP-style candidate branch selection with conservative low-frequency score.

Late timesteps:
    continue with SITCOM-ODE / DAPS-style consistency optimization.
```

Motivation:

- SITCOM-ODE is stronger in low-noise clean reconstruction.
- NP avoids some SITCOM failure cases and is more robust at high noise.
- NP's value may be early robust branch selection, while SITCOM's value is late consistency refinement.

Staged implementation:

1. NP warm start:
   - run partial or final NP;
   - save x0 estimate;
   - initialize SITCOM-ODE from this estimate if the SITCOM state parameterization allows it.
2. Partial NP to SITCOM continuation:
   - run NP to an intermediate timestep;
   - map x0 estimate into SITCOM's current ODE/noisy state;
   - continue SITCOM from corresponding step.
3. True NP-in-SITCOM:
   - add candidate branch selection inside SITCOM's early sampler loop;
   - score candidates with low-frequency measurement consistency;
   - use SITCOM's normal optimization later.

Success criterion:

```text
Improve per-run reliability and failure rate, not just best-of-k.
Preserve SITCOM's strong low-noise quality while reducing catastrophic failures.
```

## Direction C: robust measurement weighting

Finding:

- Hard projection beyond radius `0.2` is harmful.
- Late broadening to `0.4` or full is also harmful.
- SITCOM-ODE degrades sharply as measurement noise increases.

Hypothesis:

The solver should not hard-enforce noisy or high-frequency measurement components. Instead, use soft, robust, frequency-dependent weights.

Possible forms:

```text
measurement_weight(r, t, sigma_y)
```

with:

- higher weight at low frequency;
- lower weight at high frequency;
- lower overall measurement weight as `sigma_y` increases;
- robust residual loss such as Huber/Charbonnier instead of squared magnitude error.

First experiment:

```text
Replace hard low-frequency projection with a soft projection step:
x_new = (1 - eta) * x_prior + eta * projected_x
eta in {0.1, 0.25, 0.5}
radius = 0.2
```

or modify score:

```text
score = weighted robust magnitude residual
```

rather than hard projection.

## Direction D: evaluation standard going forward

For every new method, record:

```text
all-run PSNR mean/median/min
image-level best-of-1, best-of-2, best-of-4
number of runs <20 dB
number of images whose best-of-k <25 dB
per-image worst and failure identities
SSIM and LPIPS for final serious runs
runtime and effective candidate-call count
```

A method should be promoted only if it improves reliability without simply moving failures between images.

## Near-term four-GPU launch order

Recommended first batch, matching the available four-GPU workflow:

1. A1 static-to-decay S2 with `lambda0=0.005`.
2. A1 static-to-decay S2 with `lambda0=0.01`.
3. A1 pre-projection-only S2 with `lambda0=0.01`.
4. A3 memory bank with `memory_k=1` and original LF score.

If none of these reduces failures, prioritize NP-in-SITCOM rather than further widening the NP candidate count.
