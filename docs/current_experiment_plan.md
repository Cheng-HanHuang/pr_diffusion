# Current experiment plan: from multi-lambda selection to adaptive reliability

Updated: 2026-05-23

## Current state

The current best empirical method is a multi-lambda selector:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
configs:
  LF
  S2 lambda=0.005
  S2 lambda=0.02
  S2 lambda=0.05
selection statistic:
  mean post-projection winner LF-MSE vs noisy observation
seed choice:
  selector statistic directly
```

This method passed full FFHQ-25 with seeds `102,103` and with four seeds `100,101,102,103`, but failed on later two-seed validation pairs:

```text
102,103:
  selected mean ≈ 29.305
  min ≈ 26.230
  below25 = 0

100,101,102,103:
  selected mean ≈ 29.361
  min ≈ 27.081
  below25 = 0

104,105:
  selected mean ≈ 28.496
  min ≈ 9.126
  below25 = 1
  failure: 00005; oracle also fails

108,109:
  selected mean ≈ 29.227
  min ≈ 24.442
  below25 = 1
  failure: 00013; oracle also fails
```

The conclusion is now clear:

```text
The selector is usually near-oracle over the available candidates.
The current bottleneck is not primarily selector error.
The current bottleneck is candidate generation / seed diversity.
Two seeds are not enough for an always-successful method.
Single-seed simulations also fail, even with all lambda branches.
```

Therefore, the plan should shift from simply validating more fixed two-seed settings to building an **adaptive reliability strategy** and a **probabilistic understanding** of candidate generation.

## Evaluation standard

Use raw alignment first.  For selector methods, report:

```text
mean PSNR
median PSNR
minimum PSNR
images below 20 dB
images below 25 dB
SSIM mean
LPIPS mean when available
oracle over available candidates
selected result
selector regret vs oracle
selected config counts
worst images
all-run reliability
single-seed simulations
candidate success rates per image
```

Promotion target:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

For the next phase, also report the adaptive compute budget needed per image.

## Current negative ablations

Do not prioritize these unless new evidence appears:

```text
score_radius=0.4: failed on focused subset due to 00014
proj_start=200: failed on focused subset due to 00005
memory hard2 S2: worse than memory LF
old 5e-5 seed tie-break: unsafe for four-seed runs
single-seed multi-lambda as currently implemented: fails
```

Current recommended defaults remain:

```text
score_radius=0.6
proj_start=300
proj_radius=0.2
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
```

## Priority 1: complete seed-pair evidence and success-probability table

### Goal

Estimate which images are intrinsically difficult under the current candidate generator and how many seeds/config branches are needed to reach high probability of success.

### Inputs

Use all completed full-25 multi-lambda traces:

```text
100,101,102,103
102,103
104,105
108,109
106,107 if available
```

### Analysis

For each image, compute:

```text
number of successful raw candidates >25 dB
success rate by config
success rate by seed
success rate by config family: LF / low lambda / high lambda
whether the selector picked a successful candidate when one existed
whether the oracle failed
```

### Motivation

This separates:

```text
candidate generation failure
selector failure
seed-pair luck
image-specific hardness
```

### Expected output

A table like:

```text
image_id | n_candidates | n_success | success_rate | best_config_family | recurring_failure?
```

Important hard images so far:

```text
00005
00013
00027
00028
00032
00034
```

## Priority 2: adaptive compute selector

### Goal

Move from fixed two-seed or four-seed selection to an adaptive budget:

```text
start cheap;
add seeds/configs only when confidence is low;
stop when the candidate pool appears reliable.
```

### Proposed algorithm

```text
Stage 1:
  run one seed with LF + selected lambda branches, or run a cheap subset of configs.

Risk check:
  compute selector confidence features:
    config-stat margin
    seed-stat margin if multiple seeds exist
    best candidate post_winner_lf_mse_mean
    final noisy_lowfreq_mag_l2
    disagreement among configs
    whether selected candidate is near known failure-risk regions

If high confidence:
  accept result.

If low confidence:
  add another seed.

Repeat until:
  confidence is high, or max budget is reached.
```

### Motivation

Fixed two-seed selection can fail because no good candidate exists.  Four seeds pass in available tests but cost more.  Adaptive compute may get four-seed reliability at less than four-seed average cost.

### Immediate experiment

Use existing traces to simulate adaptive compute policies before launching new reconstructions.

Policies to test:

```text
1 seed -> if high-risk, add 1 seed -> if still high-risk, add 2 more seeds
2 seeds -> if high-risk, add 2 more seeds
```

Success metric:

```text
below25 = 0
average number of seeds used per image as low as possible
```

## Priority 3: targeted recovery for recurring hard images

### Goal

Create good candidates for images where the current pool can fail even with two seeds.

Known examples:

```text
00005 failed for seeds 104,105.
00013 failed for seeds 108,109, with best candidate ≈24.44 dB.
```

### Candidate diagnostics

Focused subset:

```text
00005,00013,00027,00028,00032,00034
plus guard images 00007,00009,00014,00018
```

Try targeted variants only on this subset:

```text
lambda values: 0.01, 0.02, 0.05, 0.1
proj_start: 300, 350, 400
score_radius: keep 0.6 first
soft candidates: 5 vs 8
hard candidates: 1 vs 2
```

Do not immediately run a huge full-25 grid.  First check whether these variants can produce good candidates for `00005` and `00013` under the failing seed pairs.

## Priority 4: in-loop adaptive multi-lambda scoring

This is the main algorithmic path toward an always-successful method.

The current method runs separate reconstructions and selects afterward.  Instead, implement a single-run solver that adapts the scoring rule during reconstruction.

Possible approaches:

### A. Per-step lambda arbitration

At each step, evaluate candidates under multiple scores:

```text
score_lambda = normalized LF residual + lambda * normalized previous-state distance
lambda in {0, 0.005, 0.02, 0.05}
```

Choose either:

```text
candidate with best score under the selected lambda
```

or use a conservative meta-rule based on trajectory statistics.

### B. Parallel candidate groups within one run

Instead of separate full reconstructions, sample a larger candidate set per step and score subgroups with different lambdas.  This may preserve branch diversity inside one trajectory while avoiding fully separate reconstructions.

### C. Adaptive schedule from early diagnostics

Use early trajectory features to decide whether to use LF, low lambda, or high lambda for the rest of the run.

The success criterion should be stricter than post-hoc selection:

```text
single reconstruction per seed approaches selected multi-run reliability
or at least reduces the number of required seeds/config runs.
```

## Priority 5: probabilistic theory note

A short theory study is now worthwhile.

Pure phase retrieval is ill-posed up to ambiguities, and deterministic guarantees are unlikely for the full diffusion-prior heuristic.  But the empirical structure suggests a useful probabilistic decomposition:

```text
Failure probability ≈ P(no good candidate generated) + P(selector fails | good candidate exists).
```

The experiments suggest:

```text
P(selector fails | good candidate exists) is small.
P(no good candidate generated) is the main bottleneck.
```

A theory note should formalize:

1. Candidate-generation success probability for randomized diffusion trajectories.
2. Selector consistency conditional on at least one good candidate.
3. Adaptive compute: how many seeds/config branches are needed to make failure probability <= delta.
4. Role of the pretrained/diffusion prior as a restriction of the feasible set in ill-posed phase retrieval.
5. What can and cannot be proven without modeling the generative prior.

This theory does not need to prove full deterministic recovery.  A probabilistic reliability framework would already guide algorithm design.

## Immediate next actions

1. Analyze the `106,107` folder if available.
2. Build the empirical success-probability table across all completed seeds/configs.
3. Simulate adaptive compute policies using existing traces.
4. Run targeted recovery diagnostics for `00005` and `00013` if the success table confirms they are recurring bottlenecks.
5. Begin a short theory note on probabilistic reconstruction reliability.
6. Postpone LPIPS rerun until the candidate pool and adaptive policy are finalized.
