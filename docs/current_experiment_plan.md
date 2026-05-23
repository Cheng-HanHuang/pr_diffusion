# Current experiment plan: validating multi-lambda selection and moving toward always-success

Updated: 2026-05-23

## Current state

The current best method is a multi-lambda selector:

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

This method passed full FFHQ-25 with seeds `102,103`:

```text
selected_config_seed_by_selector, raw:
  mean ≈ 29.305
  min ≈ 26.230
  below20 = 0
  below25 = 0
```

This is a major improvement over fixed LF/S2 with the same seeds, which failed on `00005` and `00014`.

However, this is still a multi-run selector.  It is not yet an always-successful single-run algorithm.  The next experiments should distinguish:

```text
robustness from multiple scoring/lambda branches
versus
robustness from multiple random seeds
```

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
```

Promotion target:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

## Current negative ablations

Do not prioritize these unless new evidence appears:

```text
score_radius=0.4: failed on focused subset due to 00014
proj_start=200: failed on focused subset due to 00005
memory hard2 S2: worse than memory LF
old 5e-5 seed tie-break: unsafe for four-seed runs
```

Current recommended defaults remain:

```text
score_radius=0.6
proj_start=300
proj_radius=0.2
```

## Next experiment 1: full-25 multi-lambda with four seeds

### Setup

```text
images = full FFHQ-25
seeds = 100,101,102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius=0.6
proj_start=300
proj_radius=0.2
```

### Motivation

Four-seed fixed LF/S2 already passed with:

```text
mean ≈ 29.360
min ≈ 27.081
below25 = 0
```

Two-seed multi-lambda passed with:

```text
mean ≈ 29.305
min ≈ 26.230
below25 = 0
```

This experiment establishes the best current multi-run selector baseline and tests whether multi-lambda improves over fixed LF/S2 under the same four-seed budget.

### What we want to see

```text
mean >= 29.36
min >= 27.08
below25 = 0
```

If it passes and improves mean/min, it becomes the active best benchmark method.

## Next experiment 2: full-25 multi-lambda with new validation seeds 104,105

### Setup

```text
images = full FFHQ-25
seeds = 104,105
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
same score/projection settings
```

### Motivation

The method passed seed pair `102,103`.  We need to know whether two-seed multi-lambda success generalizes across seed pairs or whether it is still seed-pair sensitive.

### Interpretation

```text
If selected method passes and oracle passes:
  two-seed multi-lambda is more convincing.

If selected method fails but oracle passes:
  selector statistic needs improvement.

If selected method fails and oracle fails:
  two seeds are still not enough; candidate availability remains a seed-budget issue.
```

## Next experiment 3: single-seed multi-lambda simulation

### Setup

Use existing full-25 `102,103` diagnostic traces.  No new reconstructions are needed.

Simulate:

```text
seed 102 only
seed 103 only
```

Then after Experiment 2 finishes, simulate:

```text
seed 104 only
seed 105 only
```

### Motivation

This directly addresses the ultimate goal.  If one seed plus multiple scoring branches is enough, then branch/scoring complementarity may be sufficient.  If single-seed selection fails, then multiple random seeds are still essential.

### What we want to see

```text
single-seed selected_config_seed_by_selector:
  min > 25
  below25 = 0
```

Expected outcome: single-seed likely fails, but measuring this is important for deciding whether to focus on seed budgeting or in-loop adaptation.

## Next experiment 4: reduced-lambda pool ablation

### Setup

Full FFHQ-25, seeds `102,103`, compare:

```text
A. LF + S2 lambda=0.005 + S2 lambda=0.02
B. LF + S2 lambda=0.005 + S2 lambda=0.05
C. LF + S2 lambda=0.005 + S2 lambda=0.02 + S2 lambda=0.05
```

### Motivation

The full method currently runs four configs.  We need to know whether both high lambdas are necessary or whether a smaller pool gives the same reliability with lower cost.

### What we want to see

```text
If A or B matches C, remove the redundant lambda.
If C is better, keep both 0.02 and 0.05.
```

This can be done as a post-hoc selector subset analysis from the existing full-25 trace summaries before launching new reconstruction runs.

## Next experiment 5: LPIPS/SSIM serious rerun

Once the config pool is finalized, rerun the best selected method with LPIPS enabled.

Motivation:

Recent selector runs skipped LPIPS for speed.  We need LPIPS for fair reporting against SITCOM-style metrics.

## Algorithmic next step: in-loop adaptive multi-lambda scoring

The current method runs separate reconstructions and selects afterward.  To move toward an always-successful method, implement a single-run solver that adapts the scoring rule during reconstruction.

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

Instead of separate full reconstructions, sample a larger candidate set per step and score subgroups with different lambdas.  This may preserve branch diversity inside one trajectory.

### C. Adaptive compute fallback

Start with one seed and multi-lambda branches.  If confidence is low, add another seed or config.  This keeps the method executable while avoiding unnecessary multiple reconstructions on easy images.

## Immediate launch recommendation

Use four GPUs for:

```text
GPU0: full-25 multi-lambda seeds 100,101,102,103
GPU1: full-25 multi-lambda seeds 104,105
GPU2: post-hoc single-seed / reduced-lambda analysis from existing traces if no GPU needed; otherwise another validation pair 106,107
GPU3: optional full-25 multi-lambda seeds 106,107 or LPIPS-enabled rerun after pool is finalized
```

If GPU budget is available overnight, prioritize additional two-seed validation pairs over more score_radius/proj_start ablations, because score_radius=0.4 and proj_start=200 already failed in focused tests.
