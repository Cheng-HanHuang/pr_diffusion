# Current experiment plan: multi-lambda selector after four-GPU batch

Updated: 2026-05-23

## Current state

The four-GPU batch after LF/S2 validation changed the direction of the project.

Previous question:

```text
Can LF/S2 selection with a fixed S2 lambda avoid catastrophic failures?
```

Current answer:

```text
With four seeds, fixed LF/S2 selection has enough good candidates and passes the FFHQ-25 reliability target.
With two seeds, fixed LF/S2 can still fail because some seed pairs contain no good LF/S2 candidate for hard images.
Focused tuning shows that S2 lambda selection, not projection timing or memory, is the strongest next direction.
```

The next phase should therefore prioritize a **multi-lambda selector**.

## Evaluation standard

Use raw alignment first.  For every selector method, report:

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
selected seed rule counts
worst images
```

Promotion target:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

For method development, also track all-run reliability.  A best-of-k selector may pass the image-level target while individual runs remain fragile.

## Important lessons from the four-GPU batch

### 1. Four-seed LF/S2 fixes candidate availability

Full FFHQ-25 with seeds `100,101,102,103`:

```text
selected_config_seed_by_selector:
  mean ≈ 29.360
  min ≈ 27.081
  below25 = 0
```

The candidate pool contains good reconstructions for all images.  This confirms that the `102,103` validation failure was mainly a candidate-availability issue, not a wrong config decision.

### 2. The old 5e-5 seed tie-break is unsafe for four seeds

With four seeds, the old threshold selected a bad `00032` run.  Use either:

```text
seed choice by selector statistic directly
```

or a much smaller tie threshold around:

```text
1e-5
```

Do not use `5e-5` for four-seed selection.

### 3. Projection-start tuning is not the main path

Projection-start tuning showed timing brittleness and helped some images, but it did not fix `00005`.  It should not be the next main axis.

### 4. S2 lambda selection is the strongest new direction

Focused lambda diagnostic showed:

```text
lambda=0.02 or 0.05 fixes 00005 and 00014.
lambda=0.005 preserves 00028.
```

The lambda oracle on the focused subset achieved:

```text
mean ≈ 29.629
min ≈ 28.722
below25 = 0
```

Selecting lambda by post-projection winner LF-MSE nearly matched that oracle on the focused subset.

### 5. Memory hard2 LF is complementary but not main

Memory hard2 LF fixed `00005` moderately, but failed on `00028`.  Memory hard2 S2 was worse and should be dropped.  Memory should remain a fallback-only research direction.

## Next experiment 1: focused multi-lambda selector

### Goal

Confirm that lambda selection is executable, not just oracle-complementary.

### Setup

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds = 102,103
configs:
  LF
  S2 lambda=0.005
  S2 lambda=0.02
  optional S2 lambda=0.05
proj_start = 300
schedule = pre_projection_only
score_radius = 0.6
proj_radius = 0.2
```

### Selection rule

For each image:

```text
for each config:
  compute mean post-projection winner LF-MSE vs noisy observation across seeds
choose config with lowest statistic
choose seed by selector statistic directly
```

For this focused test, do not use the old 5e-5 tie-break.  Optionally report a 1e-5 tie-break separately, but treat selector-stat seed choice as the primary row.

### Success criteria

```text
00005 > 25 dB
00014 > 25 dB
00028 > 25 dB
00007 and 00009 not broken
focused-subset min > 25 dB
```

### Interpretation

If it passes:

```text
Proceed to full FFHQ-25 multi-lambda selector.
```

If it fails:

```text
Inspect which image fails and whether the selector or the candidate pool failed.
```

## Next experiment 2: full FFHQ-25 multi-lambda selector with two seeds

Run only after focused multi-lambda selection passes.

### Setup

```text
images = full FFHQ-25
seeds = 102,103 first
configs = LF, S2 lambda=0.005, S2 lambda=0.02
optionally include S2 lambda=0.05 if focused test shows it adds value
```

### Motivation

The major open question is whether multi-lambda selection can reduce the required seed budget.  The fixed LF/S2 selector failed for `102,103` because it had no good candidates for `00005` and `00014`.  GPU2 suggests high S2 lambda can create good candidates for those cases.

### What we want to see

```text
raw mean > 28.8
raw min > 25
below25 = 0
```

If it passes with `102,103`, multi-lambda selection is a major improvement over fixed LF/S2.

## Next experiment 3: full FFHQ-25 multi-lambda selector with four seeds

Run after the two-seed test, or if the two-seed test still fails.

### Setup

```text
images = full FFHQ-25
seeds = 100,101,102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, optional lambda=0.05
```

### Motivation

This tests the best current candidate method under a larger candidate budget.  Compare against the four-seed fixed LF/S2 result:

```text
fixed LF/S2 selected_config_seed_by_selector:
  mean ≈ 29.360
  min ≈ 27.081
  below25 = 0
```

### What we want to see

```text
mean >= 29.36
min >= 27.08
below25 = 0
```

If the multi-lambda version improves mean/min, it becomes the active best method.

## Next experiment 4: LPIPS/SSIM serious rerun

Once a multi-lambda selector variant passes, rerun the selected configuration with LPIPS enabled.

Reason:

Previous recent selector runs skipped LPIPS for speed.  The method should eventually be compared on PSNR, SSIM, and LPIPS.

## Deferred directions

### Projection-start selector

Projection-start complementarity exists but is weaker than lambda complementarity, and `00005` remained unresolved.  Defer unless multi-lambda selection fails.

### Memory fallback

Memory hard2 LF is complementary but lower quality and fails on `00028`.  Use only after defining a confidence/fallback rule.  Do not add it blindly to the main selector pool.

### Soft projection and robust frequency weighting

Still scientifically interesting, especially because broad hard projection was harmful.  Defer until the selector path is exhausted or stabilized.

### NP-in-SITCOM

Still promising as a separate hybrid direction, but it requires more code integration.  Revisit after multi-lambda selector experiments clarify whether NP-side scoring/selection is enough.

## Immediate launch recommendation

Next launch should be:

```text
focused multi-lambda selector
```

Use full per-step diagnostic logging on the focused subset.  Do not run full FFHQ-25 until the focused selector confirms that the statistic can choose among lambdas without creating a new failure.
