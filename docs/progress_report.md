# Progress report: four-GPU LF/S2 selector, lambda selection, and memory fallback study

Updated: 2026-05-23

This report replaces the selector-validation report after the four-GPU next-batch experiments.  The previous active report is archived in `docs/historical/progress_report_archived_20260523_before_four_gpu_batch.md`.

## Executive summary

The four-GPU batch clarified the current state of the project:

```text
1. Four-seed LF/S2 selection solves candidate availability on FFHQ-25.
2. The old 5e-5 seed tie-break is unsafe for four seeds.
3. Projection-start tuning shows strong timing brittleness but does not fix 00005.
4. S2 lambda selection is the strongest new direction: high lambda fixes 00005/00014, low lambda preserves 00028.
5. Memory hard2 LF is complementary but weaker; memory hard2 S2 should be dropped for now.
```

The next best direction is no longer plain LF/S2 with one fixed S2 lambda.  The next candidate method should be a **multi-lambda selector** using computed trajectory/measurement statistics:

```text
Candidate configs:
  LF
  S2 lambda = 0.005
  S2 lambda = 0.02 or 0.05

Selection statistic:
  mean post-projection winner low-frequency MSE vs noisy observation
```

This is motivated by the GPU2 focused lambda diagnostic, where a lambda selector nearly matched the lambda oracle on the focused subset.

## Benchmark and evaluation conventions

Main benchmark:

```text
Dataset: FFHQ 25-image split
Resolution: 256
Noise level: sigma_y = 0.05
Measurement: DiffFPR-style centered FFT magnitude after symmetric zero padding
Oversample: 2
Split: /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt
Image root: /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
Guided diffusion checkpoint: /egr/research-pac/huang248/models/ffhq_10m.pt
```

Primary reporting convention:

```text
Use raw alignment first.
Report mean, median, min, below20, below25, SSIM, and LPIPS if available.
Distinguish best-of-k from all-run reliability.
Distinguish oracle selection from executable non-ground-truth selection.
```

Core NP parameters unless otherwise stated:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

## Background before this four-GPU batch

The previous selector-validation stage established:

- LF and S2 with `lambda=0.01` have complementary failures.
- A non-ground-truth trajectory statistic, mean post-projection winner LF-MSE vs noisy observation, can choose LF vs S2 nearly at oracle level when good candidates exist.
- With seeds `100,101`, the LF/S2 seed tie-break selector passed FFHQ-25:

  ```text
  mean ≈ 29.220
  min ≈ 25.077
  below25 = 0
  ```

- With seeds `102,103`, the same method failed because LF/S2 had no good candidates for `00005` and `00014`:

  ```text
  mean ≈ 27.921
  min ≈ 9.117
  below25 = 2
  ```

- The ground-truth oracle over LF/S2 candidates for `102,103` also failed, so the issue was candidate availability, not mainly selector error.

This motivated a four-GPU batch:

```text
GPU0: full FFHQ-25 four-seed LF/S2 selector control.
GPU1: focused projection-start diagnostic.
GPU2: focused S2 lambda diagnostic.
GPU3: focused memory fallback diagnostic.
```

The focused diagnostics used:

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds  = 102,103
```

These are mechanism diagnostics.  Because they use `--select_images`, measurement-noise indexing follows the filtered subset order rather than the full FFHQ-25 order, so they should not be treated as exact full-25 reproductions.  They are still highly useful for comparing failure patterns and complementarity.

## GPU0: full FFHQ-25 four-seed LF/S2 selector control

### Setup

```text
images = full FFHQ-25
seeds = 100,101,102,103
configs = LF and pre-projection S2 lambda=0.01
sigma_y = 0.05
np_steps = 1000
LPIPS skipped
```

### Results

Raw best-of-4 / selector summaries:

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| LF best-of-4 | 29.424 | 29.723 | 27.185 | 0.832 | 0 | 0 |
| S2 best-of-4 | 29.429 | 29.723 | 27.185 | 0.832 | 0 | 0 |
| LF/S2 oracle over all 8 candidates | 29.434 | 29.723 | 27.185 | n/a | 0 | 0 |
| selected config best-of-4 | 29.430 | 29.723 | 27.185 | 0.832 | 0 | 0 |
| selected config seed by selector | 29.360 | 29.585 | 27.081 | 0.831 | 0 | 0 |
| global run by selector | 29.374 | 29.585 | 27.081 | 0.831 | 0 | 0 |
| old 5e-5 tie-break selector | 28.575 | 29.690 | 9.378 | 0.809 | 1 | 1 |

### Interpretation

The four-seed candidate pool fixes the candidate-availability failure:

```text
With seeds 100,101,102,103, LF/S2 contains good candidates for every FFHQ-25 image.
```

The config selector remains strong.  However, the old 5e-5 seed tie-break is unsafe with four seeds.  It caused a failure on `00032`, selecting a bad S2 seed despite good S2/LF candidates being available.

Threshold sweep from the GPU0 run:

| Tie threshold | Mean PSNR | Min | Images <20 | Images <25 |
|---:|---:|---:|---:|---:|
| 0 | 29.360 | 27.081 | 0 | 0 |
| 1e-5 | 29.374 | 27.081 | 0 | 0 |
| 2e-5 | 28.592 | 9.378 | 1 | 1 |
| 3e-5 | 28.598 | 9.378 | 1 | 1 |
| 5e-5 | 28.575 | 9.378 | 1 | 1 |

Conclusion:

```text
For four-seed LF/S2, use selector-stat seed choice directly or a much smaller tie threshold around 1e-5.
Do not use the old 5e-5 threshold.
```

Important caveat: per-run reliability remains poor even though best-of-4 is good.

All-run raw statistics:

| Config | All-run mean | All-run median | Min | Runs <20 | Runs <25 |
|---|---:|---:|---:|---:|---:|
| LF | 23.441 | 28.808 | 5.076 | 32 / 100 | 32 / 100 |
| S2 | 25.157 | 29.109 | 5.687 | 23 / 100 | 23 / 100 |

Thus, the four-seed selector solves candidate selection under a larger budget, but not per-run robustness.

## GPU1: focused S2 projection-start diagnostic

### Setup

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds = 102,103
score_mode = prev_l2
lambda = 0.01
schedule = pre_projection_only
proj_start in {200,300,400,500}
```

### Results

| proj_start | Mean PSNR | Median | Min | Images <20 | Images <25 | Worst |
|---:|---:|---:|---:|---:|---:|---|
| 200 | 23.836 | 29.006 | 9.025 | 2 | 2 | 00009 |
| 400 | 21.737 | 28.925 | 9.745 | 3 | 3 | 00005 |
| 500 | 20.005 | 17.598 | 9.130 | 4 | 4 | 00005 |
| 300 | 19.450 | 13.790 | 9.127 | 4 | 4 | 00005 |

Per-image pattern:

| Image | start=200 | start=300 | start=400 | start=500 | Best |
|---|---:|---:|---:|---:|---:|
| 00005 | 10.614 | 9.127 | 9.745 | 9.130 | 10.614 |
| 00007 | 28.610 | 10.402 | 10.569 | 10.396 | 28.610 |
| 00009 | 9.025 | 30.045 | 30.065 | 30.142 | 30.142 |
| 00014 | 29.293 | 13.662 | 29.332 | 29.260 | 29.332 |
| 00018 | 29.006 | 28.913 | 28.925 | 17.598 | 29.006 |
| 00028 | 29.968 | 13.790 | 13.477 | 13.118 | 29.968 |
| 00034 | 30.334 | 30.213 | 30.046 | 30.390 | 30.390 |

Oracle over projection starts:

```text
mean ≈ 26.866
median ≈ 29.332
min ≈ 10.614
below20 = 1
below25 = 1
```

### Interpretation

Projection timing matters a lot, but it is not the strongest fix:

- `proj_start=200` is the best single timing in this subset.
- `00014` is fixable by timing.
- `00005` remains unresolved across all tested starts.
- Timing can also move failures: `00007`, `00009`, and `00028` prefer different timings.

Conclusion:

```text
Projection-start tuning confirms timing brittleness, but it does not provide a reliable replacement or fix 00005.
```

## GPU2: focused S2 lambda diagnostic

### Setup

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds = 102,103
proj_start = 300
score_mode = prev_l2
schedule = pre_projection_only
lambda in {0.005,0.01,0.02,0.05}
```

### Results

| lambda | Mean PSNR | Median | Min | Images <20 | Images <25 | Worst |
|---:|---:|---:|---:|---:|---:|---|
| 0.005 | 21.751 | 28.913 | 9.127 | 3 | 3 | 00005 |
| 0.01 | 19.450 | 13.790 | 9.127 | 4 | 4 | 00005 |
| 0.02 | 27.313 | 29.340 | 13.791 | 1 | 1 | 00028 |
| 0.05 | 27.312 | 29.340 | 13.791 | 1 | 1 | 00028 |

Per-image pattern:

| Image | λ=0.005 | λ=0.01 | λ=0.02 | λ=0.05 | Best |
|---|---:|---:|---:|---:|---:|
| 00005 | 9.127 | 9.127 | 30.161 | 30.161 | 30.161 |
| 00007 | 10.293 | 10.402 | 28.722 | 28.722 | 28.722 |
| 00009 | 30.052 | 30.045 | 30.052 | 30.045 | 30.052 |
| 00014 | 13.662 | 13.662 | 29.340 | 29.340 | 29.340 |
| 00018 | 28.913 | 28.913 | 28.913 | 28.912 | 28.913 |
| 00028 | 29.999 | 13.790 | 13.791 | 13.791 | 29.999 |
| 00034 | 30.213 | 30.213 | 30.213 | 30.212 | 30.213 |

Lambda oracle on this focused subset:

```text
mean ≈ 29.629
median ≈ 29.999
min ≈ 28.722
below20 = 0
below25 = 0
```

The best lambda varies by image:

```text
00005: lambda 0.05
00007: lambda 0.05
00009: lambda 0.005
00014: lambda 0.02
00018: lambda 0.01
00028: lambda 0.005
00034: lambda 0.02
```

A key observation from the analysis: selecting lambda by lower mean post-projection winner LF-MSE, or by final noisy low-frequency residual, nearly matched the lambda oracle on this focused subset:

```text
selected-by-stat mean ≈ 29.628
selected-by-stat min ≈ 28.722
below25 = 0
```

### Interpretation

This is the strongest result of the four-GPU batch.

S2 lambda tuning fixes the hard validation cases:

```text
00005: fixed by lambda 0.02/0.05
00014: fixed by lambda 0.02/0.05
```

but again moves failure elsewhere:

```text
00028: good at lambda 0.005, bad at lambda >= 0.01
```

Therefore, no single fixed lambda is enough, but a multi-lambda selector is promising and may be executable because the same trajectory statistic appears to select the right lambda on the focused subset.

Conclusion:

```text
The next main method should test LF + multiple S2 lambdas, selected by trajectory statistics.
```

## GPU3: focused memory fallback diagnostic

### Setup

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds = 102,103
configs:
  memory_k=1, hard=2, LF score
  memory_k=1, hard=2, S2 lambda=0.01
```

### Results

| Config | Mean PSNR | Median | Min | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---|
| memory_k1_hard2_lf | 24.963 | 26.492 | 14.691 | 1 | 1 | 00028 |
| memory_k1_hard2_s2_lam001 | 22.028 | 26.162 | 9.105 | 2 | 3 | 00005 |

Per-image pattern:

| Image | Memory LF | Memory S2 | Better |
|---|---:|---:|---|
| 00005 | 27.399 | 9.105 | Memory LF |
| 00007 | 26.269 | 26.162 | Memory LF |
| 00009 | 27.031 | 27.030 | Memory LF |
| 00014 | 25.147 | 24.935 | Memory LF |
| 00018 | 26.492 | 26.363 | Memory LF |
| 00028 | 14.691 | 12.551 | Memory LF |
| 00034 | 27.711 | 28.051 | Memory S2 |

### Interpretation

Memory hard2 LF is genuinely complementary:

- It fixes `00005` moderately (`27.399`).
- It gives non-catastrophic results on several images.
- It is much better than memory hard1 collapse.

However, it is too conservative and fails on `00028`:

```text
00028: memory LF = 14.691
```

Memory S2 is worse and should be dropped for now.

Conclusion:

```text
Memory hard2 LF may be a fallback candidate, but it is not the next main path.
The next main path should be multi-lambda S2 selection, not memory selection.
```

## Combined interpretation after GPU0--GPU3

The four experiments answer the main questions from the previous plan:

### Candidate availability

Four seeds fix the LF/S2 candidate-availability problem on FFHQ-25:

```text
selected_config_seed_by_selector with 4 seeds:
mean ≈ 29.360
min ≈ 27.081
below25 = 0
```

But all-run reliability remains poor; the solver still relies heavily on seed budget.

### Tie-break behavior

The old seed tie-break threshold `5e-5` was useful for two seeds but unsafe for four seeds.  It allowed final noisy low-frequency residual to dominate and selected a bad `00032` run.

Use:

```text
four seeds: selector-stat seed choice or tie threshold ≈ 1e-5
not 5e-5
```

### Projection timing

Projection-start tuning reveals strong brittleness but does not solve the hardest failure `00005`.  It is not the next main axis.

### Lambda strength

Lambda selection is the strongest new evidence.  High lambda fixes `00005` and `00014`; low lambda preserves `00028`.  The complementarity appears selectable by the same trajectory statistic.

### Memory fallback

Memory hard2 LF is complementary and can rescue `00005`, but it is weaker than lambda selection and fails on `00028`.  Keep it as fallback research, not as the main solver.

## Current best direction

The next method to implement is:

```text
Multi-lambda LF/S2 selector
```

Candidate configs:

```text
LF
S2 lambda = 0.005
S2 lambda = 0.02
```

Optionally include:

```text
S2 lambda = 0.05
```

Selection:

```text
choose config by mean post-projection winner LF-MSE vs noisy observation;
choose seed by the same selector statistic;
for four seeds, avoid the old 5e-5 tie-break.
```

Expected target:

```text
With seeds 102,103 on the focused subset:
  recover 00005/00014 while preserving 00028.

With four seeds on FFHQ-25:
  mean around or above 29.4
  min above 27
  below25 = 0
```

## Recommended next experiments

1. **Focused multi-lambda selector**

   ```text
   images = 00005,00014,00007,00009,00018,00028,00034
   seeds = 102,103
   configs = LF, S2 lambda=0.005, S2 lambda=0.02, optionally S2 lambda=0.05
   selection = post-projection winner LF-MSE statistic
   goal = confirm executable lambda selection, not just oracle complementarity
   ```

2. **Full FFHQ-25 multi-lambda selector, two seeds**

   Only if the focused selector works:

   ```text
   seeds = 102,103 first
   then seeds = 100,101 as a comparison
   goal = see whether multi-lambda selection reduces required seed budget
   ```

3. **Full FFHQ-25 multi-lambda selector, four seeds**

   If two-seed multi-lambda still fails:

   ```text
   seeds = 100,101,102,103
   goal = test whether the richer config pool improves over four-seed LF/S2
   ```

4. **Memory fallback only after multi-lambda tests**

   Test memory hard2 LF only as a fallback triggered by selector confidence.  Do not add memory to the main pool before proving a selector can avoid its `00028` failure.
