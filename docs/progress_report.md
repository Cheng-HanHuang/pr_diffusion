# Progress report: FFHQ phase retrieval, LF/S2 selector, and failure analysis

Updated: 2026-05-23

This report replaces the May 12 progress report after the follow-up LF/S2 selector, tie-break, and validation experiments.  The older report is archived in `docs/historical/progress_report_archived_20260523_before_selector_update.md` and remains available through Git history.

## Current project objective

The objective is to develop a reliable diffusion-prior phase retrieval solver for FFHQ-25 that avoids catastrophic failures, rather than only improving an averaged best-of-k number.  The current evidence suggests that the core problem is not simply choosing one fixed score or one fixed hyperparameter.  Different scoring rules fail on different images, so the project now focuses on reliable selection, fallback, and failure recovery using non-ground-truth diagnostics.

Important evaluation rules:

- Use raw alignment as the primary conservative metric when comparing to methods that do not perform ambiguity resolution.
- Always report mean, median, minimum PSNR, and counts below 20 dB and 25 dB.
- Treat best-of-k numbers as seed-budget results, not as per-run reliability.
- Distinguish between ground-truth oracle selection and executable selection using only computed metrics.

## Benchmark setting

Active benchmark:

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

Core NP baseline parameters inherited from the earlier FFHQ tuning phase:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

The previous tuning phase established that `proj_radius=0.2` is important, broad projection is harmful, late broadening is harmful, and simple LF scoring is fragile.

## What was tried after the May 12 plan

The May 12 plan proposed several directions: timestep-dependent S2, adaptive S2, candidate memory, NP-in-SITCOM, and robust measurement weighting.  We did not execute every direction.  The executed path was:

1. Scheduled/static S2 and memory-bank near-term experiments.
2. Diagnostic selector tracing on a focused subset and then on all FFHQ-25.
3. Lightweight LF/S2 selector using a trajectory statistic.
4. Seed tie-break postprocessor.
5. Validation on a different seed pair.

The unexecuted directions are still useful but should be reprioritized in light of the new evidence.  In particular, the validation failure suggests that the next question is not merely `2 seeds vs 4 seeds`; it is whether failures such as `00005` and `00014` require better tuning, better scoring, a complementary metric, or a fundamentally different recovery strategy.

## Clarification: the LF/S2 selector is not a PSNR oracle

The LF/S2 selector does not choose the reconstruction using ground-truth PSNR.  It runs two algorithmic configurations, LF and S2, and chooses between them using computed trajectory/measurement statistics.

Current selector logic:

```text
For each image:
  run LF for the selected seed budget;
  run pre-projection S2 for the same seed budget;

For config selection:
  compute mean post-projection winner low-frequency MSE vs the noisy observation;
  choose the config, LF or S2, with the lower mean statistic across seeds.

For seed selection with tie-break:
  inside the chosen config, compare each seed's post-projection winner LF-MSE statistic;
  if the best two seeds differ by more than threshold 5e-5:
      choose the lower-statistic seed;
  otherwise:
      treat the statistic as tied and choose the seed with lower final noisy low-frequency magnitude residual.
```

The ground-truth PSNR is used only afterward for evaluation.

This is different from a two-reconstruction oracle.  It is still a multi-run selector, but the selection criterion is non-ground-truth and executable.

## Near-term four-GPU experiments: scheduled S2 and memory

The first near-term batch tested:

```text
A1 decay S2, lambda0=0.005
A1 decay S2, lambda0=0.01
A1 pre-projection-only S2, lambda0=0.01
A3 memory_k=1 with LF score
```

Raw best-of-2 results:

| Config | Mean PSNR | Median | Min | SSIM | LPIPS | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decay S2 lambda=0.01 | 28.502 | 29.581 | 10.693 | 0.810 | 0.242 | 1 | 1 |
| preproj S2 lambda=0.01 | 27.629 | 29.582 | 7.961 | 0.774 | 0.269 | 2 | 2 |
| decay S2 lambda=0.005 | 27.080 | 29.280 | 10.539 | 0.756 | 0.283 | 3 | 3 |
| memory_k=1, hard=1 | 9.034 | 9.019 | 7.632 | 0.065 | 1.381 | 25 | 25 |

Conclusions:

- Scheduled S2 did not solve reliability.
- S2 still rescued some LF failures but created different catastrophic failures.
- Memory with `memory_k=1, hard=1` collapsed because the hard stage reused memory as the only candidate and had no fresh escape candidate.

## Second near-term batch: adaptive S2 and safer memory

Next, we tested:

```text
LF baseline in the same runner
adaptive S2 margin=0.03
adaptive S2 margin=0.10
memory_k=1, hard=2 with LF score
```

Raw best-of-2 results:

| Config | Mean PSNR | Median | Min | SSIM | LPIPS | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive S2 margin=0.03 | 27.629 | 29.582 | 7.961 | 0.774 | 0.269 | 2 | 2 |
| adaptive S2 margin=0.10 | 27.629 | 29.582 | 7.961 | 0.774 | 0.269 | 2 | 2 |
| LF baseline | 27.352 | 29.280 | 11.917 | 0.768 | 0.273 | 3 | 3 |
| memory_k=1, hard=2 | 26.775 | 26.866 | 24.892 | 0.790 | 0.197 | 0 | 1 |

Conclusions:

- Adaptive margins 0.03 and 0.10 produced identical final results, so the gate was not selective enough.
- S2 moved failures: it rescued `00018`, `00028`, `00034`, but failed on `00007`, `00009`.
- Memory hard2 avoided below-20 failures but capped quality near 26--27 dB, so it is a stabilizer/fallback candidate rather than a main method.

Important complementarity examples:

| Image | LF best-of-2 | S2 best-of-2 | Memory hard2 | Best source |
|---|---:|---:|---:|---|
| `00007` | 28.783 | 10.307 | 26.242 | LF |
| `00009` | 29.662 | 7.961 | 26.838 | LF |
| `00018` | 17.528 | 28.963 | 26.503 | S2 |
| `00028` | 12.063 | 29.996 | 27.582 | S2 |
| `00034` | 11.917 | 29.690 | 28.534 | S2 |
| `00027` | 25.102 | 25.077 | 25.932 | Memory |

The oracle over the four configs gave roughly:

```text
mean PSNR ≈ 29.271
median ≈ 29.631
min ≈ 25.932
below20 = 0
below25 = 0
```

This motivated selector diagnostics.

## Diagnostic selector experiments

A diagnostic runner was added to trace candidate scores and winners at every reconstruction step.  It records candidate-level LF/full residuals, previous-state distance, selection score, chosen winner, LF score gaps, adaptive activation, and final metrics.

First, a focused six-image diagnostic was run on:

```text
00007, 00009, 00018, 00027, 00028, 00034
```

This was useful for feature inspection, but it used filtered image order and therefore changed the measurement-noise indexing.  The full FFHQ-25 diagnostic was then run to match the benchmark setting.

The full-25 diagnostic found the strongest selector signal so far:

```text
Choose LF vs S2 by mean post-projection winner LF-MSE vs noisy observation.
```

For LF vs S2 only, this config selector nearly matched the ground-truth oracle:

```text
LF/S2 oracle: mean ≈ 29.237, min ≈ 25.102, below25 = 0
Selector:     mean ≈ 29.236, min ≈ 25.077, below25 = 0
```

It correctly handled the main catastrophic cases:

```text
00007: choose LF, avoid S2 failure
00009: choose LF, avoid S2 failure
00018: choose S2, rescue LF failure
00028: choose S2, rescue LF failure
00034: choose S2, rescue LF failure
```

## Lightweight LF/S2 selector and tie-break

A lightweight runner was implemented to avoid saving full candidate traces.  It accumulates the post-projection winner LF-MSE statistic during reconstruction and writes:

```text
run_level.csv
selected_image_level.csv
selected_summary.csv
selected_tiebreak_* files
```

For seeds `100,101`, the selector results were:

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| LF best-of-2 | 27.352 | 29.280 | 11.917 | 0.768 | 3 | 3 |
| S2 best-of-2 | 27.629 | 29.582 | 7.961 | 0.774 | 2 | 2 |
| selected config best-of-2 | 29.236 | 29.631 | 25.077 | 0.830 | 0 | 0 |
| seed by selector only | 28.981 | 29.631 | 19.617 | 0.820 | 1 | 1 |
| seed tie-break selector | 29.220 | 29.631 | 25.077 | 0.830 | 0 | 0 |

The seed-only selector failed on `00027`, where the bad seed had a slightly lower trajectory statistic.  A tie-break rule fixed this:

```text
if seed selector-stat gap <= 5e-5:
    choose the seed with lower final noisy_lowfreq_mag_l2
else:
    choose the seed with lower post-projection winner LF-MSE statistic
```

A threshold sweep on `100,101` showed a stable useful region around `3e-5` to `1e-4`; `5e-5` was selected as the default.

## Validation on seeds 102,103

The same LF/S2 tie-break selector was validated on a different seed pair, `102,103`.

Raw results:

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| LF best-of-2 | 26.396 | 29.232 | 9.130 | 0.739 | 4 | 4 |
| S2 best-of-2 | 27.280 | 29.232 | 9.130 | 0.763 | 3 | 3 |
| selected config best-of-2 | 27.934 | 29.593 | 9.130 | 0.783 | 2 | 2 |
| seed by selector only | 27.836 | 29.375 | 9.130 | 0.782 | 2 | 2 |
| seed tie-break selector | 27.921 | 29.585 | 9.117 | 0.783 | 2 | 2 |

This validation did not pass the reliability target.

However, the failure was not mainly a selector error.  The ground-truth oracle over all available LF/S2 candidates with seeds `102,103` was also bad:

```text
oracle over LF/S2 × seeds 102,103:
mean ≈ 27.934
min ≈ 9.130
below20 = 2
below25 = 2
```

The unavoidable failure images were:

| Image | Best LF | Best S2 | Best available |
|---|---:|---:|---:|
| `00005` | 9.130 | 9.130 | 9.130 |
| `00014` | 13.706 | 13.706 | 13.706 |

For these images, both LF and S2 failed for both validation seeds.  This means the seed pair did not contain a good candidate.  The selector cannot recover a good reconstruction that is absent from the candidate pool.

## Current interpretation

The evidence now supports a more precise conclusion:

```text
The LF/S2 config selector is strong and nearly oracle-level when good candidates exist.
The seed tie-break makes the method executable and worked on seeds 100,101.
The validation failure on seeds 102,103 is a seed-budget / candidate-availability failure, not primarily a config-selector failure.
```

Therefore, the next stage should not be only a larger seed-budget run, although that is one important control.  The more important scientific question is whether the remaining failures can be fixed by:

1. better tuning of the same NP/S2 family;
2. a better complementary metric or selector;
3. a failure-recovery fallback;
4. a fundamentally different scoring/selection strategy.

## Current best candidate method

The best current candidate method is:

```text
LF/S2 trajectory selector with seed tie-break
```

Algorithm:

```text
Run LF and pre-projection S2 with the chosen seed budget.
For each config, compute mean post-projection winner LF-MSE vs noisy observation.
Choose the config with the lower mean statistic.
Inside the chosen config, choose the seed by post-projection winner LF-MSE unless the seed-stat gap is <= 5e-5.
If tied, choose lower final noisy low-frequency magnitude residual.
```

Status:

- Passes FFHQ-25 for seed pair `100,101`.
- Fails validation seed pair `102,103` because two images have no good LF/S2 candidate.
- Should be treated as promising but not final.

## Open questions after validation

The main open questions are:

1. Do `00005` and `00014` fail under `102,103` because of unlucky seeds, or because the LF/S2 family has a structural blind spot for those images?
2. Would a 4-seed LF/S2 candidate pool fix the failures without changing the algorithm?
3. Can tuning `proj_start`, `score_radius`, `proj_radius`, or S2 lambda rescue `00005` and `00014` specifically?
4. Can a complementary selector metric identify when LF/S2 has no good candidate and trigger memory hard2, SITCOM-ODE, or another fallback?
5. Can a better scoring rule produce good candidates directly, rather than selecting among fragile candidates afterward?

These questions drive the updated experiment plan.
