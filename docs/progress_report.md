# Progress report: full-25 multi-lambda selector validation and next robustness goals

Updated: 2026-05-23

This report replaces the four-GPU LF/S2 selector and lambda diagnostic report after the focused and full-25 multi-lambda selector experiments.  The previous active report is archived in `docs/historical/progress_report_archived_20260523_before_full25_multilambda_validation.md`.

## Executive summary

The current best method is now a **multi-lambda LF/S2 selector**:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
configs:
  LF
  S2 lambda = 0.005
  S2 lambda = 0.02
  S2 lambda = 0.05
selection statistic:
  mean post-projection winner low-frequency MSE vs noisy observation
seed choice:
  selector statistic directly; do not use the old 5e-5 tie-break for multi-seed/four-seed runs
```

Main full-25 result with validation seed pair `102,103`:

```text
selected_config_seed_by_selector, raw:
  mean PSNR ≈ 29.305
  median    ≈ 29.584
  min       ≈ 26.230
  SSIM      ≈ 0.830
  below20   = 0
  below25   = 0
```

This is a major improvement over the fixed LF/S2 selector with the same seed pair, which failed on `00005` and `00014`.

The method is still a multi-run selector, not a per-run always-successful solver.  Individual runs remain fragile.  The next stage should therefore answer how much of the success comes from multiple scoring branches versus multiple random seeds, and whether we can move the selection into the reconstruction loop.

## Evaluation convention

Primary benchmark:

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

Always report raw alignment first:

```text
mean PSNR
median PSNR
minimum PSNR
number of images below 20 dB
number of images below 25 dB
SSIM mean
LPIPS mean when available
oracle over available candidates
selected result
selector regret vs oracle
selected config counts
worst images
```

Promotion target remains:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

## Background before the latest multi-lambda runs

Earlier experiments established:

1. Fixed LF/S2 selection with `lambda=0.01` is complementary but insufficient with some two-seed pairs.
2. The selector statistic, mean post-projection winner LF-MSE vs noisy observation, chooses LF vs S2 almost at oracle level when good candidates exist.
3. Fixed LF/S2 with seeds `100,101` passed, but seeds `102,103` failed because no good LF/S2 candidate existed for `00005` and `00014`.
4. Four-seed fixed LF/S2 fixed candidate availability but did not improve per-run reliability.
5. Focused lambda diagnostics showed that high S2 lambda fixes `00005` and `00014`, while low lambda or LF preserves `00028`.
6. Projection-start and memory fallback were useful diagnostics, but weaker than lambda selection.

This motivated the multi-lambda selector.

## Focused multi-lambda selector: score_radius=0.6, proj_start=300

### Setup

```text
images = 00005,00014,00007,00009,00018,00028,00034
seeds = 102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius = 0.6
proj_start = 300
selector statistic = post_winner_lf_mse_mean
LPIPS skipped
```

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 29.613 | 29.999 | 28.722 | 0.821 | 0 | 0 |
| selected_config_bestofk | 29.613 | 29.999 | 28.722 | 0.821 | 0 | 0 |
| global_run_by_selector | 29.628 | 29.999 | 28.722 | 0.821 | 0 | 0 |
| oracle_all_candidates | 29.629 | 29.999 | 28.722 | 0.821 | 0 | 0 |

The focused target was passed comfortably:

```text
00005 > 25
00014 > 25
00028 > 25
00007 and 00009 not broken
focused-subset min > 25
```

### Per-config single-method results

| Config | Mean PSNR | Median | Min | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---|
| LF | 27.373 | 29.999 | 13.662 | 1 | 1 | 00014 |
| S2 lambda=0.005 | 21.751 | 28.913 | 9.127 | 3 | 3 | 00005 |
| S2 lambda=0.02 | 27.313 | 29.340 | 13.791 | 1 | 1 | 00028 |
| S2 lambda=0.05 | 27.312 | 29.340 | 13.791 | 1 | 1 | 00028 |

No single config is reliable, but the selector avoids their complementary failures.

### Selected configs

```text
LF:             3 images
S2 lambda=0.005: 1 image
S2 lambda=0.02:  2 images
S2 lambda=0.05:  1 image
```

The only meaningful regret was on `00005`, where the selector chose LF at about 30.05 dB while the oracle chose S2 lambda=0.05 at about 30.16 dB.  Both are excellent, so this regret is harmless.

## Full FFHQ-25 multi-lambda selector: score_radius=0.6, proj_start=300

### Setup

```text
images = full FFHQ-25
seeds = 102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius = 0.6
proj_start = 300
selector statistic = post_winner_lf_mse_mean
LPIPS skipped
```

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 29.305 | 29.584 | 26.230 | 0.830 | 0 | 0 |
| global_run_by_selector | 29.288 | 29.584 | 26.230 | 0.830 | 0 | 0 |
| selected_config_bestofk | 29.416 | 29.723 | 27.182 | 0.832 | 0 | 0 |
| oracle_all_candidates | 29.417 | 29.723 | 27.182 | 0.832 | 0 | 0 |

This is the main result.  It passes the FFHQ-25 reliability target with the validation seed pair `102,103`.

### Comparison to fixed LF/S2 validation

Fixed LF/S2 with the same seeds had failed:

```text
fixed LF/S2 tie-break, seeds 102,103:
  mean ≈ 27.921
  min ≈ 9.117
  below25 = 2
  failure images: 00005, 00014
```

The multi-lambda selector fixes the same setting:

```text
multi-lambda selector, seeds 102,103:
  mean ≈ 29.305
  min ≈ 26.230
  below25 = 0
```

This demonstrates that the failure was not only a seed-budget problem; the configuration pool mattered.

### Single-config results in the full-25 run

| Config | Mean PSNR | Median | Min | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---|
| LF | 26.396 | 29.232 | 9.130 | 4 | 4 | 00005 |
| S2 lambda=0.005 | 26.486 | 29.232 | 9.130 | 4 | 4 | 00005 |
| S2 lambda=0.02 | 25.722 | 29.232 | 9.130 | 5 | 5 | 00005 |
| S2 lambda=0.05 | 28.753 | 29.593 | 13.792 | 1 | 1 | 00028 |

Each single config still fails.  The selector succeeds by combining complementary branches.

### Key per-image examples

| Image | LF | S2 λ=0.005 | S2 λ=0.02 | S2 λ=0.05 | Selected behavior |
|---|---:|---:|---:|---:|---|
| 00005 | 9.130 | 9.130 | 9.130 | 30.260 | selected high-lambda branch |
| 00014 | 13.706 | 13.706 | 29.312 | 29.312 | selected high-lambda branch |
| 00028 | 30.141 | 13.792 | 13.792 | 13.792 | selected LF branch |
| 00032 | 9.864 | 9.864 | 29.109 | 29.109 | selected high-lambda branch |

These are exactly the failure-moving patterns that motivated multi-lambda selection.

### Selector quality

The config selector is nearly oracle-level:

```text
selected_config_bestofk mean = 29.416462
oracle_all_candidates mean   = 29.416575
```

The remaining gap is mostly seed selection:

```text
selected_config_seed_by_selector mean = 29.305
selected_config_bestofk mean          = 29.416
gap ≈ 0.111 dB
```

Worst seed-selection regrets were moderate but non-catastrophic:

| Image | Selected PSNR | Oracle PSNR | Regret |
|---|---:|---:|---:|
| 00010 | 26.230 | 27.530 | 1.300 |
| 00013 | 27.474 | 27.977 | 0.503 |
| 00020 | 29.375 | 29.723 | 0.348 |

All selected images remained above 25 dB.

Selected config counts:

```text
S2 lambda=0.05: 9 images
S2 lambda=0.02: 6 images
S2 lambda=0.005: 6 images
LF: 4 images
```

Oracle config counts were similar:

```text
S2 lambda=0.05: 8 images
S2 lambda=0.005: 7 images
S2 lambda=0.02: 6 images
LF: 4 images
```

### Per-run fragility remains

All-run raw stats remain fragile:

| Config | All-run mean | Median | Min | Runs <20 | Runs <25 |
|---|---:|---:|---:|---:|---:|
| LF | 22.94 | 28.43 | 5.68 | 17 / 50 | 17 / 50 |
| S2 lambda=0.005 | 23.91 | 28.89 | 7.94 | 15 / 50 | 15 / 50 |
| S2 lambda=0.02 | 24.33 | 29.13 | 8.60 | 14 / 50 | 14 / 50 |
| S2 lambda=0.05 | 25.76 | 29.18 | 7.94 | 10 / 50 | 10 / 50 |

The method is therefore still a multi-run selector.  The project should now distinguish robust selection from genuinely always-successful single-run reconstruction.

## Score-radius ablation: focused score_radius=0.4, proj_start=300

### Setup

```text
images = focused seven-image subset
seeds = 102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius = 0.4
proj_start = 300
```

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 27.384 | 29.999 | 13.662 | 0.741 | 1 | 1 |
| selected_config_bestofk | 27.408 | 29.999 | 13.662 | 0.741 | 1 | 1 |
| global_run_by_selector | 27.365 | 29.999 | 13.662 | 0.741 | 1 | 1 |
| oracle_all_candidates | 27.409 | 29.999 | 13.662 | 0.741 | 1 | 1 |

Even the oracle fails, so this is a candidate-pool problem, not a selector problem.  The failure is `00014`, whose best available PSNR is only about 13.66 dB.

Conclusion:

```text
score_radius=0.4 is too narrow and should not replace score_radius=0.6.
```

## Projection-start ablation: focused score_radius=0.6, proj_start=200

### Setup

```text
images = focused seven-image subset
seeds = 102,103
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius = 0.6
proj_start = 200
```

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 26.827 | 29.293 | 10.584 | 0.742 | 1 | 1 |
| selected_config_bestofk | 26.852 | 29.293 | 10.644 | 0.745 | 1 | 1 |
| global_run_by_selector | 26.827 | 29.293 | 10.584 | 0.742 | 1 | 1 |
| oracle_all_candidates | 26.852 | 29.293 | 10.644 | 0.745 | 1 | 1 |

Even the oracle fails, due to `00005`:

```text
00005 best available at proj_start=200 ≈ 10.64 dB
```

Conclusion:

```text
proj_start=200 is too early for the multi-lambda setting and should not replace proj_start=300.
```

## Current best setting

The current best validated setting is:

```text
Multi-lambda selector
score_radius = 0.6
proj_radius = 0.2
proj_start = 300
configs:
  LF
  S2 lambda = 0.005
  S2 lambda = 0.02
  S2 lambda = 0.05
selection:
  choose config by mean post-projection winner LF-MSE vs noisy observation
  choose seed by the same statistic
```

Status:

- Focused hard/guard subset passes.
- Full FFHQ-25 with seeds `102,103` passes.
- `score_radius=0.4` fails.
- `proj_start=200` fails.
- LPIPS is still skipped in these runs and should be added once the configuration is stable.

## What this means for the ultimate goal

The current method is not a single-run always-successful algorithm.  It is a non-ground-truth multi-run selector.  This is still a useful and legitimate direction because the selector is not using PSNR; it uses computed trajectory/measurement statistics.  However, it does not yet solve per-run fragility.

The key question for the next stage is:

```text
Does success require multiple random seeds, or are multiple scoring/lambda branches enough?
```

If single-seed multi-lambda selection succeeds, then branch/scoring complementarity is the main mechanism.  If it fails, then seed diversity remains essential and we need either adaptive compute or in-loop adaptive scoring.

## Recommended next experiments

1. **Full-25 multi-lambda with four seeds**

   ```text
   seeds = 100,101,102,103
   configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
   score_radius=0.6, proj_start=300
   goal: best current multi-run selector baseline
   ```

2. **Full-25 multi-lambda with a new two-seed validation pair**

   ```text
   seeds = 104,105
   same configs/settings
   goal: determine whether two-seed success generalizes beyond 102,103
   ```

3. **Single-seed multi-lambda simulation from existing traces**

   Use the existing full-25 `102,103` trace summaries to simulate:

   ```text
   seed 102 only
   seed 103 only
   ```

   This is CPU-only and does not require new reconstructions.

4. **If needed, run more single-seed validations**

   Use the `104,105` run to simulate seed 104 only and seed 105 only.  This directly tests whether multiple scoring branches can succeed without multiple random seeds.

5. **Algorithmic next step: in-loop adaptive multi-lambda scoring**

   Instead of running separate LF/S2 lambda branches and selecting afterward, implement a solver that chooses among lambda scores inside each reconstruction step.  This is the most direct path toward an always-successful method rather than a post-hoc multi-run selector.
