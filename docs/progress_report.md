# Progress report: FFHQ phase retrieval, tuned Noise Picking, and SITCOM-ODE comparison

Updated: 2026-05-12

This document records the current project state after the FFHQ-25 guided NP benchmarking, parameter tuning, radius ablations, score-mode experiments, candidate-count experiments, and SITCOM-ODE comparisons.

## Current project objective

The project goal is not merely to beat SITCOM-ODE on one averaged table. The goal is to develop a reliable phase retrieval solver that:

1. produces good reconstructions per run, not only after heavy post-hoc selection;
2. has controlled failure modes;
3. uses a single well-defined algorithmic rule, not an oracle selector between multiple methods unless complementarity can be justified mathematically;
4. is evaluated on FFHQ as the main benchmark, with CelebA-HQ treated as historical/development-only.

## Current main benchmark

Main data/model setting:

- Dataset: FFHQ 25-image split.
- Image resolution: 256.
- Guided diffusion FFHQ checkpoint: `/egr/research-pac/huang248/models/ffhq_10m.pt`.
- Image root: `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`.
- Split file: `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt`.
- Measurement: DiffFPR-style centered FFT magnitude after symmetric zero padding.
- Oversampling: 2.
- Main noise level for method tuning: `sigma_y = 0.05`.
- Main NP evaluation convention:
  - report image-level best-of-k only when k is explicitly stated;
  - also track all-run mean, median, failure counts, and per-image worst cases;
  - use raw metrics whenever comparing against methods that do not ambiguity-align.

## Best current NP setting

The best practical NP setting found so far is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

A tiny numeric winner in one top-8 confirmation was:

```text
score_radius = 0.2
proj_radius  = 0.2
proj_start   = 500
soft_k       = 8
hard_k       = 1
```

but it was only about `0.016 dB` better than the cheaper practical setting, so the practical setting is preferred for ongoing development.

## NP parameter tuning and top-8 confirmation

A broad FFHQ tuning screen tested:

- score radius values including `0.2`, `0.4`, `0.6`, and fuller radii;
- projection radius values including `0.2`, `0.4`, `0.6`, and fuller radii;
- projection starts including `300`, `500`, `700`;
- candidate schedules including soft/hard settings such as `5/1` and `8/1`.

The clear finding was:

```text
proj_radius = 0.2 is essential.
proj_radius = 0.4 or larger is much worse.
```

Top-8 confirmation, using 25 images and 4 seeds at `sigma_y=0.05`, found:

| Config | Best-of-4 PSNR | SSIM | LPIPS | Worst best-of-4 PSNR |
|---|---:|---:|---:|---:|
| score=0.2, proj=0.2, start=500, soft=8, hard=1 | 29.440 | 0.8320 | 0.2242 | 27.243 |
| score=0.6, proj=0.2, start=300, soft=5, hard=1 | 29.424 | 0.8319 | 0.2241 | 27.185 |

These two settings are effectively tied. The second one is much cheaper and became the main practical setting.

Important interpretation:

- The best-of-4 average is respectable but not enough to beat low-noise SITCOM-ODE.
- The all-run mean is much lower because NP has seed-level failures.
- Therefore, method development should reduce per-run failure probability rather than only increasing best-of-k.

## Projection-radius and radius-schedule ablations

A two-stage projection-radius schedule was tested, motivated by the idea that NP may need small low-frequency projection early but might benefit from broader projection late.

Schedules included:

```text
500:0.2,700:0.4
500:0.2,800:0.4
500:0.2,900:0.4
500:0.2,900:0.72
300:0.2,700:0.4
300:0.2,800:0.4
300:0.2,900:0.4
300:0.2,900:0.72
```

Best schedule result:

```text
score=0.2, start=500, soft=8, hard=1, schedule=500:0.2,900:0.4
best-of-2 PSNR ≈ 26.71
```

Matching constant-radius baseline on the same 10 images and same seeds:

```text
score=0.2, proj=0.2, start=500, soft=8, hard=1
best-of-2 PSNR ≈ 29.13
```

Conclusion:

```text
Late broadening of the hard projection radius is decisively harmful in the tested form.
```

Even broadening only at step 900 from radius `0.2` to `0.4` lost roughly `2.4 dB`. Full-ish projection at `0.72` was worse.

Mechanistic conclusion:

```text
The diffusion prior tolerates a small low-frequency measurement correction.
Broader hard measurement replacement conflicts with the learned prior, even late.
```

## SITCOM-ODE comparison at sigma_y = 0.05

A same-split SITCOM-ODE run gave:

```text
SITCOM-ODE PSNR ≈ 29.690
SITCOM-ODE SSIM ≈ 0.733
SITCOM-ODE LPIPS ≈ 0.185
```

Current practical NP setting at `sigma_y=0.05`:

```text
NP best-of-4 PSNR ≈ 29.424
NP SSIM ≈ 0.832
NP LPIPS ≈ 0.224
```

Conclusion:

- NP does not beat SITCOM-ODE in PSNR or LPIPS at `sigma_y=0.05`.
- NP is much stronger in SSIM.
- NP is close enough that failure-mode analysis matters.

Per-image comparison at `sigma_y=0.05`:

- SITCOM-ODE had a major weak/failure image around PSNR `21.643`.
- NP reconstructed the same image around PSNR `30.141`.
- On the remaining non-failure images, SITCOM-ODE was typically stronger.

Interpretation:

```text
NP is not uniformly stronger, but it has complementary failure behavior.
It can avoid some SITCOM-ODE failure cases.
```

This supports failure-recovery or hybrid directions, but does not by itself justify an oracle selector.

## Noise-level comparison against SITCOM-ODE

A noise sweep used NP practical setting:

```text
score_radius=0.6, proj_radius=0.2, start=300, soft=5, hard=1
oversample=2
best-of-4 over 4 seeds
```

Noise levels:

```text
sigma_y in {0.00, 0.01, 0.05, 0.10, 0.20, 0.50}
```

NP raw best-of-4 vs SITCOM-ODE:

| sigma_y | NP raw PSNR | SITCOM-ODE PSNR | NP - SITCOM | NP SSIM | SITCOM SSIM | NP LPIPS | SITCOM LPIPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 32.372 | 35.852 | -3.480 | 0.924 | 0.935 | 0.125 | 0.067 |
| 0.01 | 33.444 | 35.736 | -2.292 | 0.947 | 0.933 | 0.100 | 0.067 |
| 0.05 | 29.424 | 29.690 | -0.266 | 0.832 | 0.733 | 0.224 | 0.185 |
| 0.10 | 24.203 | 23.302 | +0.901 | 0.626 | 0.411 | 0.491 | 0.395 |
| 0.20 | 19.606 | 17.429 | +2.177 | 0.400 | 0.168 | 0.731 | 0.629 |
| 0.50 | 13.034 | 11.121 | +1.913 | 0.143 | 0.036 | 0.834 | 0.777 |

Conclusion:

- SITCOM-ODE dominates low noise.
- NP becomes better in PSNR/SSIM at `sigma_y >= 0.10`.
- SITCOM-ODE retains better LPIPS at all tested noise levels.
- This supports a noise-robustness story for NP, not a clean-setting SOTA story.

Per-image pattern:

- At `sigma_y=0.05`, NP wins only a small number of images in PSNR but rescues a major SITCOM failure.
- At `sigma_y=0.10`, NP wins PSNR on most images.
- At `sigma_y=0.20`, NP wins PSNR on all images, though absolute quality is low.

## Candidate count ablation

Goal:

```text
Test whether doubled candidates per reconstruction can replace independent restarts.
```

Old setting:

```text
soft=5, hard=1, best-of-4
```

New settings:

```text
soft=10, hard=1, best-of-2
soft=10, hard=2, best-of-2
```

Results at `sigma_y=0.05`:

| Setting | Best-of-k | PSNR mean | Median | Min | SSIM | LPIPS | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old soft=5, hard=1 | best-of-2 | 28.531 | 29.280 | 17.528 | 0.808 | 0.249 | 1 | 1 |
| old soft=5, hard=1 | best-of-4 | 29.424 | 29.723 | 27.185 | 0.832 | 0.224 | 0 | 0 |
| new soft=10, hard=1 | best-of-2 | 27.656 | 29.356 | 13.727 | 0.777 | 0.278 | 3 | 3 |
| new soft=10, hard=2 | best-of-2 | 26.855 | 26.824 | 24.902 | 0.790 | 0.194 | 0 | 1 |

Interpretation:

- Doubling candidates did not replace independent restarts.
- `soft=10, hard=1` rescued some previous failures but created new catastrophic failures.
- `soft=10, hard=2` reduced catastrophic failures but substantially lowered reconstruction quality.
- More candidates can amplify a flawed selection criterion.

Per-image findings for `soft=10, hard=1`:

Major rescues vs old best-of-2:

```text
00018: +11.47 dB
00028: +4.15 dB
00034: +3.02 dB
00027: +2.88 dB
```

New failures:

```text
00005: -16.30 dB
00000: -16.02 dB
00007: -11.34 dB
```

Conclusion:

```text
More candidates change which images fail, but do not reliably lower failure probability.
Independent restarts remain more reliable than simply increasing candidates.
```

## Score-mode experiments

Motivation:

If more candidates make the result worse, the candidate-selection score is likely not well aligned with final reconstruction quality.

Four score modes were tested:

```text
S1: lf
    original low-frequency magnitude score.

S2: prev_l2
    low-frequency score plus penalty for moving too far from previous x0.

S3: consensus_l2
    low-frequency score plus candidate outlier penalty.

S4: huber_lf
    robust Huber low-frequency score.
```

Setup:

```text
sigma_y=0.05
25 images
2 seeds
score_radius=0.6
proj_radius=0.2
proj_start=300
soft=5
hard=1
```

Results:

| Setting | Best-of-2 PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| S1 original LF | 28.531 | 29.280 | 17.528 | 0.808 | 1 | 1 |
| S2 prev_l2 lambda=0.25 | 28.560 | 29.280 | 13.828 | 0.810 | 1 | 1 |
| S3 consensus_l2 | 27.176 | 29.280 | 11.898 | 0.768 | 3 | 3 |
| S4 Huber LF | 27.438 | 29.209 | 13.832 | 0.767 | 3 | 3 |
| old S1 best-of-4 | 29.424 | 29.723 | 27.185 | 0.832 | 0 | 0 |

Conclusion:

- S2 had slight positive signal, but was not robust.
- S3 and S4 were clearly worse in this form.
- Score changes can rescue some images but create other failures.

## Focused S2 lambda sweep

Lambdas tested:

```text
0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15
```

Best static lambda:

```text
lambda = 0.01
```

Results:

| Setting | Mean PSNR | Median | Min | SSIM | Below 20 | Below 25 |
|---|---:|---:|---:|---:|---:|---:|
| S1 LF best-of-2 | 28.531 | 29.280 | 17.528 | 0.808 | 1 | 1 |
| S2 lambda=0.005 | 28.460 | 29.582 | 17.739 | 0.801 | 1 | 2 |
| S2 lambda=0.01 | 28.818 | 29.582 | 19.269 | 0.813 | 1 | 1 |
| S2 lambda=0.02 | 28.706 | 29.280 | 19.624 | 0.809 | 1 | 1 |
| S2 lambda=0.05 | 28.544 | 29.280 | 13.836 | 0.809 | 1 | 1 |
| S1 LF best-of-4 | 29.424 | 29.723 | 27.185 | 0.832 | 0 | 0 |

S2 `lambda=0.01` rescues:

```text
00018: 17.528 -> 28.963
00028: 25.877 -> 29.996
00034: 27.596 -> 29.690
```

but creates a new failure:

```text
00007: 28.783 -> 19.269
```

Oracle over lambdas:

```text
mean PSNR ≈ 29.294
median ≈ 29.631
min ≈ 25.103
below20 = 0
below25 = 0
```

Conclusion:

```text
S2 contains useful signal, but a fixed global lambda is not robust enough.
The direction is worth exploring only as part of a single principled adaptive score, not as an oracle over lambdas.
```

## Current methodological conclusion

The evidence supports this view:

```text
Noise Picking is useful as a conservative branch-selection / measurement-guidance principle.
The current simple greedy low-frequency score is fragile.
Small low-frequency projection is crucial.
Broad projection is harmful.
Increasing candidate count amplifies score errors.
Score regularization can rescue some failures but may move failures elsewhere.
NP is more robust than SITCOM-ODE under higher measurement noise in PSNR/SSIM.
```

The project should now prioritize one well-defined, reliable solver rather than chasing an averaged best-of-k table.
