# Progress report: FFHQ phase retrieval, tuned Noise Picking, and SITCOM-ODE comparison

This document records the current project state after the FFHQ-25 guided NP benchmarking, parameter tuning, radius ablations, score-mode experiments, and SITCOM-ODE comparisons.

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
- Main noise level for method tuning: sigma_y = 0.05.
- Main NP evaluation convention:
  - report image-level best-of-k only when k is explicitly stated;
  - also track all-run mean, median, failure counts, and per-image worst cases.

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

The tiny numeric winner in one top-8 confirmation was:

```text
score_radius = 0.2
proj_radius  = 0.2
proj_start   = 500
soft_k       = 8
hard_k       = 1
```

but it was only about 0.016 dB better than the cheaper practical setting, so the practical setting is preferred for ongoing development.

## NP parameter tuning and top-8 confirmation

A broad FFHQ tuning screen tested score radius, projection radius, projection starts, and candidate schedules.

Top-8 confirmation, using 25 images and 4 seeds at sigma_y=0.05, found:

| Config                                         | Best-of-4 PSNR |   SSIM |  LPIPS | Worst best-of-4 PSNR |
| ---------------------------------------------- | -------------: | -----: | -----: | -------------------: |
| score=0.2, proj=0.2, start=500, soft=8, hard=1 |         29.440 | 0.8320 | 0.2242 |               27.243 |
| score=0.6, proj=0.2, start=300, soft=5, hard=1 |         29.424 | 0.8319 | 0.2241 |               27.185 |

These two settings are effectively tied. The second one is much cheaper and became the main practical setting.

## Radius ablations

A two-stage projection-radius schedule was tested. Late broadening of hard projection radius was decisively harmful.

Best schedule (still poor): best-of-2 PSNR ≈ 26.71, versus constant-radius baseline best-of-2 PSNR ≈ 29.13.

Conclusion: broad hard measurement replacement conflicts with the learned prior, even late.

## SITCOM-ODE comparison at sigma_y = 0.05

Same-split results:

- SITCOM-ODE: PSNR ≈ 29.690, SSIM ≈ 0.733, LPIPS ≈ 0.185
- NP practical setting: PSNR ≈ 29.424, SSIM ≈ 0.832, LPIPS ≈ 0.224

NP does not beat SITCOM-ODE in PSNR/LPIPS at sigma_y=0.05, but is stronger in SSIM and shows complementary failure behavior.

## Noise-level comparison vs SITCOM-ODE

| sigma_y | NP raw PSNR | SITCOM-ODE PSNR | NP - SITCOM | NP SSIM | SITCOM SSIM | NP LPIPS | SITCOM LPIPS |
| ------: | ----------: | --------------: | ----------: | ------: | ----------: | -------: | -----------: |
|    0.00 |      32.372 |          35.852 |      -3.480 |   0.924 |       0.935 |    0.125 |        0.067 |
|    0.01 |      33.444 |          35.736 |      -2.292 |   0.947 |       0.933 |    0.100 |        0.067 |
|    0.05 |      29.424 |          29.690 |      -0.266 |   0.832 |       0.733 |    0.224 |        0.185 |
|    0.10 |      24.203 |          23.302 |      +0.901 |   0.626 |       0.411 |    0.491 |        0.395 |
|    0.20 |      19.606 |          17.429 |      +2.177 |   0.400 |       0.168 |    0.731 |        0.629 |
|    0.50 |      13.034 |          11.121 |      +1.913 |   0.143 |       0.036 |    0.834 |        0.777 |

SITCOM dominates low noise; NP becomes stronger in PSNR/SSIM for sigma_y >= 0.10; SITCOM keeps LPIPS advantage.

## Candidate-count and score-mode ablations

- Doubling candidates did not replace independent restarts.
- More candidates can amplify score errors.
- `prev_l2` score regularization contains useful signal, but fixed global lambda is not robust.

## Methodological conclusion

Noise Picking is useful as conservative branch-selection / measurement-guidance. Next phase should prioritize single-solver reliability and failure reduction, not only best-of-k averages.
