# Progress report: FFHQ phase retrieval, tuned Noise Picking, and SITCOM-ODE comparison

Updated: 2026-05-12

This document records the project state after FFHQ-25 guided NP benchmarking, tuning, radius ablations, score-mode experiments, candidate-count experiments, and SITCOM-ODE comparisons.

## Current practical NP setting

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

## Key quantitative summary

- NP practical (sigma_y=0.05, FFHQ-25, best-of-4): PSNR ≈ 29.424, SSIM ≈ 0.8319, LPIPS ≈ 0.2241.
- Alt NP (score=0.2,start=500,soft=8): PSNR ≈ 29.440 (only +0.016 dB).
- SITCOM-ODE at sigma_y=0.05: PSNR ≈ 29.690, SSIM ≈ 0.733, LPIPS ≈ 0.185.
- NP wins robustness at higher noise in PSNR/SSIM (>=0.10), SITCOM stays better in LPIPS.

## Important conclusions

1. `proj_radius=0.2` is essential.
2. Late radius broadening hurts substantially.
3. More candidates can amplify score errors.
4. `prev_l2` has signal but fixed lambda is not robust.
5. Next goal is reliability and failure reduction, not only best-of-k.
