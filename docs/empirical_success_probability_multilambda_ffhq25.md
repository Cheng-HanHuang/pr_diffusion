# Empirical success probability: FFHQ-25 multi-lambda selector candidate pool

Updated: 2026-05-23

## Scope

This report aggregates the full-25 multi-lambda validation traces with the current default setting:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
configs      = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
raw success  = raw PSNR >= 25 dB
```

Included full-25 seed traces: `100,101,102,103`, `104,105`, `106,107`, and `108,109`, giving seeds `100` through `109` once each. Focused-subset runs are excluded because their filtered image ordering changes the measurement-noise indexing.

## Executive summary

- Total candidates: **1000** = 25 images × 10 seeds × 4 configs.
- Successful raw candidates: **729 / 1000 = 72.9%**.
- Candidate generation, not selector error, is the present bottleneck: the failing two-seed validations were oracle failures.
- The hardest image is **00028**, with only **6 / 40** successful raw candidates and only **4 / 10** seeds having any successful config.
- The recurring hard set is **00028, 00005, 00034, 00013, 00007, 00027, 00000**. These should drive targeted recovery and adaptive-compute experiments.

## Success by config family

| config_family | n | successes | success_rate | mean_psnr | min_psnr | max_psnr |
|---|---:|---:|---:|---:|---:|---:|
| LF | 250 | 175 | 0.700 | 24.013 | 5.075 | 31.347 |
| S2_0.005 | 250 | 180 | 0.720 | 24.442 | 5.685 | 31.382 |
| S2_0.02 | 250 | 182 | 0.728 | 24.724 | 5.685 | 31.382 |
| S2_0.05 | 250 | 192 | 0.768 | 25.308 | 5.076 | 31.382 |

Interpretation: `S2 lambda=0.05` has the highest overall candidate success rate, but it is not sufficient alone because it fails on images such as `00028`. Complementarity remains essential.

## Candidate availability by seed set

| seed_set | n_seeds | images_with_candidate_success | images_without_candidate_success | worst_best_psnr | failed_images |
|---|---:|---:|---:|---:|---|
| 100,101 | 2 | 24 | 1 | 13.836 | 00028 |
| 102,103 | 2 | 25 | 0 | 27.182 |  |
| 104,105 | 2 | 24 | 1 | 10.819 | 00005 |
| 106,107 | 2 | 24 | 1 | 16.691 | 00028 |
| 108,109 | 2 | 24 | 1 | 24.442 | 00013 |
| 100,101,102,103 | 4 | 25 | 0 | 27.185 |  |
| 100-109 | 10 | 25 | 0 | 27.289 |  |

Interpretation: two-seed sets often leave one image with no successful candidate. Four seeds `100,101,102,103` and the aggregate `100-109` contain at least one successful candidate for every image.

## Per-seed candidate availability

| seed | candidate_success_rate | mean_psnr | min_psnr | images_with_any_success | images_failed |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.740 | 24.909 | 5.076 | 22 | 3 |
| 101 | 0.700 | 23.469 | 5.076 | 22 | 3 |
| 102 | 0.750 | 24.739 | 5.679 | 22 | 3 |
| 103 | 0.690 | 23.732 | 7.939 | 22 | 3 |
| 104 | 0.800 | 25.952 | 9.119 | 23 | 2 |
| 105 | 0.740 | 24.458 | 5.075 | 23 | 2 |
| 106 | 0.710 | 24.822 | 5.679 | 22 | 3 |
| 107 | 0.600 | 22.795 | 5.685 | 17 | 8 |
| 108 | 0.820 | 26.332 | 8.906 | 23 | 2 |
| 109 | 0.740 | 25.010 | 7.823 | 21 | 4 |

Interpretation: no single seed gives successful candidates for all images. This confirms that one seed plus multiple lambda branches is not enough for the current method.

## Hardest images by candidate success rate

| image_id | n_success | success_rate | n_seeds_with_any_success | seed_success_rate | best_raw_psnr | best_seed | best_config | LF | S2_0.005 | S2_0.02 | S2_0.05 | risk_class |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 00028 | 6 | 0.150 | 4 | 0.4 | 30.1407 | 103 | LF | 2 | 2 | 0 | 2 | hard (<25%) |
| 00005 | 12 | 0.300 | 6 | 0.6 | 30.3195 | 108 | S2_0.02 | 3 | 0 | 4 | 5 | medium (25-50%) |
| 00034 | 17 | 0.425 | 6 | 0.6 | 30.6282 | 104 | S2_0.005 | 3 | 5 | 5 | 4 | medium (25-50%) |
| 00007 | 18 | 0.450 | 9 | 0.9 | 28.8777 | 102 | S2_0.005 | 5 | 5 | 1 | 7 | medium (25-50%) |
| 00013 | 18 | 0.450 | 7 | 0.7 | 28.0316 | 106 | S2_0.02 | 4 | 4 | 4 | 6 | medium (25-50%) |
| 00000 | 19 | 0.475 | 9 | 0.9 | 30.2157 | 107 | LF | 4 | 4 | 5 | 6 | medium (25-50%) |
| 00027 | 19 | 0.475 | 5 | 0.5 | 28.2196 | 106 | S2_0.02 | 5 | 5 | 4 | 5 | medium (25-50%) |
| 00032 | 24 | 0.600 | 7 | 0.7 | 29.2001 | 106 | S2_0.05 | 6 | 6 | 5 | 7 | easy-ish (50-75%) |
| 00009 | 26 | 0.650 | 10 | 1.0 | 29.9078 | 101 | S2_0.005 | 5 | 6 | 8 | 7 | easy-ish (50-75%) |
| 00014 | 28 | 0.700 | 8 | 0.8 | 29.3483 | 108 | S2_0.05 | 6 | 6 | 8 | 8 | easy-ish (50-75%) |

## Full per-image table

| image_id | n_success | success_rate | n_seeds_with_any_success | seed_success_rate | best_raw_psnr | best_seed | best_config | LF | S2_0.005 | S2_0.02 | S2_0.05 | risk_class |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 00000 | 19 | 0.475 | 9 | 0.9 | 30.2157 | 107 | LF | 4 | 4 | 5 | 6 | medium (25-50%) |
| 00004 | 38 | 0.950 | 10 | 1.0 | 27.3780 | 102 | S2_0.05 | 10 | 10 | 10 | 8 | easy (>75%) |
| 00005 | 12 | 0.300 | 6 | 0.6 | 30.3195 | 108 | S2_0.02 | 3 | 0 | 4 | 5 | medium (25-50%) |
| 00007 | 18 | 0.450 | 9 | 0.9 | 28.8777 | 102 | S2_0.005 | 5 | 5 | 1 | 7 | medium (25-50%) |
| 00008 | 36 | 0.900 | 10 | 1.0 | 29.8435 | 106 | LF | 9 | 9 | 10 | 8 | easy (>75%) |
| 00009 | 26 | 0.650 | 10 | 1.0 | 29.9078 | 101 | S2_0.005 | 5 | 6 | 8 | 7 | easy-ish (50-75%) |
| 00010 | 36 | 0.900 | 9 | 0.9 | 28.4027 | 108 | LF | 9 | 9 | 9 | 9 | easy (>75%) |
| 00011 | 40 | 1.000 | 10 | 1.0 | 29.5976 | 106 | S2_0.005 | 10 | 10 | 10 | 10 | easy (>75%) |
| 00012 | 40 | 1.000 | 10 | 1.0 | 29.2379 | 104 | S2_0.05 | 10 | 10 | 10 | 10 | easy (>75%) |
| 00013 | 18 | 0.450 | 7 | 0.7 | 28.0316 | 106 | S2_0.02 | 4 | 4 | 4 | 6 | medium (25-50%) |
| 00014 | 28 | 0.700 | 8 | 0.8 | 29.3483 | 108 | S2_0.05 | 6 | 6 | 8 | 8 | easy-ish (50-75%) |
| 00015 | 33 | 0.825 | 9 | 0.9 | 29.9640 | 103 | LF | 8 | 8 | 8 | 9 | easy (>75%) |
| 00016 | 40 | 1.000 | 10 | 1.0 | 29.8388 | 100 | S2_0.02 | 10 | 10 | 10 | 10 | easy (>75%) |
| 00017 | 28 | 0.700 | 9 | 0.9 | 31.3821 | 102 | S2_0.05 | 6 | 8 | 8 | 6 | easy-ish (50-75%) |
| 00018 | 29 | 0.725 | 9 | 0.9 | 29.1442 | 102 | S2_0.02 | 6 | 8 | 8 | 7 | easy-ish (50-75%) |
| 00019 | 33 | 0.825 | 10 | 1.0 | 30.0659 | 105 | S2_0.05 | 8 | 8 | 8 | 9 | easy (>75%) |
| 00020 | 40 | 1.000 | 10 | 1.0 | 29.8550 | 109 | S2_0.05 | 10 | 10 | 10 | 10 | easy (>75%) |
| 00025 | 39 | 0.975 | 10 | 1.0 | 27.2886 | 105 | S2_0.005 | 10 | 10 | 9 | 10 | easy (>75%) |
| 00027 | 19 | 0.475 | 5 | 0.5 | 28.2196 | 106 | S2_0.02 | 5 | 5 | 4 | 5 | medium (25-50%) |
| 00028 | 6 | 0.150 | 4 | 0.4 | 30.1407 | 103 | LF | 2 | 2 | 0 | 2 | hard (<25%) |
| 00029 | 31 | 0.775 | 10 | 1.0 | 30.8649 | 101 | S2_0.05 | 6 | 8 | 8 | 9 | easy (>75%) |
| 00032 | 24 | 0.600 | 7 | 0.7 | 29.2001 | 106 | S2_0.05 | 6 | 6 | 5 | 7 | easy-ish (50-75%) |
| 00034 | 17 | 0.425 | 6 | 0.6 | 30.6282 | 104 | S2_0.005 | 3 | 5 | 5 | 4 | medium (25-50%) |
| 00037 | 40 | 1.000 | 10 | 1.0 | 30.7688 | 103 | S2_0.005 | 10 | 10 | 10 | 10 | easy (>75%) |
| 00039 | 39 | 0.975 | 10 | 1.0 | 29.0939 | 108 | S2_0.005 | 10 | 9 | 10 | 10 | easy (>75%) |

Column meanings:

- `n_success`: successful raw candidates among 40 = 10 seeds × 4 configs.
- `success_rate`: `n_success / 40`.
- `n_seeds_with_any_success`: number of seeds for which at least one of the four configs succeeds.
- `LF`, `S2_0.005`, `S2_0.02`, `S2_0.05`: number of successful seeds out of 10 for that config.

## Algorithmic implications

The empirical decomposition now looks like:

```text
failure probability ≈ P(no good candidate generated) + P(selector fails | good candidate exists)
```

The completed runs suggest the second term is comparatively small: selected failures usually coincide with oracle failures. The first term is the bottleneck.

Next steps suggested by this table:

1. Simulate adaptive compute policies from the existing traces: start with 1–2 seeds and add seeds only for high-risk images.
2. Target recovery diagnostics at `00028`, `00005`, `00013`, and `00034`, using `00007`, `00027`, and `00000` as additional guard/hard cases.
3. Develop an in-loop adaptive multi-lambda scoring method that increases candidate-generation success, not merely post-hoc selection.
4. Start a theory note around candidate-generation probability and selector consistency.
