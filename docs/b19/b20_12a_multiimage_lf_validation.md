# B20.12A — Held-out multi-image validation of compact LF guidance portfolio

Date: 2026-07-01  
Branch: `b19_solver_integration`  
PAC repo path: `/egr/research-pac/huang248/pr_diffusion_b19_solver`  
Output base: `/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver`  
Task: phase retrieval, sigma `0.05`  
Measurement seed: `5001`  
DAPS setting: pixel task, final-only, `NUM_RUNS=1`, `diff_steps=5`

## Purpose

B20.11 developed low-frequency measurement guidance on hard image `00046`. It showed that LF guidance can increase basin-hit coverage and that most of the gain can be compressed into a small portfolio.

B20.12A tests whether the compact B20.11 portfolio generalizes beyond `00046`, rather than being overfit to that development image.

## Held-out images

Validation images:

```text
00971, 00480, 00171, 00746
```

Preflight confirmed that both DAPS datasets and measurement files exist for all four images.

## Portfolio tested

The compact three-arm portfolio was selected from B20.11, but this validation excludes the B20.11 development image `00046`.

```text
base_ann400
lf025_ann350
lf050_ann400
```

Seeds:

```text
6500--6563 inclusive, 64 seeds per image
```

Total runs:

```text
4 images × 64 seeds × 3 arms = 768 final-only DAPS runs
```

Completion and analyzer status:

```text
rows = 768
expected = 768
missing = 0
```

Main outputs:

```text
B20_12A_multiimage_heldout4_64seed_3arm_long.csv
B20_12A_multiimage_heldout4_64seed_3arm_by_arm.csv
B20_12A_multiimage_heldout4_64seed_3arm_per_seed.csv
B20_12A_multiimage_heldout4_64seed_3arm_summary.csv
```

## Per-arm result

Success threshold: `PSNR >= 25`.

| image | arm | good25 / 64 | good rate | mean PSNR | median PSNR | min PSNR | q10 PSNR | max PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 00171 | base_ann400 | 16 | 0.250000 | 15.831052 | 11.473399 | 10.819324 | 10.956750 | 30.646481 |
| 00171 | lf025_ann350 | 16 | 0.250000 | 16.093247 | 11.606203 | 10.927096 | 11.072431 | 30.986893 |
| 00171 | lf050_ann400 | 17 | 0.265625 | 16.178920 | 11.426324 | 10.822182 | 10.979501 | 30.739260 |
| 00480 | base_ann400 | 39 | 0.609375 | 21.839559 | 29.078721 | 8.469062 | 8.480316 | 29.525202 |
| 00480 | lf025_ann350 | 27 | 0.421875 | 18.468099 | 13.830135 | 8.457829 | 8.479972 | 29.508572 |
| 00480 | lf050_ann400 | 37 | 0.578125 | 21.174775 | 29.028076 | 8.460758 | 8.477485 | 29.525213 |
| 00746 | base_ann400 | 13 | 0.203125 | 20.677261 | 20.143903 | 11.640532 | 12.245710 | 31.712208 |
| 00746 | lf025_ann350 | 14 | 0.218750 | 21.204549 | 20.277871 | 11.506013 | 12.108051 | 31.581005 |
| 00746 | lf050_ann400 | 14 | 0.218750 | 20.470103 | 20.101595 | 10.896402 | 11.852054 | 31.712048 |
| 00971 | base_ann400 | 51 | 0.796875 | 27.268854 | 32.116947 | 8.136176 | 8.155338 | 32.301220 |
| 00971 | lf025_ann350 | 45 | 0.703125 | 24.983189 | 32.042786 | 8.142923 | 8.155382 | 32.253044 |
| 00971 | lf050_ann400 | 40 | 0.625000 | 23.146220 | 32.057566 | 8.135796 | 8.148868 | 32.300724 |

At the single-arm level, LF does not uniformly dominate base. In particular, `base_ann400` is stronger than the two LF arms on images `00480` and `00971`. Therefore LF guidance should not currently replace the base arm.

## Portfolio result

| image | seeds | base any good | LF any good | portfolio any good | LF rescues | LF lost | base best mean | LF best mean | portfolio best mean | base min | portfolio min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00171 | 64 | 16 | 27 | 28 | 12 | 1 | 15.831052 | 19.329978 | 19.641661 | 10.819324 | 11.121949 |
| 00480 | 64 | 39 | 44 | 47 | 8 | 3 | 21.839559 | 23.562453 | 24.529349 | 8.469062 | 8.477345 |
| 00746 | 64 | 13 | 20 | 22 | 9 | 2 | 20.677261 | 23.088151 | 23.694963 | 11.640532 | 12.071914 |
| 00971 | 64 | 51 | 54 | 56 | 5 | 2 | 27.268854 | 28.394919 | 29.160414 | 8.136176 | 8.152317 |
| ALL_HELDOUT | 256 | 119 | 145 | 153 | 34 | 8 | 21.404181 | 23.593875 | 24.256597 | 8.136176 | 8.152317 |

Main held-out result:

```text
base only:      119 / 256 = 46.5%
base + LF:      153 / 256 = 59.8%
absolute gain:   34 / 256 = +13.3 percentage points
relative gain:   153 / 119 ≈ 1.29× successful seeds
```

The compact portfolio improves every held-out image:

```text
00171: 16 -> 28  (+12)
00480: 39 -> 47  (+8)
00746: 13 -> 22  (+9)
00971: 51 -> 56  (+5)
```

## Best-arm counts

Best arm per seed by PSNR:

| image | best arm | count |
|---|---|---:|
| 00171 | base_ann400 | 13 |
| 00171 | lf025_ann350 | 35 |
| 00171 | lf050_ann400 | 16 |
| 00480 | base_ann400 | 16 |
| 00480 | lf025_ann350 | 26 |
| 00480 | lf050_ann400 | 22 |
| 00746 | base_ann400 | 21 |
| 00746 | lf025_ann350 | 30 |
| 00746 | lf050_ann400 | 13 |
| 00971 | base_ann400 | 25 |
| 00971 | lf025_ann350 | 20 |
| 00971 | lf050_ann400 | 19 |

Although base can be the best single arm on some images, LF arms frequently produce the best reconstruction for individual seeds. This supports the interpretation that LF guidance opens complementary basins rather than uniformly improving all base trajectories.

## Interpretation

B20.12A is a positive held-out validation of the B20.11 direction.

The correct claim is not that LF guidance always improves DAPS, nor that it should replace the base schedule. The correct claim is:

> Early low-frequency measurement guidance is complementary to the base DAPS trajectory. In a compact three-arm portfolio, it consistently expands the set of successful seeds on held-out hard images.

This makes the direction worth taking seriously, but it should not be treated as the only possible direction. Other interventions, schedule designs, selection policies, or repair mechanisms may still be competitive or better.

## Recommended next experiment

Run a broader hard-image validation with the same fixed three-arm portfolio:

```text
base_ann400
lf025_ann350
lf050_ann400
```

Suggested design:

```text
more hard images × 64 seeds × 3 arms
```

If this broader panel is also positive, then the LF-guidance direction should be elevated from a prototype intervention to a serious method-development branch. The next step would then be either (i) deeper microscope/trajectory analysis to understand when LF helps or harms, or (ii) method development around adaptive LF guidance and portfolio selection.
