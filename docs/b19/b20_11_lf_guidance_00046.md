# B20.11 — Low-frequency measurement guidance on hard image 00046

Date: 2026-07-01  
Branch: `b19_solver_integration`  
PAC repo path: `/egr/research-pac/huang248/pr_diffusion_b19_solver`  
Output base: `/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver`  
Image: `00046`  
Measurement seed: `5001`  
Task: phase retrieval, sigma `0.05`  
DAPS setting: pixel task, final-only, `NUM_RUNS=1`, `diff_steps=5`

## Purpose

B20.10 showed that annealing schedule changes basin entry for hard image `00046`, but schedule variation alone did not give a reliable seed-independent fix. B20.11 tests an active clean-free basin-entry intervention: early low-frequency measurement guidance inside the DAPS trajectory.

The patch is inserted in `external/daps/sampler.py` after the MCMC measurement update produces `x0y` and before the next `xt` is formed. The intervention is off by default and is controlled by environment variables.

```bash
B20_LF_ENABLE=1
B20_LF_ALPHA={0.10,0.25,0.50}
B20_LF_FRAC=0.35
B20_LF_RADIUS_FRAC=0.12
```

## Experiment panel

Variants:

```text
base, lf010, lf025, lf050
```

Schedules:

```text
ann350_diff5, ann400_diff5
```

Seeds:

```text
6200--6455 inclusive, 256 seeds
```

Total completed runs:

```text
256 seeds × 2 schedules × 4 variants = 2048 final-only DAPS runs
```

Completion status:

```text
base panel: done 512, fail 0
LF panel:   done 1536, fail 0
analyzer:   rows 2048, expected 2048, missing 0
```

Main output files:

```text
B20_11_00046_meas5001_lf_guidance_long.csv
B20_11_00046_meas5001_variant_schedule_summary.csv
B20_11_00046_meas5001_base_vs_lf_pairs.csv
B20_11_00046_meas5001_base_vs_lf_summary.csv
B20_11_00046_meas5001_seed_schedule_oracle.csv
B20_11_00046_meas5001_seed_oracle_summary.csv
B20_11_00046_meas5001_global_oracle_summary.csv
B20_11_00046_meas5001_portfolio_replay.csv
```

## Variant by schedule summary

Success threshold: `PSNR >= 25`.

| variant | ann | good25 / 256 | mean PSNR | median PSNR | min PSNR | max PSNR |
|---|---:|---:|---:|---:|---:|---:|
| base | 350 | 11 | 15.336922 | 15.873600 | 9.638234 | 31.389044 |
| lf010 | 350 | 11 | 15.478444 | 15.968635 | 9.640988 | 31.348343 |
| lf025 | 350 | 14 | 15.547786 | 15.920928 | 9.619759 | 31.348116 |
| lf050 | 350 | 12 | 15.086710 | 15.472594 | 9.509188 | 31.348103 |
| base | 400 | 30 | 16.159156 | 15.730723 | 9.644577 | 31.468611 |
| lf010 | 400 | 30 | 16.285558 | 15.974709 | 9.660698 | 31.474945 |
| lf025 | 400 | 25 | 16.142918 | 16.199777 | 9.678291 | 31.474506 |
| lf050 | 400 | 33 | 16.427668 | 16.002269 | 9.644546 | 31.474632 |

Best single option:

```text
lf050_ann400: 33 / 256 good seeds
```

Best base single option:

```text
base_ann400: 30 / 256 good seeds
```

At fixed schedule and fixed LF strength, the gain is modest but real.

## Seed-level oracle summary

Oracle over base schedules only:

```text
base_any_good = 40 / 256
```

Oracle over LF schedules and LF strengths only:

```text
lf_any_good = 60 / 256
```

Oracle over base plus LF:

```text
all_any_good = 65 / 256
```

Seed-level changes relative to base:

```text
lf_rescue_seed = 25
lf_lost_seed   = 5
```

Mean best-over-schedules PSNR:

```text
base_best_mean        = 17.863979
lf_best_mean          = 19.811802
all_variant_best_mean = 20.253194
```

This is the main B20.11 result: low-frequency measurement guidance increases basin-hit coverage on hard image `00046` and is complementary to schedule variation.

## Portfolio replay

Full 8-option oracle:

```text
base/lf010/lf025/lf050 × ann350/ann400 = 65 / 256
```

Best one-attempt policy:

```text
lf050_ann400: 33 / 256
```

Best two-attempt policy:

```text
lf025_ann350 + lf050_ann400: 43 / 256
```

Base two-schedule oracle:

```text
base_ann350 + base_ann400: 40 / 256
```

Best three-attempt policies:

```text
lf010_ann400 + lf025_ann350 + lf050_ann400: 52 / 256
base_ann400 + lf025_ann350 + lf050_ann400: 52 / 256
```

The second three-attempt policy is preferred scientifically because it retains the best base arm and adds two complementary LF interventions.

Best four-attempt policy:

```text
base_ann400 + lf010_ann400 + lf025_ann350 + lf050_ann400: 57 / 256
```

This captures about `57 / 65 = 87.7%` of the full 8-option oracle with half as many attempts.

## Interpretation

B20.11 validates low-frequency measurement guidance as a real basin-entry intervention.

B20.10 conclusion:

```text
Annealing schedule affects basin entry, but schedule alone is not reliable.
```

B20.11 conclusion:

```text
Early low-frequency measurement guidance also affects basin entry and rescues additional seeds.
```

The intervention is not uniformly beneficial. It creates both rescues and harms, so it is not yet a deterministic health-preserving improvement. The strongest current story is that LF guidance creates complementary basins and can be compressed into a small 3--4 attempt portfolio.

## Recommended microscope seeds

Rescues:

```text
6347: base 9.76, best LF 31.22
6426: base 9.99, best LF 31.26
6412: base 10.08, best LF 31.27
6208: base 15.30, best LF 31.32
```

Harms:

```text
6312: base 31.34, best LF 16.44
6329: base 31.42, best LF 18.12
6230: base 31.39, best LF 18.19
```

## Recommended multi-image validation

Do not overfit to image `00046`. Next validation should test compact portfolios on a broader hard-image panel.

Suggested first validation portfolio:

```text
base_ann400
lf025_ann350
lf050_ann400
```

Suggested stronger validation portfolio:

```text
base_ann400
lf010_ann400
lf025_ann350
lf050_ann400
```

Candidate hard images from previous fixed-budget/oracle-failure analysis:

```text
00046, 00971, 00480, 00171, 00746
```
