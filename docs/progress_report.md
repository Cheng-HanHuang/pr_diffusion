# Progress report: multi-lambda selector validation and next robustness goals

Updated: 2026-05-23

This report records the current state after the focused and full-25 multi-lambda selector experiments, plus the later seed-pair validation runs.  The previous four-GPU LF/S2 selector and lambda diagnostic report is archived in `docs/historical/progress_report_archived_20260523_before_full25_multilambda_validation.md`.

A separate empirical candidate-generation table is maintained in `docs/empirical_success_probability_multilambda_ffhq25.md`.  That file aggregates all full-25 multi-lambda traces across seeds `100`--`109` and should be updated whenever new seed/config validation traces are added.

## Executive summary

The current best method remains a **multi-lambda LF/S2 selector**:

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

The strongest positive validation is still the full FFHQ-25 run with seeds `102,103`:

```text
selected_config_seed_by_selector, raw:
  mean PSNR ≈ 29.305
  median    ≈ 29.584
  min       ≈ 26.230
  SSIM      ≈ 0.830
  below20   = 0
  below25   = 0
```

However, later two-seed validation pairs show that this is **not yet an always-successful two-seed method**:

```text
seeds 104,105:
  selected mean ≈ 28.496
  min ≈ 9.126
  below25 = 1
  failure: 00005
  oracle also fails, so this is candidate availability failure

seeds 106,107:
  selected mean ≈ 28.615
  min ≈ 11.997
  below25 = 1
  failure: 00028
  oracle also fails, so this is candidate availability failure

seeds 108,109:
  selected mean ≈ 29.227
  min ≈ 24.442
  below25 = 1
  failure: 00013
  oracle also fails, so this is candidate availability failure
```

Four seeds `100,101,102,103` pass strongly:

```text
selected_config_seed_by_selector, raw:
  mean ≈ 29.361
  min ≈ 27.081
  below25 = 0
```

The empirical success-probability table over all seeds `100`--`109` gives:

```text
1000 total candidates = 25 images × 10 seeds × 4 configs
729 successful raw candidates above 25 dB
candidate success rate = 72.9%
hardest image: 00028, with 6/40 successful candidates
recurring hard set: 00028, 00005, 00034, 00013, 00007, 00027, 00000
```

The current conclusion is precise:

```text
Multi-lambda score/config selection is a real improvement over fixed LF/S2.
The selector is usually near-oracle over the available candidate pool.
But two random seeds still do not always provide a good candidate.
Therefore the remaining problem is candidate generation / seed diversity, not primarily post-hoc selection.
```

This is meaningful progress, but it is still a multi-run selector.  It is not yet a single-run always-successful algorithm.

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
all-run reliability
candidate-generation success rate when enough traces are available
```

Promotion target remains:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

## Background before multi-lambda validation

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

The focused target was passed comfortably.  No single config is reliable, but the selector avoids their complementary failures.

## Full FFHQ-25 multi-lambda selector: seeds 102,103

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 29.305 | 29.584 | 26.230 | 0.830 | 0 | 0 |
| global_run_by_selector | 29.288 | 29.584 | 26.230 | 0.830 | 0 | 0 |
| selected_config_bestofk | 29.416 | 29.723 | 27.182 | 0.832 | 0 | 0 |
| oracle_all_candidates | 29.417 | 29.723 | 27.182 | 0.832 | 0 | 0 |

This run showed that multi-lambda selection fixes the exact seed pair where fixed LF/S2 failed.

Key examples:

| Image | LF | S2 λ=0.005 | S2 λ=0.02 | S2 λ=0.05 | Selected behavior |
|---|---:|---:|---:|---:|---|
| 00005 | 9.130 | 9.130 | 9.130 | 30.260 | selected high-lambda branch |
| 00014 | 13.706 | 13.706 | 29.312 | 29.312 | selected high-lambda branch |
| 00028 | 30.141 | 13.792 | 13.792 | 13.792 | selected LF branch |
| 00032 | 9.864 | 9.864 | 29.109 | 29.109 | selected high-lambda branch |

Config selection is nearly oracle-level:

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

## Full FFHQ-25 multi-lambda selector: four seeds 100,101,102,103

### Raw results

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 |
|---|---:|---:|---:|---:|---:|---:|
| selected_config_seed_by_selector | 29.361 | 29.584 | 27.081 | 0.831 | 0 | 0 |
| global_run_by_selector | 29.398 | 29.584 | 27.081 | 0.831 | 0 | 0 |
| selected_config_bestofk | 29.454 | 29.723 | 27.185 | 0.832 | 0 | 0 |
| oracle_all_candidates | 29.459 | 29.723 | 27.185 | 0.832 | 0 | 0 |

This is a clean pass and is currently the strongest selected multi-run baseline.  However, single-seed simulations from the same traces all fail even when all four configs are allowed:

| Seed | Selected mean | Min | Images <20 | Images <25 | Oracle min | Oracle <25 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 27.510 | 11.740 | 3 | 3 | 11.917 | 3 |
| 101 | 26.813 | 9.128 | 3 | 3 | 9.378 | 3 |
| 102 | 27.250 | 9.117 | 3 | 3 | 9.117 | 3 |
| 103 | 27.375 | 9.864 | 3 | 3 | 9.865 | 3 |

Conclusion:

```text
Multiple scoring/lambda branches alone are not enough.
Random seed diversity remains essential for the current method.
```

## Full FFHQ-25 multi-lambda selector: seeds 104,105

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---:|---|
| selected_config_seed_by_selector | 28.496 | 29.556 | 9.126 | 0.804 | 1 | 1 | 00005 |
| global_run_by_selector | 28.498 | 29.556 | 9.126 | 0.804 | 1 | 1 | 00005 |
| selected_config_bestofk | 28.541 | 29.556 | 9.126 | 0.805 | 1 | 1 | 00005 |
| oracle_all_candidates | 28.618 | 29.556 | 10.819 | 0.811 | 1 | 1 | 00005 |

The oracle fails, so this is candidate availability failure.  The best available `00005` candidate is only about 10.8 dB.

## Full FFHQ-25 multi-lambda selector: seeds 106,107

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---:|---|
| selected_config_seed_by_selector | 28.615 | 29.500 | 11.997 | 0.809 | 1 | 1 | 00028 |
| global_run_by_selector | 28.615 | 29.500 | 11.997 | 0.809 | 1 | 1 | 00028 |
| selected_config_bestofk | 28.723 | 29.550 | 13.699 | 0.811 | 1 | 1 | 00028 |
| oracle_all_candidates | 28.856 | 29.551 | 16.691 | 0.819 | 1 | 1 | 00028 |

This pair fails because no config/seed produces a good raw `00028` reconstruction.  Best raw candidates for `00028` are:

| Config / seed | Raw PSNR |
|---|---:|
| S2 λ=0.02, seed 107 | 16.691 |
| LF, seed 106 | 13.699 |
| S2 λ=0.005, seed 106 | 13.699 |
| LF, seed 107 | 11.997 |

Even resolve alignment remains below 25 dB for this image, so this is not only a raw-orientation issue.

Single-seed simulations again fail:

| Seed | Selected mean | Min | Images <20 | Images <25 | Oracle min | Oracle <25 |
|---:|---:|---:|---:|---:|---:|---:|
| 106 | 27.899 | 13.345 | 2 | 3 | 13.622 | 3 |
| 107 | 24.313 | 5.685 | 7 | 8 | 5.685 | 8 |

## Full FFHQ-25 multi-lambda selector: seeds 108,109

| Method | Mean PSNR | Median | Min | SSIM | Images <20 | Images <25 | Worst |
|---|---:|---:|---:|---:|---:|---:|---|
| selected_config_seed_by_selector | 29.227 | 29.745 | 24.442 | 0.827 | 0 | 1 | 00013 |
| global_run_by_selector | 29.226 | 29.729 | 24.442 | 0.827 | 0 | 1 | 00013 |
| selected_config_bestofk | 29.255 | 29.752 | 24.442 | 0.827 | 0 | 1 | 00013 |
| oracle_all_candidates | 29.255 | 29.753 | 24.442 | 0.827 | 0 | 1 | 00013 |

This run is much better than `104,105` and `106,107`, but still misses the promotion target because `00013` is below 25 dB.  Again the oracle also fails; the issue is candidate availability.

For `00013`, best-of-2 by config is:

| Config | Best PSNR on 00013 |
|---|---:|
| LF | 9.476 |
| S2 λ=0.005 | 11.627 |
| S2 λ=0.02 | 24.442 |
| S2 λ=0.05 | 11.603 |

Thus `λ=0.02` nearly rescues `00013`, but the best available candidate remains just below the 25 dB threshold.

## Negative ablations

### Focused score_radius=0.4, proj_start=300

This failed because even the oracle had no good `00014` candidate:

```text
selected mean ≈ 27.384
min ≈ 13.662
below25 = 1
```

Conclusion:

```text
score_radius=0.4 is too narrow and should not replace score_radius=0.6.
```

### Focused score_radius=0.6, proj_start=200

This failed because even the oracle had no good `00005` candidate:

```text
selected mean ≈ 26.827
min ≈ 10.584
below25 = 1
```

Conclusion:

```text
proj_start=200 is too early and should not replace proj_start=300.
```

## Current interpretation

The project is now in a clearer state:

1. **Selection across scoring rules works.**  The trajectory statistic chooses configurations nearly at oracle level when good candidates exist.
2. **Multi-lambda is better than fixed LF/S2.**  It fixes the `102,103` failure case that fixed LF/S2 could not fix.
3. **Two seeds are not enough for an always-successful method.**  Seed pairs `104,105`, `106,107`, and `108,109` each have an oracle failure on different images.
4. **Single-seed selection is far from enough.**  Single-seed oracles fail, so this is not only a selector-statistic problem.
5. **The remaining bottleneck is candidate generation / adaptive compute.**  The algorithm needs either more seed diversity, better in-loop adaptation, or a recovery mechanism for detected high-risk images.

## What this means for the ultimate goal

The current method is a legitimate non-ground-truth selector, not a PSNR oracle.  It is useful because different scoring rules are complementary and the selector can exploit that complementarity.

But the ultimate goal is an always-successful method.  The data now suggests that an always-successful method should probably be framed probabilistically:

```text
For an image x, each seed/config branch has some probability p_x of producing a good candidate.
The selector has an error probability e_x conditional on at least one good candidate existing.
The total failure probability is roughly:
  P(no good candidate generated) + P(selector chooses wrong | good candidate exists).
```

Empirically, the second term is small; the first term is the current bottleneck.  This suggests two parallel directions:

1. **engineering / algorithmic:** adaptive compute and in-loop multi-lambda scoring to increase candidate success probability;
2. **mathematical / probabilistic:** formalize the guarantee in terms of candidate-generation probability, selector consistency, and diffusion-prior regularity rather than deterministic uniqueness of phase retrieval.

The math direction is worth a short study phase, because pure phase retrieval is ill-posed up to ambiguities, and the diffusion/pretrained prior changes the effective feasible set in a nontrivial way.  A useful near-term theory target is not a full deterministic guarantee, but a probabilistic reliability framework that explains how many seeds/config branches are needed to make failure probability small.

## Recommended next actions

1. Use `docs/empirical_success_probability_multilambda_ffhq25.md` as the active candidate-generation table and update it as new seeds/configs arrive.
2. Simulate adaptive-compute policies using existing traces.
3. Add an adaptive-compute selector:

   ```text
   start with two seeds;
   if selector confidence or candidate pool risk is bad, run more seeds/configs;
   stop when confidence is high or max budget is reached.
   ```

4. Run targeted recovery diagnostics for recurring hard images:

   ```text
   00005
   00013
   00028
   00027 / 00032 / 00034 as guard/failure-moving images
   ```

5. Start a short theory note on probabilistic reconstruction reliability:

   ```text
   candidate-generation success probability;
   selector consistency conditional on good candidates;
   adaptive compute budget needed for high probability of success;
   role of diffusion/prior manifold in ill-posed phase retrieval.
   ```

6. After the algorithmic direction stabilizes, rerun the best method with LPIPS enabled.
