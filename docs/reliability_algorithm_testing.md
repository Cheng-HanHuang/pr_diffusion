# Reliability algorithm/testing workflow

Updated: 2026-05-23

This document records the concrete algorithm/testing workflow following `docs/probabilistic_reliability_note.md`.

The theory note decomposes failure as

\[
  \mathbb P(\mathrm{Fail}_\tau)
  \le
  \mathbb P(\mathrm{NoGood}_\tau)
  +
  \mathbb P(\mathrm{SelFail}_\tau\mid\mathrm{GoodExists}_\tau).
\]

The experiments below are designed to measure and improve the two terms separately.

## Added scripts

### 1. `scripts/analyze_reliability_from_traces.py`

Purpose:

```text
Consume existing *_run_trace_summary.csv files and produce:
  adaptive-compute simulation tables;
  selector-statistic calibration tables;
  candidate availability tables;
  hard-image summaries.
```

Primary outputs:

```text
normalized_candidates.csv
selector_calibration_global.csv
selector_calibration_by_image.csv
candidate_availability_by_image.csv
hard_image_candidate_availability.csv
adaptive_policy_summary.csv
adaptive_policy_image_level.csv
```

What this tests:

```text
1. Candidate generation:
   How often does each image/seed/config produce PSNR >= tau?

2. Selector calibration:
   Does post_winner_lf_mse_mean separate successful candidates from failures?

3. Adaptive compute:
   How many seeds would be used by fixed/adaptive policies, and how many
   below-threshold failures remain?
```

### 2. `scripts/run_reliability_analysis_existing_traces.sh`

Purpose:

```text
One-command launcher for analyzing the existing full-25 multi-lambda traces.
```

Default command on PAC:

```bash
bash scripts/run_reliability_analysis_existing_traces.sh
```

Expected default input root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_multilambda_validation_batch
```

Override example:

```bash
BATCH_ROOT=/path/to/np_ffhq_multilambda_validation_batch \
OUTDIR=/path/to/reliability_analysis \
bash scripts/run_reliability_analysis_existing_traces.sh
```

### 3. `scripts/launch_ffhq_hard_image_reliability_ablation.sh`

Purpose:

```text
Run targeted candidate-generation ablations on recurring hard images:
  00028, 00005, 00013, 00034, 00027, 00007, 00000.
```

The goal is not simply higher mean PSNR.  The goal is to increase the hard-image candidate-generation probability \(p_x\).

Default command on PAC:

```bash
bash scripts/launch_ffhq_hard_image_reliability_ablation.sh
```

Small first run:

```bash
SEEDS=110,111 \
LAMBDAS="0.01 0.1" \
PROJ_STARTS="350" \
SOFT_VALUES="5" \
HARD_VALUES="1" \
bash scripts/launch_ffhq_hard_image_reliability_ablation.sh
```

Default grid:

```text
images       = 00028, 00005, 00013, 00034, 00027, 00007, 00000
seeds        = 110,111,112,113
lambdas      = 0.01, 0.02, 0.05, 0.1
proj_start   = 300, 350, 400
soft         = 5, 8
hard         = 1, 2
score_radius = 0.6
proj_radius  = 0.2
sigma_y      = 0.05
```

The script runs LF baselines and S2 branches, merges traces, applies the multi-config selector, and then runs the reliability analysis on the resulting hard-image pool.

## How to interpret the existing-trace analysis

After running

```bash
bash scripts/run_reliability_analysis_existing_traces.sh
```

start with:

```bash
grep mean_over_orders <OUTDIR>/adaptive_policy_summary.csv | column -s, -t | less -S
column -s, -t < <OUTDIR>/selector_calibration_global.csv | less -S
column -s, -t < <OUTDIR>/hard_image_candidate_availability.csv | less -S
```

### Adaptive-compute simulation

Look at `adaptive_policy_summary.csv`.

Important columns:

```text
policy_display
psnr_mean
psnr_min_mean_over_orders
psnr_min_worst_over_orders
n_images_below_threshold_mean
n_images_below_threshold_max
oracle_failures_mean
selector_failures_given_oracle_success_mean
avg_seeds_used
max_seeds_used_max
failed_images_union
```

Interpretation:

```text
fixed1/fixed2/fixed4/etc. estimate how fixed budgets behave under random seed order.

oracle_until_success is not executable because it uses PSNR, but it gives a
candidate-generation lower bound: how many seeds would be needed if the selector
were perfect.

adaptive_stat_q50/q75/q90/q95 are executable-style policies that stop when the
selected candidate's statistic is below a threshold calibrated from successful
candidate statistics.
```

If adaptive policies reduce average seeds while keeping `n_images_below_threshold_max = 0`, then adaptive compute is promising.

If oracle failures remain even with larger budgets, candidate generation is still insufficient.

If selector failures dominate while oracle failures are small, the selector statistic is insufficient.

### Selector calibration

Look at `selector_calibration_global.csv` and `selector_calibration_by_image.csv`.

Important columns:

```text
stat_mean_success
stat_median_success
stat_mean_failure
stat_median_failure
pearson_stat_vs_psnr
spearman_stat_vs_psnr
auc_neg_stat_predicts_success
```

Interpretation:

```text
Good sign:
  success statistics are lower than failure statistics;
  Pearson/Spearman correlation between stat and PSNR is negative;
  auc_neg_stat_predicts_success is well above 0.5.

Bad sign:
  success and failure statistic distributions overlap heavily;
  AUC is near 0.5;
  per-image AUC is poor on hard images.
```

If AUC is weak globally or on hard images, the next selector should use more than `post_winner_lf_mse_mean`, for example margins, full residuals, and disagreement features.

### Hard-image candidate availability

Look at `hard_image_candidate_availability.csv`.

Important columns:

```text
image_basename
threshold_db
n_candidates
n_success
candidate_success_rate
n_seeds
n_seeds_with_any_success
seed_success_rate
best_raw_psnr
best_config_family
selected_by_stat_raw_psnr
selector_regret_vs_oracle
success_count_<config>
```

Interpretation:

```text
Low seed_success_rate:
  candidate-generation failure dominates.

High seed_success_rate but large selector_regret_vs_oracle:
  selector failure dominates.

A new config with high success_count on a hard image:
  candidate-generation improvement worth promoting.
```

## How to interpret the hard-image ablation

After running

```bash
bash scripts/launch_ffhq_hard_image_reliability_ablation.sh
```

start with:

```bash
column -s, -t < <OUTROOT>/reliability_analysis/hard_image_candidate_availability.csv | less -S
column -s, -t < <OUTROOT>/hard_ablation_post_winner_lf_mse_mean_selected_summary.csv | less -S
column -s, -t < <OUTROOT>/reliability_analysis/selector_calibration_global.csv | less -S
```

The promotion criterion for a new branch should be reliability-oriented:

```text
Promote a branch if it increases hard-image seed_success_rate or candidate_success_rate,
especially for 00028, 00005, and 00013, without creating new failures on guard
hard images 00034, 00027, 00007, and 00000.
```

Do not promote a branch solely because it improves mean PSNR.

## Recommended decision logic after results

### Case A: adaptive compute succeeds with existing branches

If adaptive policies achieve:

```text
n_images_below_threshold_max = 0
avg_seeds_used substantially below fixed4 or fixed6
selector_failures_given_oracle_success near 0
```

then the next algorithm should implement adaptive compute directly:

```text
run two seeds;
check calibrated risk;
add two seeds if risk is high;
stop when risk is low or max budget is reached.
```

### Case B: selector calibration is weak

If `auc_neg_stat_predicts_success` is weak or hard-image AUC is bad, then the next step should be a better selector feature set:

```text
post_winner_lf_mse_mean
post_winner_full_mse_mean
post_lf_mse_margin_mean
pre_winner_lf_mse_mean
final noisy_lowfreq_mag_l2
config disagreement
seed disagreement
```

This would move from a single-statistic selector to a calibrated risk model.

### Case C: hard-image candidate generation remains weak

If hard images still have low seed success even after the ablation, then the next algorithmic work should focus on changing the candidate generator, not the selector.

Candidate-generator ideas:

```text
in-loop lambda arbitration instead of separate full branches;
parallel lambda groups inside one trajectory;
adaptive projection-start schedule;
more diverse noise/candidate memory;
SITCOM-style or DAPS-style exploration modifications.
```

## Current status

The scripts are ready, but the raw trace CSVs are not committed to the GitHub repo.  Therefore the numeric tables must be generated on PAC from the existing output folders.

Once the output CSVs are produced, update:

```text
docs/progress_report.md
docs/empirical_success_probability_multilambda_ffhq25.md
docs/current_experiment_plan.md
```

with the adaptive-policy summary, selector calibration result, and hard-image ablation conclusion.