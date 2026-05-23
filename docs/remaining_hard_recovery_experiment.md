# Remaining hard-image recovery experiment

Updated: 2026-05-23

## Motivation

The hard-image ablations with seeds `110--115` showed that the current expanded pool is promising for a **25 dB reliability target**, but not yet reliable for a **28 dB target**.

The remaining bottleneck images are:

```text
00013
00027
00028
```

Empirical diagnosis:

```text
00013:
  Step 2 had no >28 dB candidate among seeds 112--115, even with LF and
  S2 lambdas {0.01,0.02,0.05,0.1}.  This is candidate-generation failure.

00027:
  Some good candidates exist, especially around mid-lambda branches, but
  seed-level success is low.  Candidate generation and selector calibration both matter.

00028:
  Good candidates exist, often LF or high lambda, but seed-level success is very low.
  Bad seeds cannot be rescued by selector improvements alone.
```

Therefore the next experiment should not be a full FFHQ-25 run yet.  It should first increase the candidate-generation probability \(p_x\) for these remaining hard cases.

## Added launcher

```text
scripts/launch_ffhq_remaining_hard_recovery_grid.sh
```

Default command on PAC:

```bash
bash scripts/launch_ffhq_remaining_hard_recovery_grid.sh
```

Default grid:

```text
images      = 00013,00027,00028
seeds       = 116,117,118,119,120,121,122,123
lambdas     = 0.03,0.05,0.08,0.10,0.12,0.15
proj_start  = 300,350,400
soft/hard   = 5/1
GPUs        = 0,1,2,3
parallel    = 4 jobs
```

The grid intentionally focuses on mid/high S2 lambda values:

```text
lambda 0.03 / 0.05:
  tests the 00027-like regime where mid-lambda worked better than 0.1.

lambda 0.08 / 0.10 / 0.12 / 0.15:
  tests whether 00013 and 00028 need stronger trajectory regularization than
  the original 0.01/0.02/0.05 pool.

proj_start 300 / 350 / 400:
  tests whether high lambda should act earlier, at the previously promising 350,
  or later to preserve diffusion-prior exploration.
```

## Optional diversity grid

If the main grid still has oracle failures, run:

```bash
RUN_DIVERSITY=1 bash scripts/launch_ffhq_remaining_hard_recovery_grid.sh
```

This adds a diversity-oriented follow-up:

```text
lambdas     = 0.05,0.08,0.10,0.12
proj_start  = 350
soft        = 8
hard        = 1,2
```

Use this only if needed, because it is more expensive.

## What to inspect after completion

Main output root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_remaining_hard_recovery_grid_<timestamp>
```

Important files:

```text
hard_ablation_diag_run_trace_summary.csv
hard_ablation_post_winner_lf_mse_mean_selected_summary.csv
hard_ablation_post_winner_lf_mse_mean_image_diagnostics.csv
reliability_analysis/hard_image_candidate_availability.csv
reliability_analysis/adaptive_policy_summary.csv
reliability_analysis/selector_calibration_global.csv
```

Quick checks:

```bash
column -s, -t < <OUTROOT>/reliability_analysis/hard_image_candidate_availability.csv | less -S
grep mean_over_orders <OUTROOT>/reliability_analysis/adaptive_policy_summary.csv | column -s, -t | less -S
column -s, -t < <OUTROOT>/hard_ablation_post_winner_lf_mse_mean_selected_summary.csv | less -S
```

## Success criteria

For candidate generation, the key metric is not mean PSNR.  The key metric is seed-level success probability.

For each of `00013`, `00027`, and `00028`, check:

```text
seed_success_rate at 25 dB
seed_success_rate at 28 dB
best_raw_psnr
best_config_family
success counts by config family
```

A setting is promising if it increases the number of seeds with at least one >28 dB candidate.

## Decision after this experiment

### Case A: all three images get frequent >28 candidates

Then proceed to selector/risk calibration and full FFHQ-25 validation.

Candidate pool to test on full FFHQ-25 should include:

```text
LF
original robust branches: S2 lambda=0.005,0.02,0.05 at proj_start=300
new recovery branches discovered from this experiment
```

### Case B: oracle still fails on one image often

Then the bottleneck remains candidate generation.  Run the optional diversity grid or consider changing the candidate generator:

```text
soft_candidates = 8
hard_candidates = 2
noise_memory_k > 0
in-loop lambda arbitration
parallel lambda groups inside one trajectory
```

### Case C: oracle succeeds but selector misses

Then the bottleneck is selector calibration.  Move from single-statistic selection to a multi-feature selector using:

```text
post_winner_lf_mse_mean
post_winner_full_mse_mean
raw_noisy_lowfreq_mag_l2
raw_noisy_mag_l2
pre/post LF margin features
winner_is_lf_best_frac_pre/post
config-level statistic margins
seed-level statistic margins
```

## Current hypothesis

The most likely outcome is mixed:

```text
00013 may need lambda around 0.10--0.15 and/or later proj_start.
00027 may prefer mid lambda around 0.03--0.08.
00028 may remain seed-sensitive and require adaptive resampling.
```

This experiment is designed to turn those hypotheses into candidate-generation statistics.