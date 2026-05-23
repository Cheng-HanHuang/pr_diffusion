# Selector policy sweep plan

Updated: 2026-05-23

## Motivation

The remaining-hard recovery grid showed that the candidate pool contains >28 dB candidates for:

```text
00013
00027
00028
```

However, the executable selector based only on `post_winner_lf_mse_mean` still selected a <28 dB seed for `00013`.

This means the immediate next step should be an **offline selector-policy sweep**, not another GPU-heavy reconstruction grid.

The purpose is to answer:

```text
Can the existing candidate pool be selected reliably by a better non-oracle rule?
```

## Added scripts

### `scripts/simulate_selector_policy_variants.py`

Consumes one or more `*_run_trace_summary.csv` files and simulates selector variants.

Outputs:

```text
selector_policy_normalized_candidates.csv
selector_policy_image_level.csv
selector_policy_summary.csv
selector_policy_by_threshold.csv
selector_policy_failures.csv
```

### `scripts/run_selector_policy_sweep_existing_traces.sh`

Convenience launcher for PAC.

Usage:

```bash
TRACE_ROOT=/path/to/np_ffhq_remaining_hard_recovery_grid_<timestamp> \
bash scripts/run_selector_policy_sweep_existing_traces.sh
```

If `TRACE_ROOT` is omitted, the script tries to use the newest `np_ffhq_remaining_hard_recovery_grid_*` folder.

### Optional: `scripts/launch_ffhq_00013_margin_recovery.sh`

Run only if the selector sweep shows that 00013 still needs more candidate margin.

Default grid:

```text
image       = 00013
seeds       = 124,125,126,127,128,129,130,131
lambdas     = 0.05,0.08,0.1,0.12,0.15,0.18,0.2
proj_start  = 350,400,450
soft        = 5,8
hard        = 1
```

## Selector variants tested

The offline sweep includes:

```text
current_config_mean_stat_seed_min_stat
  current two-level selector.

top1cfg_top2stat_rerank_lf_resid
  choose best config by averaged statistic;
  keep top 2 seeds by trajectory statistic;
  rerank using final noisy low-frequency residual.

top1cfg_top3stat_rerank_lf_resid
  same but top 3 seeds.

top2cfg_top2stat_rerank_lf_resid
  keep top 2 configs by averaged statistic;
  keep top 2 seeds per config;
  rerank using final noisy low-frequency residual.

top2cfg_combined_stat_lf_full
  keep top 2 configs;
  score by normalized trajectory statistic + low-frequency residual + full residual.

top3cfg_top3stat_combined_stat_lf_full
  keep top 3 configs and top 3 seeds per config;
  use a combined normalized score.
```

Diagnostic oracle policies are also included:

```text
diagnostic_selected_config_oracle_seed
diagnostic_oracle_all_candidates
```

These use PSNR and are not executable.  They measure candidate-pool and config-selection limits.

## What to inspect

After running:

```bash
bash scripts/run_selector_policy_sweep_existing_traces.sh
```

inspect:

```bash
column -s, -t < <OUTDIR>/selector_policy_summary.csv | less -S
grep ',28' <OUTDIR>/selector_policy_by_threshold.csv | column -s, -t | less -S
column -s, -t < <OUTDIR>/selector_policy_failures.csv | less -S
```

Key columns:

```text
psnr_min
mean_regret_vs_oracle
max_regret_vs_oracle
n_selector_fail_given_oracle_success at 28 dB
selector_failed_images
```

## Decision logic

### Case A: a non-oracle selector achieves zero 28 dB failures

Promote that selector to the next executable algorithm.

Then test it on a broader hard set or full FFHQ-25.

### Case B: all non-oracle selectors still fail only 00013

Run:

```bash
bash scripts/launch_ffhq_00013_margin_recovery.sh
```

This adds candidate-generation margin for `00013`.

### Case C: selector failures occur on multiple images

The current scalar/stat-residual features are not enough.  Build a learned or calibrated risk model using the trace features:

```text
post_winner_lf_mse_mean
post_winner_full_mse_mean
raw_noisy_lowfreq_mag_l2
raw_noisy_mag_l2
pre/post LF margin features
winner_is_lf_best_frac_pre/post
config-stat margin
seed-stat margin
```

## Recommended next command

Run the offline selector sweep first:

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
git pull
bash scripts/run_selector_policy_sweep_existing_traces.sh
```

If the newest remaining-hard grid is not automatically detected, pass it explicitly:

```bash
TRACE_ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_remaining_hard_recovery_grid_<timestamp> \
bash scripts/run_selector_policy_sweep_existing_traces.sh
```