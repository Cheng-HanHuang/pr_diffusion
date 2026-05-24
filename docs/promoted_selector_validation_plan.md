# Promoted selector validation plan

Updated: 2026-05-24

## What the extended 00013 selector sweep showed

The extended selector-policy sweep on the `00013` margin-recovery trace gave a strong result:

```text
The current selector fails 00013 at 28 dB:
  selected = 26.538 dB
  oracle   = 28.693 dB

Several non-oracle policies recover the oracle candidate:
  top1cfg_top4stat_rerank_lf_resid
  top1cfg_top4stat_rerank_full_resid
  top1cfg_statband0p04_rerank_lf_resid
  top1cfg_statband0p04_rerank_full_resid
  top2cfg_top4stat_rerank_lf_resid
  top2cfg_top4stat_rerank_full_resid
```

The best practical policy family is:

```text
choose config by averaged post_winner_lf_mse_mean;
keep the top 4 seeds by post_winner_lf_mse_mean inside that config;
rerank those top 4 seeds by final noisy low-frequency residual.
```

This fixes the observed failure because the true best seed was rank 4 by trajectory statistic.

## Why we should not immediately run a bigger blind grid

The latest result says:

```text
candidate generation exists;
config selection works;
seed selection needs validation.
```

Therefore the next step is validation, not broad search.

## Added scripts

### 1. Combined offline validation

```text
scripts/run_extended_selector_policy_sweep_combined_hard.sh
```

Purpose:

```text
Evaluate the promoted selector policies jointly on multiple existing trace roots.
```

Recommended command:

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
git pull
bash scripts/run_extended_selector_policy_sweep_combined_hard.sh
```

If auto-detection picks the wrong folders, pass roots explicitly:

```bash
TRACE_ROOTS="/path/to/np_ffhq_00013_margin_recovery_xxx /path/to/np_ffhq_remaining_hard_recovery_grid_xxx /path/to/np_ffhq_hard_image_reliability_ablation_xxx" \
bash scripts/run_extended_selector_policy_sweep_combined_hard.sh
```

Main outputs:

```text
extended_selector_policy_summary.csv
extended_selector_policy_by_threshold.csv
extended_selector_policy_image_level.csv
extended_selector_risk_diagnostics.csv
```

### 2. Compact GPU validation pool

```text
scripts/launch_ffhq_selector_validation_pool.sh
```

Purpose:

```text
Run a compact candidate pool on the hard set using the promoted branches, then
apply current and extended selector-policy analysis.
```

Default command:

```bash
bash scripts/launch_ffhq_selector_validation_pool.sh
```

Default images:

```text
00000,00005,00007,00013,00027,00028,00034
```

Default seeds:

```text
132,133,134,135
```

Default branches:

```text
LF ps350 soft5 hard1
S2 lambda=0.15 ps350 soft8 hard1   # stable-safe fallback
S2 lambda=0.15 ps450 soft8 hard1   # 00013 sharp branch
S2 lambda=0.10 ps450 soft5 hard1   # sharp branch
S2 lambda=0.12 ps450 soft5 hard1   # sharp branch
S2 lambda=0.03 ps400 soft5 hard1   # 00027-like mid branch
S2 lambda=0.05 ps400 soft5 hard1   # 00027/00013 mid branch
S2 lambda=0.05 ps450 soft5 hard1   # selected-stat competitor
```

For a smoke test on only the remaining hard three:

```bash
HARD_IMAGES="00013,00027,00028" SEEDS=132,133 bash scripts/launch_ffhq_selector_validation_pool.sh
```

## What to send back

For the combined offline sweep:

```text
extended_selector_policy_summary.csv
extended_selector_policy_by_threshold.csv
extended_selector_policy_image_level.csv
extended_selector_risk_diagnostics.csv
```

For the compact GPU validation pool:

```text
selector_validation_diag_run_trace_summary.csv
selector_validation_post_winner_lf_mse_mean_selected_summary.csv
selector_validation_post_winner_lf_mse_mean_image_diagnostics.csv
extended_selector_policy_sweep/extended_selector_policy_summary.csv
extended_selector_policy_sweep/extended_selector_policy_by_threshold.csv
extended_selector_policy_sweep/extended_selector_risk_diagnostics.csv
reliability_analysis/hard_image_candidate_availability.csv
reliability_analysis/adaptive_policy_summary.csv
```

## Decision criteria

### Promote selector if

A non-oracle policy, preferably `top1cfg_top4stat_rerank_lf_resid`, has:

```text
zero below-28 failures on the combined hard traces;
zero or near-zero selector failures given oracle success;
reasonable behavior on 00027 and 00028, not just 00013.
```

### Promote candidate pool if

The compact GPU validation pool has:

```text
no below-25 failures under current or promoted selector;
substantially fewer below-28 failures than previous pools;
acceptable oracle coverage on all 7 hard images.
```

### If selector still fails

Use `extended_selector_risk_diagnostics.csv` to check whether failures are at least detected by non-oracle risk/defer rules. If yes, the algorithm should trigger more adaptive compute instead of finalizing.

## Current hypothesis

The likely next viable algorithm is:

```text
1. Choose config by mean post_winner_lf_mse_mean.
2. Keep top 4 seeds in the chosen config by post_winner_lf_mse_mean.
3. Rerank those top 4 seeds by final noisy low-frequency residual.
4. If seed/config margins are weak or stat/residual disagree, defer and add compute.
```

This is still simple, non-oracle, and directly motivated by the failure mode observed on 00013.