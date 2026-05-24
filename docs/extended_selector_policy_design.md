# Extended selector policy design

Updated: 2026-05-23

## Motivation

The `00013` margin-recovery run changed the diagnosis:

```text
candidate generation: improved; every seed has at least one >28 dB candidate
config selection: solved on the run; selected config = oracle config
seed selection: still fails; current selector chooses 26.54 dB while selected-config oracle is 28.69 dB
```

Inside the selected config `hard_s2_lam0p15_ps450_soft8_hard1`, the best seed was rank 4 by `post_winner_lf_mse_mean`.  Therefore the first selector-policy sweep with top-2/top-3 reranking was too narrow.

## Added scripts

```text
scripts/simulate_selector_policy_variants_extended.py
scripts/run_extended_selector_policy_sweep.sh
```

Recommended command:

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
git pull
bash scripts/run_extended_selector_policy_sweep.sh
```

If the newest margin-recovery folder is not detected automatically:

```bash
TRACE_ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_00013_margin_recovery_<timestamp> \
bash scripts/run_extended_selector_policy_sweep.sh
```

## Policy families tested

### 1. Wider top-k seed reranking

The previous top-2/top-3 reranking cannot recover a seed that is rank 4 by trajectory statistic.  The extended sweep tests:

```text
top1cfg_top4stat_rerank_lf_resid
top1cfg_top5stat_rerank_lf_resid
top1cfg_top6stat_rerank_lf_resid
top1cfg_top8stat_rerank_lf_resid
```

and the corresponding full-residual versions.

### 2. Second-best residual diagnostic

Because the absolute residual minimum can be a measurement-overfit candidate, the sweep also tests:

```text
top1cfg_top4stat_second_lf_resid
top1cfg_top5stat_second_lf_resid
top1cfg_top6stat_second_lf_resid
top1cfg_top8stat_second_lf_resid
```

This is diagnostic, not yet proposed as a final method.

### 3. Top-2 config pools

The sweep tests whether keeping the top two configs by averaged statistic helps:

```text
top2cfg_top4stat_rerank_lf_resid
top2cfg_top5stat_rerank_lf_resid
top2cfg_top6stat_rerank_lf_resid
top2cfg_top8stat_rerank_lf_resid
```

### 4. Rank aggregation

Instead of using raw normalized values, rank aggregation combines rankings by:

```text
post_winner_lf_mse_mean
raw_noisy_lowfreq_mag_l2
raw_noisy_mag_l2
```

Policies include:

```text
top1cfg_top4_rankagg_stat1_lf1_full0p25
top1cfg_top5_rankagg_stat1_lf1_full0p25
top1cfg_top6_rankagg_stat1_lf1_full0p25
top2cfg_top4_rankagg_stat1_lf1_full0p25
top3cfg_top4_rankagg_stat1_lf1_full0p25
```

### 5. Statistic-band policies

These keep all candidates within a relative band of the best trajectory statistic and then rerank by residual:

```text
top1cfg_statband0p02_rerank_lf_resid
top1cfg_statband0p04_rerank_lf_resid
top1cfg_statband0p06_rerank_lf_resid
top1cfg_statband0p08_rerank_lf_resid
```

plus full-residual and top-2-config variants.

### 6. Sharp-safe branch heuristic

The run suggests two branch types:

```text
stable-safe branch:
  lambda=0.15, proj_start=350, soft=8
  stable around 25--28 dB

sharp branch:
  lambda in [0.08,0.20], proj_start >= 400
  can produce >28 dB candidates
```

The heuristic first searches sharp branches, then falls back to the stable-safe branch.

## Risk diagnostics

The extended sweep writes:

```text
extended_selector_risk_diagnostics.csv
```

This contains non-oracle diagnostics for whether the algorithm should defer / add compute instead of finalizing:

```text
config_margin_rel
seed_margin_rel
stat_residual_disagreement
defer_by_nonoracle_risk_rule
defer_reasons
```

The current diagnostic defer rule triggers if:

```text
config margin is small, or
seed margin is small, or
stat-selected seed disagrees with residual-selected seed.
```

This is not yet a final theorem-backed algorithm.  It is a way to test whether the failure case can be detected before committing to a bad seed.

## What to inspect

After running the extended sweep, inspect:

```bash
column -s, -t < <OUTDIR>/extended_selector_policy_summary.csv | less -S
grep ',28' <OUTDIR>/extended_selector_policy_by_threshold.csv | column -s, -t | less -S
column -s, -t < <OUTDIR>/extended_selector_risk_diagnostics.csv | less -S
```

Important columns:

```text
psnr_min
mean_regret_vs_oracle
max_regret_vs_oracle
n_selector_fail_given_oracle_success
selector_failed_images
defer_by_nonoracle_risk_rule
defer_reasons
```

## Decision logic

### Case A: a non-oracle policy selects >=28 dB on 00013

Promote that policy to the next selector candidate and test it on:

```text
00013,00027,00028 combined trace
then full hard set
then full FFHQ-25
```

### Case B: no non-oracle policy selects >=28 dB, but risk diagnostics defer

This is still useful.  It means the algorithm can avoid making a bad final choice by triggering more adaptive compute.

Promote the rule as a risk gate:

```text
if seed/config confidence is low, do not finalize; add seeds/configs.
```

### Case C: no policy succeeds and risk diagnostics do not defer

Then the current scalar features are inadequate.  The next selector should use richer information, such as saved low-dimensional trajectory curves or image-space consistency checks.

## Current expected outcome

Given the latest 00013 run, the likely useful result is not necessarily a perfect selector.  The most valuable result may be:

```text
The risk gate detects that current selection is ambiguous and should not finalize.
```

This would align with the reliability framework: avoid bad acceptance, and spend more compute only when necessary.