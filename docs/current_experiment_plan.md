# Current experiment plan: NP-SITCOM Branch A and Branch B

Updated: 2026-06-22

## June 22 Branch A status update

Branch A is no longer a final-output selector project.

Current state:

- A0-A4: final-output residual selection was a useful scaffold, but not a viable final solver direction.
- A5-A6: hard-image trajectory logging found early risk signal, with inter-run relative features more promising than absolute residuals.
- A7-A10: held-out retrospective controller analysis found a practical first50 residual-rank AND policy on A8.
- A11: prospectively validated the frozen first50 controller on a fresh trajectory run, but left `7` bad25 misses and `9` false-positive replacements.
- A12: showed many A11 misses were late-developing rather than fundamentally invisible.
- A13: showed first80 residual-rank policies transfer better across A8 / A11 than first50 policies.
- A13.5: showed consensus / nearest-neighbor outlier features catch several residual-rank invisible failures.
- A14: prospectively validated both predeclared frozen policies on a fresh run; both reduced failures substantially, with the conservative policy as primary and the aggressive policy as secondary higher-replacement-budget evidence.
- A15: diagnosed the remaining A14 miss; both frozen policies still miss `image 00017/run0`, which looks like a certificate-invisible floor case rather than a reason to retune A14.
- A16: fresh replication showed the conservative gate is brittle, but the aggressive residual+consensus OR policy replicated strongly and is the best practical Branch A controller.
- A17: broad anytime diagnostics showed that all `96/96` bad25 and `86/86` bad20 runs become visible before `50%` under union diagnostics; this is diagnostic evidence only, not an executable detector.
- A17.5: strict cross-fit audits of the strongest anytime candidates did not yield a budget-feasible frozen rule; the best thresholds collapsed toward do-nothing and image `00017` stayed uncaught by the selected candidate rules.
- A18: population / candidate-set control looks conceptually promising; many image groups keep at least one good SITCOM survivor, and A8/00007 is the clean whole-population-bad case where NP fallback is needed.
- A18.5: the first population score saturates at `2.0` on almost every run, AUROC is `0.5`, and the crude top2 rule mostly degenerates into run-index tie-breaking; A14/00017 is the clean failure example.
- A18.6: the saturation bug was corrected; `top3_weighted` is the safer diagnostic and `top2_weighted` is the more aggressive variant, but both are still candidate-set oracle style rather than executable controllers.
- A18.7: the candidate-set oracle signal is real; the best non-degenerate executable-style rule is `top2_remove_aggressive_weighted` with `lowest_full_residual_proxy` or `lowest_lowfreq_residual_proxy`, but the remaining `bad25/bad20 = 2/2` floor shows the within-set selector is still the bottleneck.
- A18.8: split wiring was fixed and provisional outputs were overwritten; the canonical fit is train `A8+A11 -> A14+A16`, the reverse diagnostic is train `A14+A16 -> A8+A11`, and the best regenerated candidate is `lowest_full_residual_proxy / fallback_if_selected_residual_high`, but selective fallback still does not fix the remaining canonical A18.7 misses.

## Frozen A14 policy definitions

Frozen config paths:

- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config/frozen_policy.json`
- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config/frozen_policy_aggressive.json`
- `/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14/frozen_policy_conservative.json`
- `/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14/frozen_policy_aggressive.json`

Primary conservative A14 policy:

```text
policy_name: consensus_lowfreq_nn
policy_family: consensus_single
feature: lowfreq_dist_to_nearest_neighbor
direction: high_is_risky
threshold: 27.855274200439453
fallback: NP selected fallback from the existing FFHQ-25 NP selector CSV
```

Combined A8+A11 development metrics:

- remaining bad25: `16`
- remaining bad20: `13`
- false-positive replacements: `1`
- total replacements: `40`
- run mean PSNR: `29.210`
- image best-of-4 min PSNR: `28.661`

Secondary aggressive A14 policy:

```text
policy_name: residual_or_lowfreq_nn
policy_family: residual_or_consensus
residual arm:
  x0y_full_residual_normed__interrun_rank__first80pct__slope >= 0.8767080745341616
  AND
  x0y_full_residual_normed__interrun_rank__first80pct__last_in_window >= 3.0
consensus arm:
  lowfreq_dist_to_nearest_neighbor >= 27.855274200439453
rule:
  residual_first80 OR lowfreq nearest-neighbor outlier
```

Combined A8+A11 development metrics:

- remaining bad25: `6`
- remaining bad20: `6`
- false-positive replacements: `4`
- total replacements: `53`
- run mean PSNR: `29.944`
- image best-of-4 min PSNR: `28.661`

Selection note:

- the conservative policy remains the primary A14 policy;
- the aggressive policy is predeclared from A8+A11 only and is intentionally marked as higher-replacement-budget;
- the repo-side copies under `configs/branch_A/a14/` are the in-tree references for A14 evaluation;
- no A14 results may be used to change thresholds or features.

## A14 Prospective Result

A14 used GPU `3` with seeds `71/72` and fresh chunk ordering for task-management reasons. That is recorded in provenance and is not a methodological issue. Both policies were frozen before A14 and were applied unchanged.

| policy | run mean | run min | bad25 | bad20 | replaced | TP repl | FP repl | FN bad25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SITCOM only | `27.184` | `5.084` | `20` | `19` | `0` | `0` | `0` | `20` |
| conservative `consensus_lowfreq_nn` | `30.231` | `5.084` | `2` | `2` | `19` | `18` | `1` | `2` |
| aggressive `residual_or_lowfreq_nn` | `30.428` | `5.084` | `1` | `1` | `21` | `19` | `2` | `1` |
| oracle-risk NP fallback | `30.689` | `26.766` | `0` | `0` | `20` | `20` | `0` | `0` |

A14 row-count sanity passed:

- `trajectory_step_metrics.csv = 20000`
- `run_level_summary.csv = 100`

Main conclusion:

```text
Branch A now has prospective evidence across A11 and A14 that frozen clean-free controllers reduce failures.
The conservative policy remains the primary result, and the aggressive policy is a valid secondary higher-replacement-budget result.
A15 sharpened the diagnosis: the remaining bottleneck is rare certificate-invisible failures, not a lack of signal.
Both policies still miss the same catastrophic floor case, so run-level minimum PSNR remains 5.084.
```

## A15: Remaining-Miss Anatomy After A14

A15 went after the one thing A14 still could not fix: the shared catastrophic floor miss.

The focal case was `image 00017/run0`:

- SITCOM PSNR: `5.084`
- NP-selected fallback: `27.185`
- conservative flag: `False`
- aggressive flag: `False`

The diagnosis was nuanced:

- the run is a strong pixel-space outlier among the four SITCOM runs for that image;
- it is not a strong inlier/outlier signal for the frozen low-frequency consensus gate used by the conservative policy;
- it also does not light up the first80 residual-rank gate used by the aggressive policy;
- the overall pattern is consistent with a rare certificate-invisible failure rather than a reason to change the frozen A14 policies after the fact.

A15 therefore did not reopen policy design. It clarified the failure mode.

## A16: Fresh Replication of the Frozen A14 Policies

A16 replayed the same frozen A14 policies on a fresh SITCOM run with new seeds and chunk ordering.

Replicated result table:

| policy | run mean | run min | bad25 | bad20 | replaced | TP repl | FP repl | FN bad25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SITCOM only | `27.236` | `5.084` | `21` | `16` | `0` | `0` | `0` | `21` |
| `consensus_lowfreq_nn` | `29.646` | `5.084` | `7` | `4` | `15` | `14` | `1` | `7` |
| `residual_or_lowfreq_nn` | `30.337` | `5.084` | `1` | `1` | `24` | `20` | `4` | `1` |
| oracle-risk NP fallback | `30.605` | `25.005` | `0` | `0` | `21` | `21` | `0` | `0` |

Interpretation:

- A16 is a mixed but useful replication.
- The conservative consensus-only policy is weaker than in A14, so it looks brittle.
- The aggressive residual+consensus OR policy replicated strongly and is now the best practical Branch A controller.
- The remaining catastrophic miss is still `image 00017`, now `run1` rather than the A14 `run0` case.
- Do not retune the frozen A14/A16 policies based on this result.

A16 provenance:

- GPUs: `0,3`
- Seeds / chunk ordering: `81/82` with fresh chunk ordering
- Git commit: `a6c4cedaf2c41aa0db7c6172ec8f3fd49c5f49c7`
- Row-count sanity: `trajectory_step_metrics.csv = 20000`, `run_level_summary.csv = 100`

## Current Conclusion Through A16

What Branch A now establishes:

- final-output NP/SITCOM complementarity is real, but insufficient;
- a frozen first50 controller can prospectively improve fresh trajectories (A11), but it still leaves late-developing misses;
- later-window residual-rank policies transfer better across A8 and A11 than first50 policies (A13);
- consensus / nearest-neighbor outlier features catch several residual-rank invisible failures (A13.5);
- frozen clean-free controllers now have prospective evidence across A11, A14, and A16 of reducing failures on fresh SITCOM trajectories;
- A15 and A16 show the remaining failure is a persistent certificate-invisible floor case, not a sign that the whole controller idea is broken.

What Branch A still does not establish:

- elimination of the remaining catastrophic floor case;
- a detector that lifts run-level minimum PSNR above `5.084`;
- a real online mid-trajectory intervention rather than retrospective NP replacement.

So the sharpened Branch A answer is:

```text
Branch A now has prospective evidence across A11, A14, and A16 that frozen clean-free controllers reduce failures.
A17 adds strong anytime visibility diagnostics, but A17.5 shows that stable budgeted anytime control is not solved yet.
A18 through A18.8 make the population / candidate-set direction look real, but the frozen population story is still not ready for prospective A19.
The conservative policy turned out to be brittle in A16, but the aggressive residual+consensus OR policy replicated strongly and remains the best validated Branch A controller so far.
It still does not yet eliminate the remaining catastrophic floor case, so it is not a final solver story yet.
The next direction should be corrected population-health/fallback certificate design using existing trajectories first, rather than freezing a new anytime policy now.
```

## Clean-free certificate note

The conceptual interpretation of Branch A as a clean-free reliability-controller experiment is written in:

`docs/branch_A_clean_free_certificates.md`

That note explains the current certificate families:

- late-window residual-rank behavior;
- low-frequency cross-run consensus;
- the residual+consensus OR policy;
- image `00017` as the persistent certificate-invisible floor case;
- candidate future certificates such as pixel/perceptual consensus and temporal consensus.

## Future controller directions

The two longer-horizon Branch A directions are written in:

`docs/branch_A_future_controller_directions.md`

They are:

```text
Direction A: anytime risk detection
  Replace fixed first50/first80 windows with a per-step cumulative risk process.

Direction B: population / beam controller
  Maintain multiple SITCOM trajectories and use clean-free in-distribution certificates to select, prune, respawn, or fallback.
```

This future-directions note should guide the next Codex development cycle. A17 through A18.8 are diagnostics only; the next recommended algorithmic direction is population / beam controller design using existing trajectories first, not a new anytime-policy freeze.

## Current objective

The project objective is still to develop a reliable diffusion-prior phase-retrieval solver for FFHQ-25. Reliability means not only high average PSNR, but also controlled failure modes: no catastrophic per-image failures, good minimum PSNR, and a method that can be explained as a solver rather than as an offline oracle over many unrelated runs.

## Active paths and environments

Use the following path convention on PAC:

```text
Repo:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Original SITCOM-ODE:
  /egr/research-pac/huang248/external/SITCOM_ODE

Patched SITCOM-ODE for handoff:
  /egr/research-pac/huang248/external/SITCOM_ODE_npsitcom

External DiffFPR utilities:
  /egr/research-pac/huang248/external/DiffFPR

Guided FFHQ checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt
```
