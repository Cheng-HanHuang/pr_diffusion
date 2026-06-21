# Progress report: NP-SITCOM two-branch phase-retrieval update

Updated: 2026-06-20

## June 20 addendum: Branch A through A15 miss anatomy

Branch A has now crossed five important milestones beyond the earlier A10 retrospective controller story.

1. **A11 prospective validation:**
   the frozen first50 controller was tested on a fresh SITCOM trajectory run and helped materially:
   - run mean PSNR: `26.358 -> 29.319`
   - bad25: `24 -> 7`
   - bad20: `23 -> 6`
   - replacements: `26` with `9` FP and `17` TP

2. **A12 failure anatomy:**
   the main A11 weakness was not that all misses were invisible. Many were late-developing:
   - `7` bad25 misses total
   - `5 / 7` became jointly risky only after the frozen `first50pct` window

3. **A13 / A13.5 redesign:**
   - A13 showed that `first80` residual-rank detectors transfer better across A8 / A11 than `first50` detectors.
   - A13.5 showed that consensus / nearest-neighbor outlier features catch several residual-rank invisible failures.
   - These A13 / A13.5 results are development evidence only, not prospective validation.

4. **A14 prospective dual-policy validation:**
   both A14 policies were frozen before the fresh run and then evaluated unchanged on GPU `3` with seeds `71/72`.

| policy | run mean | run min | bad25 | bad20 | replaced | TP repl | FP repl | FN bad25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SITCOM only | `27.184` | `5.084` | `20` | `19` | `0` | `0` | `0` | `20` |
| conservative `consensus_lowfreq_nn` | `30.231` | `5.084` | `2` | `2` | `19` | `18` | `1` | `2` |
| aggressive `residual_or_lowfreq_nn` | `30.428` | `5.084` | `1` | `1` | `21` | `19` | `2` | `1` |
| oracle-risk NP fallback | `30.689` | `26.766` | `0` | `0` | `20` | `20` | `0` | `0` |

Row-count sanity passed:
- `trajectory_step_metrics.csv = 20000`
- `run_level_summary.csv = 100`

Interpretation:

```text
Branch A now has prospective evidence across both A11 and A14 that frozen clean-free controllers reduce failures on fresh SITCOM trajectories.
The conservative policy is the primary A14 result and the aggressive policy is a valid secondary higher-replacement-budget result.
The remaining caveat is that both policies still miss the same catastrophic floor case, so run-level minimum PSNR remains 5.084 rather than being lifted by the controller.
```

5. **A15 remaining-miss anatomy:**
   the shared A14 floor miss was traced to `image 00017/run0`.
   - SITCOM PSNR: `5.084`
   - NP-selected fallback: `27.185`
   - both frozen A14 policies missed it
   - the run is a strong pixel-space outlier, but it was not caught by the frozen low-frequency consensus gate or the first80 residual-rank gate

Interpretation:

```text
The remaining bottleneck is now rare certificate-invisible failures, not a total lack of signal.
Branch A should not retune the A14 policies after A14/A15; the right next move is either a fresh replication with the same frozen policies or a writeup / foundations pass on clean-free certificates.
```

Frozen config folders remain:

`/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config`
`/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14`

For the full writeup, see:

- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/BRANCH_A_CONSOLIDATED_REPORT.md`

This report records the current project state after the June 2026 NP-SITCOM experiments. The previous active progress report is archived as `docs/historical/progress_report_archived_20260610_before_npsitcom_two_branch_update.md`.

## Executive summary

The project has moved from pure Noise Picking (NP) tuning toward a hybrid reliability question:

```text
Can we combine NP's conservative failure-avoidance behavior with SITCOM-ODE's higher successful-reconstruction ceiling to obtain a more reliable phase-retrieval solver?
```

Two branches remain active:

1. **Branch A: NP-SITCOM controller / selector path.**
   This is now the main path. The strongest current evidence is that SITCOM failures can be detected with clean-free relative trajectory features, and that consensus-style outlier features add information beyond residual rank alone.

2. **Branch B: sigma-space NP-to-SITCOM handoff.**
   This path remains useful diagnostically, but is still lower priority than Branch A because naive handoff has not yet closed the quality gap.
