# Progress report: NP-SITCOM two-branch phase-retrieval update

Updated: 2026-06-20

## June 20 addendum: Branch A through A13.5 and frozen A14 candidate

Branch A has now crossed three important milestones beyond the earlier A10 retrospective controller story.

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

The resulting frozen A14 development policies are now:

Primary conservative policy:

```text
consensus_lowfreq_nn
feature = lowfreq_dist_to_nearest_neighbor
direction = high_is_risky
threshold = 27.855274200439453
```

Secondary aggressive policy:

```text
residual_or_lowfreq_nn
residual first80 AND lowfreq-nearest OR rule
predeclared from A8+A11 only
```

Frozen config folders:

`/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config`
`/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14`

Combined A8+A11 development metrics:

Primary conservative:
- remaining bad25: `16`
- remaining bad20: `13`
- false-positive replacements: `1`
- total replacements: `40`
- run mean PSNR: `29.210`
- image best-of-4 min PSNR: `28.661`

Secondary aggressive:
- remaining bad25: `6`
- remaining bad20: `6`
- false-positive replacements: `4`
- total replacements: `53`
- run mean PSNR: `29.944`
- image best-of-4 min PSNR: `28.661`

Interpretation:

```text
Branch A now has prospective evidence that a frozen controller can help (A11),
and stronger development evidence that late-window and consensus features improve the detector design space (A13 / A13.5).
The next step is a true prospective A14 run evaluating both predeclared frozen policies, with the conservative policy as primary and the aggressive policy as secondary.
```

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
