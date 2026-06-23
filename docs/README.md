# Project docs

Updated: 2026-06-23

This directory is organized around the June 2026 NP-SITCOM phase-retrieval update.  The active docs should be enough to restart the project without scrolling through chat logs or manually rediscovering PAC paths.

## Active starting points

- `progress_report.md`  
  Main project-state report.  It now records Branch A through A18.8 and Branch B through B18D.  The main Branch-B update is that naive NP-to-SITCOM handoff is no longer the main path; the best fair Branch-B result is fixed-budget 4S SITCOM population selection.

- `current_experiment_plan.md`  
  Active roadmap.  It separates validated / diagnostic evidence from next recommended work and emphasizes fixed-budget comparisons rather than unbounded fallback retries.

- `branch_A_clean_free_certificates.md`  
  Conceptual Branch-A note on clean-free reliability certificates, residual-rank behavior, consensus/outlier features, and why image `00017` remains a persistent hard case for current controller families.

- `branch_A_future_controller_directions.md`  
  Longer-horizon Branch-A directions: anytime risk detection and population / beam control.

- `branch_B_fixed_budget_population_selector.md`  
  Current Branch-B note.  It records the B11--B18D result: a fixed-budget 4S SITCOM selector is the defensible Branch-B method; 4-to-8 fallback and same-budget NP hybrids are diagnostic but not yet executable final algorithms.

- `runbooks/tmp_runner_record.md`  
  Record of temporary/local runner scripts and command patterns used during FFHQ testing.

- `repo_audit_notes.md`  
  Script/doc inventory and cleanup checks.

- `external_np_difffpr_benchmark_setup.md`  
  Notes for external DiffFPR-style benchmarking setup.

- `historical/`  
  Archived older plans, previous progress reports, PAC migration notes, NeurIPS phased experiment docs, and backups.  The May/May-23 NP-only active docs are archived here after the June 10 NP-SITCOM update.

## Current methodological state

The active project story is:

```text
SITCOM-ODE has a high successful-reconstruction ceiling but occasional catastrophic failures.
NP can rescue some SITCOM failures, but current one-shot NP also has severe image-specific failures.
Naive NP-to-SITCOM sigma handoff is technically useful but not competitive as a solver.
The most defensible Branch-B result is fixed-budget 4S SITCOM population selection.
The most mature Branch-A result is the frozen aggressive residual+consensus controller, though it still has a catastrophic floor.
```

## Current branches

### Branch A: clean-free controller / selector path

Branch A is a reliability-controller experiment.  The strongest validated policy is the aggressive residual+consensus OR controller from the A14/A16 frozen-policy line.  A17--A18.8 show that anytime and population/candidate-set diagnostics are promising, but a new prospective population policy is not ready yet.

Main files:

```text
scripts/npsitcom/run_sitcom_official_ffhq_one_gpu.sh
scripts/npsitcom/make_sitcom_image_folder.py
scripts/npsitcom/parse_sitcom_metrics.py
scripts/npsitcom/mix_select_candidates.py
configs/branch_A/a14/frozen_policy_conservative.json
configs/branch_A/a14/frozen_policy_aggressive.json
```

Main output folders:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchA_mix
```

### Branch B: fixed-budget SITCOM population selection and diagnostics

Branch B began as sigma-space NP-to-SITCOM handoff.  B3--B8 showed that forcing NP states through SITCOM continuation damages NP estimates and is not the current solver path.

The useful Branch-B result is now the fixed-budget 4S SITCOM population selector:

```text
run 4 independent SITCOM trajectories;
at tau = 0.8 select the run with lowest correction_norm;
use population spread as a diagnostic warning only.
```

Pooled over B11, B12, and B16-stage1, this gives mean selected PSNR `30.565`, min selected PSNR `5.087`, bad25 `2/75`, and bad20 `1/75`.  The two failures split into one selector failure (`B11/00027`) and one SITCOM population-generation failure (`B12/00017`).

The 4-to-8 escalation and 3S+1NP hybrid experiments are not final methods.  They are diagnostic: extra candidates can repair failures in an oracle sense, and NP rescues `00017`, but NP also catastrophically fails on several images where SITCOM is good.

Main files:

```text
scripts/npsitcom/run_branchA_sitcom_trajectory_hard.py
scripts/npsitcom/run_branchB_export_np_handoff_with_measurement_ffhq_one_gpu.sh
scripts/npsitcom/export_np_handoff_states_with_measurement.py
scripts/npsitcom/sitcom_patch/npsitcom_handoff_sample.py
scripts/npsitcom/run_branchB_sitcom_handoff_one_gpu.sh
```

Main external/code paths:

```text
/egr/research-pac/huang248/external/SITCOM_ODE
/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom
```

Main output folders:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_B
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_handoff
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_sitcom_handoff
```

## Recommended reading order

1. Read `progress_report.md` for the current project state.
2. Read `branch_B_fixed_budget_population_selector.md` for the current Branch-B conclusion.
3. Read `branch_A_clean_free_certificates.md` for the Branch-A certificate interpretation.
4. Read `current_experiment_plan.md` for next work.
5. Use the root `README.md` for the high-level project orientation and path map.
6. Use `historical/` only when reconstructing older NP-only context.
