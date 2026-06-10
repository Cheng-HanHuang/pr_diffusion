# Project docs

Updated: 2026-06-10

This directory is now organized around the June 2026 NP-SITCOM phase-retrieval update.  The active docs should be enough to restart the project without scrolling through chat logs or manually rediscovering the PAC paths.

## Active starting points

- `progress_report.md`  
  Main June 10 progress report.  It explains why the project moved from pure NP tuning to two NP-SITCOM branches, summarizes the current all-noise Branch A oracle-complementarity result, and records the current negative-but-informative Branch B sigma-handoff result.

- `current_experiment_plan.md`  
  Active roadmap.  It separates Branch A and Branch B into different experimental directions and lists immediate next experiments for each.

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
Noise Picking is conservative and can rescue some SITCOM failures.
SITCOM-ODE has a higher successful-reconstruction ceiling.
Their candidate pools are complementary across FFHQ-25 and multiple noise levels.
However, one-shot final fallback is too crude, and naive one-shot NP-to-SITCOM sigma handoff is not competitive yet.
The most promising direction is a per-step risk detector/controller that uses NP-style intervention when SITCOM trajectories become unstable or out-of-distribution.
```

## Current branches

### Branch A: candidate selection and per-step controller

Branch A currently mixes NP and SITCOM run-level candidates.  The oracle NP+SITCOM candidate pool beats the individual source oracles across tested noise levels, but executable selection remains unsolved because SITCOM candidates do not yet have comparable measurement-side residual/risk features.

Main files:

```text
scripts/npsitcom/run_sitcom_official_ffhq_one_gpu.sh
scripts/npsitcom/make_sitcom_image_folder.py
scripts/npsitcom/parse_sitcom_metrics.py
scripts/npsitcom/mix_select_candidates.py
```

Main output folders:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/sitcom_official
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchA_mix
```

### Branch B: sigma handoff and continuation diagnostics

Branch B exports NP reconstructions as SITCOM/DAPS sigma states and continues them in a patched copy of SITCOM-ODE.  The 400-row 25-image run executed successfully, but the current naive handoff result is not competitive.  Branch B should be treated as a diagnostic branch unless narrow sigma/measurement-scaling studies improve it.

Main files:

```text
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
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_handoff
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_sitcom_handoff
```

## Recommended reading order

1. Read `progress_report.md` for the current understanding and evidence.
2. Read `current_experiment_plan.md` for the next launches.
3. Use the root `README.md` for the high-level project orientation and path map.
4. Use `historical/` only when reconstructing older NP-only context.
