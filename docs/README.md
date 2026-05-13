# Project docs

Updated: 2026-05-12

This directory is now organized around the May 12 FFHQ phase-retrieval update.  The active docs should be enough to restart the project without scrolling back through chat logs or re-opening old experiment outputs.

## Active starting points

- `progress_report.md`  
  Detailed May 12 analysis report.  This is the main recap of what has been run and what the results mean.  It records FFHQ-25 guided NP tuning, projection-radius/schedule ablations, SITCOM-ODE comparisons, noise sweeps, candidate-count ablations, score-mode experiments, and the focused S2 lambda sweep.

- `current_experiment_plan.md`  
  Current next-step plan.  This is the active experiment roadmap after the May 12 analysis.  It prioritizes one reliable solver rather than oracle selection, with directions for timestep-dependent/adaptive S2 scoring, candidate-memory/noise-bank selection, NP-in-SITCOM, and robust soft measurement weighting.

- `runbooks/tmp_runner_record.md`  
  Record of temporary/local runner scripts that have been moved into `scripts/` or are relevant for reproducing the recent FFHQ analyses.

- `repo_audit_notes.md`  
  Inventory of local scripts/docs and high-priority code checks after the FFHQ tuning phase.

- `external_np_difffpr_benchmark_setup.md`  
  Notes for external DiffFPR-style benchmarking setup.

- `historical/`  
  Archived older plans, previous progress reports, PAC migration notes, NeurIPS phased experiment docs, and backups taken before the May 12 FFHQ full update.

## Current methodological state

The active project story is:

```text
Noise Picking is useful as conservative branch selection and measurement guidance,
but the current greedy low-frequency score is fragile.  Small low-frequency hard
projection is crucial; broader hard projection is harmful.  Increasing candidate
count alone amplifies score errors.  The next step is to convert NP into a more
reliable single solver or a principled NP-in-SITCOM hybrid.
```

The current practical NP setting is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

At `sigma_y=0.05` on FFHQ-25, this setting gives roughly `29.424 dB` best-of-4 PSNR, `0.832` SSIM, and `0.224` LPIPS.  SITCOM-ODE is still slightly stronger in PSNR/LPIPS at this noise level, but NP has complementary failure behavior and becomes stronger in PSNR/SSIM at higher noise levels (`sigma_y >= 0.10`).

## Recommended reading order

1. Read `progress_report.md` for the full evidence and tables.
2. Read `current_experiment_plan.md` for the next launches.
3. Check `runbooks/tmp_runner_record.md` and `repo_audit_notes.md` to locate the scripts and cleanup items.
4. Use `historical/` only when reconstructing older context.