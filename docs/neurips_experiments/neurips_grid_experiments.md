# `scripts/neurips_grid_experiments.py`

## What this script does
Runs plan-aligned sweeps for phases 2-5:
- `sitcom_lr`: `lr_inner` sweep
- `sitcom_noise`: `eta_scale` and `init_scale` sweep
- `np_schedule`: soft/hard candidate count and projection-start sweeps
- `budget`: step-budget/runtime sweeps
- `mechanism`: full/score-only/projection-only/no-masking Noise Picking ablation

Outputs `run_level.csv` with metrics and runtime.

## Parameter choices
Defaults map directly to experiment plan values:
- SITCOM LR: `{0.02,0.05,0.1}`
- SITCOM noise/init: `{0.5,1.0}` × `{0.75,1.0,1.25}`
- Noise Picking schedule: soft `{3,5,7}`, hard `{1,2,3}`, proj_start `{200,400,600}`
- Budget steps: NP `{250,500,750,1000}` and SITCOM `{20:20,50:10,100:5}`

## Expected time
Depends heavily on mode; runtime is dominated by Noise Picking runs. Use this script with smaller split files for quick validation and with full split files for final runs.
