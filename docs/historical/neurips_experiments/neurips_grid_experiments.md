# `scripts/neurips_grid_experiments.py`

## What this script does
Runs plan-aligned sweeps for phases 2-5:
- `sitcom_lr`: `lr_inner` sweep
- `sitcom_noise`: `eta_scale` and `init_scale` sweep
- `np_schedule`: soft/hard candidate count and projection-start sweeps
- `budget`: step-budget/runtime sweeps
- `mechanism`: full/score-only/projection-only/no-masking Noise Picking ablation

Outputs `run_level.csv` with one row per executed method run (`method ∈ {sitcom, noise_picking}`).

## Parameter choices
Defaults map directly to current grid values:
- Seeds: `100,101,102,103,104` (5 restarts)
- SITCOM LR: `{0.02,0.05,0.1}`
- SITCOM noise/init: `{0.5,1.0}` × `{0.75,1.0,1.25}`
- Noise Picking schedule: soft `{3,5,7}`, hard `{1,2,3}`, proj_start `{200,400,600}`
- Budget steps: NP `{250,500,750,1000}` and SITCOM `{20:20,50:10,100:5}`

## Method execution policy
`--methods auto` is default and uses compute-efficient behavior:
- `sitcom_lr`, `sitcom_noise`: SITCOM only
- `np_schedule`, `mechanism`: Noise Picking only
- `budget`: both methods

Override with `--methods {both,sitcom,noise_picking}` if needed.

## Postprocessing
Use `scripts/neurips_postprocess_grid.py` on each run directory to produce:
- `image_level.csv` (per-image mean/median/max PSNR + runtime/error summaries)
- `split_summary.csv` (averaged image-level summaries across the split)

## Expected time
Depends heavily on mode and selected methods; Noise Picking remains dominant in runtime.
