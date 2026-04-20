# `scripts/neurips_canonical_compare.py`

## What this script does
Runs the canonical paper comparison and writes:
1. `run_level.csv` (each image/seed/method run)
2. `image_level.csv` (mean/median/max over restart seeds per image/method/radius)

It always runs:
- Noise Picking with masked score+projection at radius `r`
- SITCOM as either:
  - unmasked (`meas_radius=None`) for headline baseline
  - masked (`meas_radius=r`) for secondary ablation

## Parameter choices
- Seeds default to `100..109` (R=10).
- Radii passed with `--radii` (e.g., `0.1,0.2,0.5` for validation).
- Default SITCOM config follows existing repo defaults.
- Default Noise Picking config follows current project defaults (`num_steps=1000`, soft=5, hard=2, `proj_start=400`).

## Expected time
Using provided estimate anchors:
- SITCOM: ~800 sec per 1000 steps or ~15 sec per 20 steps
- Noise Picking: ~60 min per default run (1000-step setting)

Total time scales linearly with image count × seed count × radius count.
