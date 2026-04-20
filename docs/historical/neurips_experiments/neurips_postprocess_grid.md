# `scripts/neurips_postprocess_grid.py`

## What this script does
Postprocesses a single `neurips_grid_experiments.py` run directory.

Input:
- `run_level.csv`

Outputs:
- `image_level.csv`
- `split_summary.csv`

## Aggregations
`image_level.csv` is grouped by:
- mode
- setting
- image
- method

Per-image statistics include:
- mean/median/max PSNR
- mean full-magnitude error
- mean low-frequency magnitude error
- mean runtime and total runtime

`split_summary.csv` averages the image-level statistics across each split for each `(mode, setting, method)`.

## Usage
```bash
python scripts/neurips_postprocess_grid.py --run_dir /path/to/<mode>_YYYYMMDD_HHMMSS
```
