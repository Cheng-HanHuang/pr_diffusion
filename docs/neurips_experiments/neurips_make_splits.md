# `scripts/neurips_make_splits.py`

## What this script does
Creates fixed image split files for NeurIPS experiments:
- `dev_10.txt`
- `validation_25.txt`
- `test_50.txt`
- `seed_list_10.txt` (100-109)

It scans `--data_root` for `.jpg` files and only keeps IDs `00000` to `05401` (inclusive), matching the experiment plan constraints.

## Parameter choices
- `--seed 20260411`: deterministic split creation.
- `--dev_count 10`, `--val_count 25`, `--test_count 50`: fixed protocol sizes.
- `--max_image_id 5401`: enforces partial dataset rule.

## Expected time
- Usually under 1 minute for typical image counts.
