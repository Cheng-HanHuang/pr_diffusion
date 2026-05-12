# FFHQ-25 Guided NP Pilot: Comparison Matrix and Run Spec

This document defines a **detailed pilot benchmark spec** for running Noise Picking (NP) with the guided-diffusion FFHQ checkpoint on the currently available FFHQ-25 image list. It is designed so that the full benchmark can later be run by replacing only the image list/test split.

## 1) Goal and scope

- Primary pilot target: **canonical NP** only (`np_canonical`) as baseline.
- Dataset subset: **existing FFHQ 25-image list** in the environment.
- Backend: `prdiffusion/guided_backend.py` with guided diffusion checkpoint (`ffhq_10m.pt` style).
- Measurement protocol: DiffFPR-style oversampled Fourier magnitude (with optional additive amplitude noise).
- Reconstruction count per image: **4 reconstructions** (4 seeds).
  - For future method comparisons using 3 reconstructions (e.g., DiffFPR), use the **first three seeds** from the same list.

## 2) Methods/settings matrix

## Methods in pilot now

- `np_canonical`
  - soft/hard candidate schedule: `5 / 1`
  - late projection start: `proj_start = 400`
  - default `np_steps = 1000`

## Methods prepared for later direct comparison (not required in this pilot run)

- `np_fixedk_lateproj` (already supported by script) for equal-k comparisons.
- DiffFPR/DAPS/SITCOM external baselines can be compared later via the same condition grid.

## Condition sweep (pilot)

- Oversampling ratio: **`0, 2, 4`**
- Measurement noise std (`sigma_y`): **`0, 0.01, 0.05, 0.1`**
- Radius: `score_radius = proj_radius = 0.5` (relative-frequency radius)
  - If switching to an implementation that interprets radius as absolute pixels, scale radius by resolution.

## 3) Alignment/evaluation modes

Each reconstruction is evaluated in 3 modes:

1. `raw`: no ambiguity resolution.
2. `rot180`: best per-channel 180-degree ambiguity resolution.
3. `resolve`: `rot180` + axis flips + phase-correlation shift alignment.

This gives separate metric tables for “raw/aligned/ambiguity-resolved” comparisons.

## 4) Metrics required

For each run, compute:

- PSNR
- SSIM
- LPIPS (if `lpips` Python package available; otherwise recorded as NaN)
- Runtime per reconstruction (seconds)
- NFE / denoiser calls (`np_steps - 1`)

For each condition bucket `(method, alignment_mode, oversample, noise_std)` aggregate:

- `best / mean / median` for PSNR, SSIM, LPIPS
- Count of reconstructions with `PSNR < threshold` (default threshold = 20 dB)
- mean/median runtime
- mean NFE

## 5) Output artifacts (CSV)

The benchmark now emits under timestamped run folder:

- `difffpr_np_guided_<timestamp>__configs.csv` — uniquely named run settings per condition.
- `difffpr_np_guided_<timestamp>__run_level.csv` — uniquely named one-row-per-(image,seed,alignment) table.
- `difffpr_np_guided_<timestamp>__image_level_summary.csv` — uniquely named per-image aggregation.
- `difffpr_np_guided_<timestamp>__condition_level_summary.csv` — uniquely named per-condition aggregation.
- Backward-compatible aliases (`configs.csv`, `run_level.csv`, etc.) are also emitted in the same run folder for existing scripts.

## 6) Run commands

Use the new pilot wrapper script:

```bash
DATA_ROOT=/path/to/ffhq/root \
IMAGE_LIST_FILE=/path/to/ffhq_available25.txt \
GUIDED_MODEL_PATH=/path/to/ffhq_10m.pt \
GUIDED_DIFFUSION_DIR=/path/to/DiffFPR_or_guided_diffusion_repo \
OUTDIR=/path/to/output_root \
bash scripts/pr_external_ffhq25_np_guided_pilot.sh
```

Optional for canonical-only 4 reconstructions (default already set):

```bash
VARIANTS=np_canonical SEEDS=100,101,102,103 bash scripts/pr_external_ffhq25_np_guided_pilot.sh
```

## 7) Full benchmark transition

To scale from pilot to full benchmark, keep code/settings identical and only change:

- `IMAGE_LIST_FILE` to the final test split.
- `OUTDIR` to full-experiment destination.

Everything else (conditions, metrics, alignment analysis) is already compatible.
