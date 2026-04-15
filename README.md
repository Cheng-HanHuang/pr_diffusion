# Phase Retrieval with Diffusion Models (SITCOM + Noise Picking)

This repository contains reproducible phase-retrieval experiments built around a pretrained DDPM denoiser (default: `google/ddpm-celebahq-256`) and magnitude-only Fourier measurements.

It includes:
- **SITCOM** reconstruction.
- **Noise Picking** reconstruction.
- **NeurIPS experiment runners** for canonical comparison, phase sweeps, and split generation.
- Slurm templates for multi-phase HPC execution.

### Notice: Many information here are outdated, please check docs/progress_report.md for up to date progress.

## Repository layout

```text
prdiffusion/
  algorithms/
    sitcom.py
    noise_picking.py
  diffusion.py
  fft_ops.py
  io.py
  metrics.py
  seed.py

scripts/
  neurips_canonical_compare.py      # primary paper comparison flow
  neurips_grid_experiments.py       # phase 2-5 sweeps
  neurips_postprocess_grid.py       # run_level -> image_level/split_summary
  neurips_make_splits.py            # fixed split generation

  # legacy scripts retained for backward compatibility
  compare_methods.py
  compare_methods_no_lowfreq.py
  sitcom_lr_sweep.py
  sitcom_noise_ablation.py

  # Slurm scripts for NeurIPS phases
  slurm_neurips_phase0_sanity.sh
  slurm_neurips_phase1_radius.sh
  slurm_neurips_phase2_sitcom_tuning.sh
  slurm_neurips_phase3_np_schedule.sh
  slurm_neurips_phase4_budget.sh
  slurm_neurips_phase5_mechanism.sh
  slurm_neurips_phase6_main.sh
  slurm_neurips_phase7_sitcom_masked_ablation.sh
  slurm_neurips_make_splits.sh
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Data

Provide a local dataset root via `--data_root` for all experiment scripts.
Scripts search by image basename (e.g. `09375.jpg`) recursively under `--data_root`.

## Canonical NeurIPS workflow

### 1) Generate fixed splits

```bash
python scripts/neurips_make_splits.py \
  --data_root /path/to/celeba_hq_256 \
  --outdir docs/neurips_splits
```

This writes standard split files including `dev_10.txt`, `validation_10.txt`, `validation_20.txt`, `validation_25.txt`, `test_20.txt`, and `test_50.txt`.

### 2) Canonical comparison (headline method vs baseline)

```bash
python scripts/neurips_canonical_compare.py \
  --data_root /path/to/celeba_hq_256 \
  --image_list_file docs/neurips_splits/test_50.txt \
  --outdir out_canonical \
  --radii 0.2 \
  --sitcom_variant unmasked
```

Outputs:
- `run_level.csv`
- `image_level.csv`
- per-radius config CSVs

### 3) Grid sweeps for phases 2-5

```bash
python scripts/neurips_grid_experiments.py \
  --mode np_schedule \
  --data_root /path/to/celeba_hq_256 \
  --image_list_file docs/neurips_splits/validation_10.txt \
  --outdir out_grid
```

Default compute-efficient behavior:
- `sitcom_lr`, `sitcom_noise`: runs **SITCOM only**.
- `np_schedule`, `mechanism`: runs **Noise Picking only**.
- `budget`: runs both methods.

Override with `--methods {auto,both,sitcom,noise_picking}`.

### 4) Postprocess grid outputs

```bash
python scripts/neurips_postprocess_grid.py --run_dir out_grid/np_schedule_YYYYMMDD_HHMMSS
```

This produces:
- `image_level.csv` (per-image mean/median/max PSNR and runtime/error summaries)
- `split_summary.csv` (averaged image-level summaries across the split)

## Slurm scripts

Use the NeurIPS phase scripts in `scripts/slurm_neurips_phase*.sh` and `scripts/slurm_neurips_make_splits.sh` for cluster runs.

## Reproducibility notes

- Seed lists are explicit and configurable.
- Radius controls masked constraints for Noise Picking (and masked SITCOM ablation).
- Phase 2-5 sweep defaults match current `neurips_grid_experiments.py` defaults (`5` seeds and `np_proj_start_values=200,400,600`).
