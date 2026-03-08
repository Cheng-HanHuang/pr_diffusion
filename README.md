# Phase Retrieval with Diffusion Models (SITCOM + Noise Picking)

This repository contains reproducible phase-retrieval experiments built around a pretrained DDPM denoiser (default: `google/ddpm-celebahq-256`) and magnitude-only Fourier measurements.

It includes:
- **SITCOM** reconstruction (with optional paper-faithful backprop through the denoiser).
- **Noise Picking** reconstruction.
- End-to-end experiment scripts for sweeps/ablations/comparisons.
- **Institution HPC Slurm templates** for diagnostics, smoke tests, and full runs.

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
  compare_methods.py
  compare_methods_no_lowfreq.py
  sitcom_lr_sweep.py
  sitcom_noise_ablation.py
  make_celeba_subset5.py

  # Slurm scripts
  slurm_diag.sh
  slurm_smoke_compare_no_lowfreq.sh
  slurm_compare_subset5.sh
  slurm_compare_subset5_full.sh
  slurm_compare_no_lowfreq.sh

scriptstemplate_h200.sh   # generic institution template for future jobs
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Minimal manual install (if preferred):

```bash
pip install torch torchvision diffusers pillow numpy
# Optional, only for in-script Kaggle download:
pip install kagglehub
```

## Data

You can use either:
1. A local `celeba_hq_256`-style folder (recommended for HPC), or
2. Kaggle download fallback (only if outbound network/auth is available).

Scripts search by image basename (e.g. `09375.jpg`) recursively under `--data_root`.

## Experiment scripts

### 1) SITCOM learning-rate sweep
Sweeps `lr_inner` over a list for one image and logs per-seed metrics.

```bash
python scripts/sitcom_lr_sweep.py \
  --image 09375.jpg \
  --data_root /path/to/celeba_hq_256 \
  --outdir out_sitcom_lr \
  --lr_list 0.01,0.02,0.05,0.1 \
  --num_steps 1000 --K 20 --lam 0.1
```

### 2) SITCOM noise ablation
Sweeps `eta_scale × init_scale` with fixed `lr`.

```bash
python scripts/sitcom_noise_ablation.py \
  --image 09375.jpg \
  --data_root /path/to/celeba_hq_256 \
  --outdir out_sitcom_noise \
  --eta_list 0.0,0.25,0.5,1.0 \
  --init_list 0.75,1.0,1.25 \
  --lr 0.05
```

### 3) Method comparison (original)
Compares SITCOM vs Noise Picking while sweeping Noise Picking `score_radius`.

```bash
python scripts/compare_methods.py \
  --image 09375.jpg \
  --data_root /path/to/celeba_hq_256 \
  --outdir out_compare \
  --score_radii 0.005,0.01,0.02,0.03,0.04,0.05 \
  --n_runs 4
```

### 4) Method comparison with low-frequency masking disabled
Primary current comparison flow for institution runs.

```bash
python scripts/compare_methods_no_lowfreq.py \
  --images 09375.jpg,09671.jpg \
  --data_root /path/to/celeba_hq_256 \
  --outdir out_compare_no_lowfreq \
  --n_runs 10 --base_seed 100 \
  --sitcom_outer_steps 20 --sitcom_inner_steps 20 \
  --noise_picking_steps 1000
```

### 5) Build 5-image subset + generate Slurm array script
Creates a 5-image subset (always includes `09375.jpg` and `09671.jpg`, plus 3 random images), writes a manifest, and generates `scripts/slurm_compare_subset5.sh`.

```bash
python scripts/make_celeba_subset5.py \
  --dataset_root /path/to/celeba_hq_256 \
  --subset_dir "$HOME/data/prdiff_subset5" \
  --seed 123 --conda_env dip
```

## Slurm scripts

- `scripts/slurm_diag.sh`: environment/data/import sanity checks + tiny run.
- `scripts/slurm_smoke_compare_no_lowfreq.sh`: quick smoke test configuration.
- `scripts/slurm_compare_subset5.sh`: array job over 5-image subset (per-image outputs).
- `scripts/slurm_compare_subset5_full.sh`: full subset run with HF cache environment setup.
- `scripts/slurm_compare_no_lowfreq.sh`: minimal two-image launcher variant.
- `scriptstemplate_h200.sh`: generic institution H200 template to copy when writing new jobs.

## Institution HPC framework note (important)

For future HPC scripts, **always start from `scriptstemplate_h200.sh` (or an existing Slurm script already aligned to it)** to preserve a consistent institutional framework (logs pathing, conda activation pattern, cache setup style, and job metadata logging).

> All parameters are expected to be customized per experiment (time, memory, GPU type/count, array size, script arguments, etc.). The template is a **framework baseline**, not a fixed configuration.

## Outputs

Experiment scripts typically write:
- A timestamped CSV summary with run-level metrics and settings.
- Optional PNG reconstructions (`--save_png` where supported).
- In `compare_methods_no_lowfreq.py`, per-image folders with `configs.csv`, `gt.png`, and method reconstructions.

## Reproducibility notes

- Seeds are explicit (`--seeds` or `--base_seed` + run index).
- `eta_scale` controls re-noise strength in SITCOM resampling.
- `init_scale` controls initial latent noise scale.
- `--plugin_denoiser` toggles off denoiser backprop in SITCOM for ablations.
