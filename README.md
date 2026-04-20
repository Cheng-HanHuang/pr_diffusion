# Phase Retrieval with Diffusion Models (SITCOM + Noise Picking)

This repository contains reproducible phase-retrieval experiments built around a pretrained DDPM denoiser (default: `google/ddpm-celebahq-256`) and magnitude-only Fourier measurements.

It includes:
- **SITCOM** reconstruction.
- **Noise Picking** reconstruction.
- experiment runners for canonical comparison, phase sweeps, and hybrid studies.
- Slurm / launch templates for multi-phase HPC execution.

### Current planning pointers

- Main progress summary: `docs/progress_report.md`
- Active experiment plan: `docs/current_experiment_plan.md`
- Historical planning / execution notes: `docs/historical/README.md`

### Current major machine settings

- PAC, the lab machine and the major machine now, default to but notice changes can be made: 'env/machine.lab.env'

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
  neurips_canonical_compare.py      # canonical comparison flow
  neurips_grid_experiments.py       # phase 2-5 sweeps
  neurips_postprocess_grid.py       # run_level -> image_level/split_summary
  neurips_make_splits.py            # fixed split generation

  pr_phase8_9_schedule.py           # scheduled SITCOM variants
  pr_phase10_11_np_grid.py          # NP mechanism decoupling / hard-late studies
  pr_phase12_hybrid_ladder.py       # NP -> SITCOM hybrid ladder

  # legacy scripts retained for backward compatibility
  compare_methods.py
  compare_methods_no_lowfreq.py
  sitcom_lr_sweep.py
  sitcom_noise_ablation.py
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Data

Provide a local dataset root via `--data_root` for all experiment scripts.
Scripts search by image basename (e.g. `09375.jpg`) recursively under `--data_root`.

## Reproducibility notes

- Seed lists are explicit and configurable.
- Radius controls masked constraints for Noise Picking and scheduled SITCOM variants.
- `docs/progress_report.md` is the main source of frozen defaults and current status.
- `docs/current_experiment_plan.md` is the active planning document.
