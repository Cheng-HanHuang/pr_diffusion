# Phase Retrieval with Diffusion Models (SITCOM + Noise Picking)

This repository contains reproducible phase-retrieval experiments built around a pretrained DDPM denoiser (default: `google/ddpm-celebahq-256`) and magnitude-only Fourier measurements.

It includes:
- **SITCOM** reconstruction. Paper at SITCOM.pdf in this repo.
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
README.md
LATEST_EXPERIMENT_RECORD.md
pyproject.toml
requirements.txt

docs/
  progress_report.md
  current_experiment_plan.md
  historical/                       # archived plans, logs, and execution notes

env/
  README.md
  machine.lab.env
  machine.*.env.example

prdiffusion/
  algorithms/
    sitcom.py
    noise_picking.py
    hybrid_np_sitcom.py
  diffusion.py
  fft_ops.py
  io.py
  metrics.py
  seed.py

scripts/
  # canonical / NeurIPS workflows
  neurips_canonical_compare.py
  neurips_grid_experiments.py
  neurips_postprocess_grid.py
  neurips_make_splits.py

  # current PRDiffusion workflows
  pr_canonical_compare.py
  pr_canonical_compare_v2.py
  pr_grid_experiments.py
  pr_postprocess_grid.py
  pr_mechanism_grid.py
  pr_phase8_9_schedule.py
  pr_phase10_11_np_grid.py
  pr_phase12_hybrid_ladder.py

  # ablations / legacy-compatible entry points
  compare_methods.py
  compare_methods_lowfreq_ablation.py
  compare_methods_no_lowfreq.py
  noise_picking_projstart_ablation.py
  sitcom_lr_sweep.py
  sitcom_noise_ablation.py

  # Slurm / cluster launch helpers
  pac_*.sh
  slurm_*.sh

  # external benchmarking (DiffFPR-style setting)
  pr_external_difffpr_np_benchmark.py
  pr_external_difffpr_np_paper_matrix.sh

scriptstemplate_h200.sh             # legacy shell template at repo root
```

External benchmark setup note:

- `docs/external_np_difffpr_benchmark_setup.md`

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
