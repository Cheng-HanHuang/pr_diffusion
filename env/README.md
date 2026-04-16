# Machine environment profiles

This directory stores **template** environment profiles for machine-specific settings.

> Do not commit private credentials or personal tokens.

## Files

- `machine.lab.env.example`: Example profile for PAC/lab machine runs.
- `machine.institution.env.example`: Example profile for institution machine runs.

## Usage

1. Copy a template locally:

```bash
cp env/machine.lab.env.example env/machine.lab.env
```

2. Edit paths/env names.
3. Load before running scripts:

```bash
source env/machine.lab.env
```

4. Run experiments according to machine mode:

**Institution machine (SLURM examples):**

```bash
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/slurm_neurips_phase6_main.sh
```

**PAC / lab machine (direct Python, no SLURM):**

Use `docs/pac_direct_python_workflow.md` as the canonical run guide.

## Variables

- `REPO_ROOT`
- `DATA_ROOT`
- `RUN_ROOT`
- `SPLIT_DIR`
- `CONDA_ENV`
- `HF_HOME` (optional)
- `PYTHONPATH` (recommended for direct script execution)
- `CUDA_VISIBLE_DEVICES` (optional for local single-GPU runs)
