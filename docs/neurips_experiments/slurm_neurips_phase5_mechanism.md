## SLURM job description
This SLURM script runs one NeurIPS experiment phase on the institutional HPC with `#!/bin/bash --login`, creates `logs/`, activates conda, and executes the matching Python experiment script from `$SLURM_SUBMIT_DIR`.

## Parameters
- 1 node, 1 task, 1 H200 GPU.
- `CONDA_ENV` default: `dip`.
- Data/output/split paths are configurable through environment variables in each `.sh` file.
- Walltime is conservative and sized per phase.

## Expected runtime
Use the plan assumptions:
- SITCOM 1000 steps: ~800 sec/run; SITCOM 20 steps: ~15 sec/run.
- Noise Picking default run: ~60 min/run.

So runtime scales with number of images × restarts × radii/settings.
