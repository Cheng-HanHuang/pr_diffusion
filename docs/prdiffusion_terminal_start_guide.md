# prdiffusion terminal startup guide

This is the standard checklist to follow **every time you open a new terminal** for the `pr_diffusion` project.

It is written for the current setup:

- repo: `/mnt/home/huang248/pr_diffusion_repo`
- data: `/mnt/home/huang248/data/celeba_hq_256`
- default run root: `/mnt/home/huang248/runs/pr_diffusion/neurips_20260411`
- default conda env: `dip`

The goal is to make terminal startup consistent and prevent mistakes with:
- wrong working directory
- wrong data path
- scattered outputs
- forgetting which split/output folder is being used

---

## 1. Go to the repo root

Always start by entering the repo root:

```bash
cd /mnt/home/huang248/pr_diffusion_repo
```

This matters because the SLURM scripts use relative paths like `scripts/...` and assume the submit directory is the repo root.

---

## 2. Check repo status and update if needed

Run:

```bash
git status
```

If the working tree is clean and you want the latest version:

```bash
git checkout main
git pull origin main
```

If there are local changes you want to keep, either commit them first or stash them before pulling.

---

## 3. Activate the conda environment

Run:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dip
```

To verify:

```bash
which python
python -V
```

---

## 4. Set the standard project paths

Run this block in the terminal:

```bash
export REPO_ROOT=/mnt/home/huang248/pr_diffusion_repo
export DATA_ROOT=/mnt/home/huang248/data/celeba_hq_256
export RUN_ROOT=/mnt/home/huang248/runs/pr_diffusion/neurips_20260411
export SPLIT_DIR=$RUN_ROOT/splits
export CONDA_ENV=dip
```

These variables are the standard ones to use when submitting jobs.

---

## 5. Make sure the output folders exist

Run:

```bash
mkdir -p $RUN_ROOT
mkdir -p $SPLIT_DIR
mkdir -p $RUN_ROOT/phase0
mkdir -p $RUN_ROOT/phase1
mkdir -p $RUN_ROOT/phase2
mkdir -p $RUN_ROOT/phase3
mkdir -p $RUN_ROOT/phase4
mkdir -p $RUN_ROOT/phase5
mkdir -p $RUN_ROOT/phase6
mkdir -p $RUN_ROOT/phase7
```

This keeps outputs under one clean campaign root instead of scattering many `out_*` folders in `/mnt/home/huang248`.

---

## 6. Optional quick sanity checks

Before submitting anything, it is useful to confirm the paths exist:

```bash
ls $REPO_ROOT
ls $DATA_ROOT | head
```

Also useful:

```bash
test -d $DATA_ROOT && echo "DATA_ROOT OK"
test -d $REPO_ROOT && echo "REPO_ROOT OK"
```

---

## 7. Standard job submission pattern

Always submit from the repo root:

```bash
cd $REPO_ROOT
```

Then pass paths through environment variables rather than editing the scripts.

General pattern:

```bash
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/<some_slurm_script>.sh
```

For split generation, use `OUT_SPLIT_DIR` instead of `OUT_ROOT`:

```bash
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/slurm_neurips_make_splits.sh
```

---

## 8. Typical first-wave commands

When starting a fresh experiment campaign, do these in order.

### 8.1 Generate fixed splits

```bash
cd $REPO_ROOT
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/slurm_neurips_make_splits.sh
```

### 8.2 Submit Phase 0 sanity

```bash
cd $REPO_ROOT
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
sbatch scripts/slurm_neurips_phase0_sanity.sh
```

### 8.3 Submit Phase 2 SITCOM tuning

```bash
cd $REPO_ROOT
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/slurm_neurips_phase2_sitcom_tuning.sh
```

### 8.4 Submit Phase 1 radius validation

```bash
cd $REPO_ROOT
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
SPLIT_DIR=$SPLIT_DIR \
sbatch scripts/slurm_neurips_phase1_radius.sh
```

These are the standard first jobs to submit.

---

## 9. Standard radius-dependent submission pattern

After Phase 1 finishes and you choose a radius (for example `0.2`), submit later phases like this:

```bash
cd $REPO_ROOT
CONDA_ENV=$CONDA_ENV \
DATA_ROOT=$DATA_ROOT \
OUT_ROOT=$RUN_ROOT \
SPLIT_DIR=$SPLIT_DIR \
RADIUS=0.2 \
sbatch scripts/slurm_neurips_phase3_np_schedule.sh
```

The same pattern applies to phases 4, 5, 6, and 7.

---

## 10. How to check job queue and logs

Check queued/running jobs:

```bash
squeue -u huang248
```

Cancel a job if needed:

```bash
scancel <jobid>
```

The SLURM scripts write logs under the repo’s `logs/` folder because they submit from the repo root.

Check recent logs:

```bash
ls -lt logs | head
```

Watch a running log:

```bash
tail -f logs/<jobname>_<jobid>.out
```

---

## 11. How to inspect outputs after a phase finishes

### Splits

```bash
ls $SPLIT_DIR
```

### Phase outputs

Example:

```bash
ls $RUN_ROOT/phase1
```

Most phases create timestamped run folders.

Useful files to look for:
- `run_level.csv`
- `image_level.csv`
- `split_summary.csv` for grid phases

---

## 12. The standard "start terminal" checklist

Every time you start a new terminal, do this:

```bash
cd /mnt/home/huang248/pr_diffusion_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dip

export REPO_ROOT=/mnt/home/huang248/pr_diffusion_repo
export DATA_ROOT=/mnt/home/huang248/data/celeba_hq_256
export RUN_ROOT=/mnt/home/huang248/runs/pr_diffusion/neurips_20260411
export SPLIT_DIR=$RUN_ROOT/splits
export CONDA_ENV=dip
```

Then optionally:

```bash
mkdir -p $RUN_ROOT $SPLIT_DIR
cd $REPO_ROOT
git status
```

That is the minimum startup routine.

---

## 13. Recommended convention for future campaigns

For a new experiment campaign, create a new run root like:

```text
/mnt/home/huang248/runs/pr_diffusion/neurips_YYYYMMDD
```

Examples:
- `/mnt/home/huang248/runs/pr_diffusion/neurips_20260411`
- `/mnt/home/huang248/runs/pr_diffusion/neurips_20260418`

This keeps each campaign self-contained and makes reruns easier to manage.

---

## 14. Common mistakes to avoid

Do not:
- submit jobs from outside the repo root
- hardcode paths inside Python scripts unless necessary
- mix outputs from multiple campaigns into the same random folder
- forget to set `RADIUS` for radius-dependent phases
- tune on the test split

Do:
- keep one clean `RUN_ROOT`
- pass paths via environment variables at submit time
- check `git status` before running new experiments
- inspect `image_level.csv` and `split_summary.csv` after runs finish

---

## 15. One-line quick start block

If you just want the shortest reusable startup block, use this every time:

```bash
cd /mnt/home/huang248/pr_diffusion_repo && \
source "$(conda info --base)/etc/profile.d/conda.sh" && \
conda activate dip && \
export REPO_ROOT=/mnt/home/huang248/pr_diffusion_repo && \
export DATA_ROOT=/mnt/home/huang248/data/celeba_hq_256 && \
export RUN_ROOT=/mnt/home/huang248/runs/pr_diffusion/neurips_20260411 && \
export SPLIT_DIR=$RUN_ROOT/splits && \
export CONDA_ENV=dip
```

Then continue with whichever submit command you need.
