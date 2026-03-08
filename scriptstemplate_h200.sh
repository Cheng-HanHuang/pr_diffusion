#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Always run from where you submitted the job
cd "${SLURM_SUBMIT_DIR}"

# Make sure logs folder exists
mkdir -p logs

CONDA_ENV="dip"
DATA_ROOT="$HOME/data/prdiff_subset5"
OUT_ROOT="$HOME/out_prdiff_runs"
MODEL_ID="google/ddpm-celebahq-256"

# Optional: HF cache dirs (recommended on HPC)
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

echo "START: $(date) job=$SLURM_JOB_ID host=$(hostname)"
echo "PWD:   $(pwd)"
echo "CONDA_ENV=$CONDA_ENV"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "MODEL_ID=$MODEL_ID"

# Activate conda and run your command(s)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Optional: confirm GPU
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# ---- Put your python command here ----
# conda run -n "$CONDA_ENV" python scripts/your_script.py --args ...
# --------------------------------------

echo "END: $(date)"
