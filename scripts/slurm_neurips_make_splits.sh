#!/bin/bash --login
#SBATCH --job-name=prdiff_neurips_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-dip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/prdiffusion_images}"
OUT_SPLIT_DIR="${OUT_SPLIT_DIR:-docs/neurips_splits}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/neurips_make_splits.py \
  --data_root "$DATA_ROOT" \
  --outdir "$OUT_SPLIT_DIR"
