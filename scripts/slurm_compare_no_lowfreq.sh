#!/bin/bash
#SBATCH --job-name=prdiff_compare_no_low_freq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/prdiff_compare_%A_%a.out
#SBATCH --error=logs/prdiff_compare_%A_%a.err

set -euo pipefail

# Usage:
#   sbatch scripts/slurm_compare_no_lowfreq.sh
#
# Required env vars to adapt on your cluster:
#   CONDA_ENV      (e.g. prdiff)
#   DATA_ROOT      (path to CelebA-HQ-256 image folder)
# Optional env vars:
#   OUT_ROOT       (default: out_hpc_compare_no_lowfreq)
#   MODEL_ID       (default: google/ddpm-celebahq-256)
#   BASE_SEED      (default: 100)

mkdir -p logs

: "${CONDA_ENV:=dip}"
: "${DATA_ROOT:=/path/to/celeba_hq_256}"
: "${OUT_ROOT:=out_hpc_compare_no_lowfreq}"
: "${MODEL_ID:=google/ddpm-celebahq-256}"
: "${BASE_SEED:=100}"

IMAGES=("09375.jpg" "09671.jpg")
IMAGE="${IMAGES[$SLURM_ARRAY_TASK_ID]}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/compare_methods_no_lowfreq.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUT_ROOT/$IMAGE" \
  --model_id "$MODEL_ID" \
  --n_runs 10 \
  --base_seed "$BASE_SEED" \
  --sitcom_outer_steps 20 \
  
