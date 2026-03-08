#!/bin/bash --login
#SBATCH --job-name=prdiff_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gpus=h200:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

CONDA_ENV="dip"
DATA_ROOT="$HOME/data/prdiff_subset5"
OUTDIR="$HOME/out_prdiff_smoke_no_lowfreq"
MODEL_ID="google/ddpm-celebahq-256"
IMAGE="09375.jpg"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda run -n "$CONDA_ENV" python scripts/compare_methods_no_lowfreq.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --model_id "$MODEL_ID" \
  --n_runs 1 \
  --base_seed 123 \
  --sitcom_outer_steps 3 \
  --sitcom_inner_steps 3 \
  --noise_picking_steps 20

echo "DONE. Outputs in: $OUTDIR"
