#!/bin/bash --login
#SBATCH --job-name=prdiff_lowfreq5
#SBATCH --partition=general-long-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gpus=h200:1
#SBATCH --array=0-4%1
#SBATCH --output=logs/prdiff_lowfreq5_%A_%a.out
#SBATCH --error=logs/prdiff_lowfreq5_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# HuggingFace caches (important on clusters)
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME

CONDA_ENV="dip"
DATA_ROOT="$HOME/data/prdiff_subset5"
OUT_ROOT="$HOME/out_hpc_compare_lowfreq_subset5"
MODEL_ID="google/ddpm-celebahq-256"
BASE_SEED=100
RADIUS_LIST="0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5"

IMAGES=(
  "00004.jpg"
  "09375.jpg"
  "09671.jpg"
  "10277.jpg"
  "19500.jpg"
)

N=${#IMAGES[@]}
if [ "${SLURM_ARRAY_TASK_ID:-999999}" -ge "$N" ]; then
  echo "ERROR: task id $SLURM_ARRAY_TASK_ID out of range (N=$N)"
  exit 1
fi
IMAGE="${IMAGES[$SLURM_ARRAY_TASK_ID]}"

start_ts=$(date +%s)
echo "START: $(date) | job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID image=$IMAGE host=$(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda run -n "$CONDA_ENV" python scripts/compare_methods_lowfreq_ablation.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUT_ROOT" \
  --model_id "$MODEL_ID" \
  --n_runs 10 \
  --base_seed "$BASE_SEED" \
  --radius_list "$RADIUS_LIST" \
  --sitcom_outer_steps 20 \
  --sitcom_inner_steps 20 \
  --noise_picking_steps 1000

end_ts=$(date +%s)
echo "END: $(date) | ELAPSED_SECONDS=$((end_ts-start_ts))"
echo "Outputs root: $OUT_ROOT"
