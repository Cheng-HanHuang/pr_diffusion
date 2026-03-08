#!/bin/bash --login
#SBATCH --job-name=prdiff_compare_subset5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# GPU request (keep this if it works on QUEST)
#SBATCH --gpus=h200:1

# If your cluster doesn't support --gpus=h200:1, use these instead:
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --constraint=h200

# Array for 5 images; "%1" means run 1 task at a time
#SBATCH --array=0-4%1

#SBATCH --output=logs/prdiff_compare_%A_%a.out
#SBATCH --error=logs/prdiff_compare_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

CONDA_ENV="dip"
DATA_ROOT="$HOME/data/prdiff_subset5"
OUT_ROOT="$HOME/out_hpc_compare_no_lowfreq"
MODEL_ID="google/ddpm-celebahq-256"
BASE_SEED=100

IMAGES=(
  "00004.jpg"
  "09375.jpg"
  "09671.jpg"
  "10277.jpg"
  "19500.jpg"
)

N=${#IMAGES[@]}
if [ "${SLURM_ARRAY_TASK_ID:-999999}" -ge "$N" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset} out of range (N=$N)."
  exit 1
fi
IMAGE="${IMAGES[$SLURM_ARRAY_TASK_ID]}"

start_ts=$(date +%s)
echo "START: $(date) | job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID image=$IMAGE"
echo "HOST:  $(hostname)"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python scripts/compare_methods_no_lowfreq.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUT_ROOT/$IMAGE" \
  --model_id "$MODEL_ID" \
  --n_runs 10 \
  --base_seed "$BASE_SEED" \
  --sitcom_outer_steps 20 \
  --sitcom_inner_steps 20 \
  --noise_picking_steps 1000

end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
echo "END: $(date) | ELAPSED_SECONDS=$elapsed"
