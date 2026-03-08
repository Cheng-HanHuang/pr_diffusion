#!/bin/bash --login
#SBATCH --job-name=prdiff_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/prdiff_diag_%j.out
#SBATCH --error=logs/prdiff_diag_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "=== BASIC INFO ==="
echo "DATE:   $(date)"
echo "HOST:   $(hostname)"
echo "PWD:    $(pwd)"
echo "JOBID:  ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

CONDA_ENV="dip"
DATA_ROOT="$HOME/data/prdiff_subset5"
IMAGE="09375.jpg"
OUTDIR="$HOME/out_prdiff_diag/$IMAGE"
MODEL_ID="google/ddpm-celebahq-256"

echo "=== PATHS ==="
echo "CONDA_ENV=$CONDA_ENV"
echo "DATA_ROOT=$DATA_ROOT"
echo "IMAGE=$IMAGE"
echo "OUTDIR=$OUTDIR"
echo "MODEL_ID=$MODEL_ID"

echo "=== CHECK DATA ==="
ls -la "$DATA_ROOT" || true
ls -la "$DATA_ROOT/$IMAGE" || true

echo "=== ACTIVATE CONDA ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== PYTHON INFO ==="
which python
python -V
python -c "import prdiffusion; print('prdiffusion ok:', prdiffusion.__file__)"

echo "=== SCRIPT HELP (ARGPARSE CHECK) ==="
python scripts/compare_methods_no_lowfreq.py --help | head -n 80

echo "=== TINY RUN (CPU) ==="
mkdir -p "$OUTDIR"

python scripts/compare_methods_no_lowfreq.py \
  --images "$IMAGE" \
  --data_root "$DATA_ROOT" \
  --outdir "$OUTDIR" \
  --model_id "$MODEL_ID" \
  --n_runs 1 \
  --base_seed 123 \
  --sitcom_outer_steps 1 \
  --sitcom_inner_steps 1 \
  --noise_picking_steps 5

echo "=== DONE ==="
