#!/bin/bash --login
#SBATCH --job-name=prdiff_neurips_p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs
CONDA_ENV="${CONDA_ENV:-dip}"; DATA_ROOT="${DATA_ROOT:-$HOME/data/prdiffusion_images}"; OUT_ROOT="${OUT_ROOT:-$HOME/out_prdiff_neurips}"; SPLIT_DIR="${SPLIT_DIR:-docs/neurips_splits}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
python scripts/neurips_grid_experiments.py --mode np_schedule --data_root "$DATA_ROOT" --image_list_file "$SPLIT_DIR/validation_10.txt" --outdir "$OUT_ROOT/phase3" --radius "${RADIUS:-0.2}"
LATEST_RUN_DIR=$(ls -td "$OUT_ROOT/phase3"/np_schedule_* 2>/dev/null | head -n1 || true)
if [ -n "$LATEST_RUN_DIR" ]; then
  python scripts/neurips_postprocess_grid.py --run_dir "$LATEST_RUN_DIR"
fi
