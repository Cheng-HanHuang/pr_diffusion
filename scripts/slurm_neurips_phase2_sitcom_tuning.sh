#!/bin/bash --login
#SBATCH --job-name=prdiff_neurips_p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs
CONDA_ENV="${CONDA_ENV:-dip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/prdiffusion_images}"
OUT_ROOT="${OUT_ROOT:-$HOME/out_prdiff_neurips}"
SPLIT_DIR="${SPLIT_DIR:-docs/neurips_splits}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
python scripts/neurips_grid_experiments.py --mode sitcom_lr --data_root "$DATA_ROOT" --image_list_file "$SPLIT_DIR/dev_10.txt" --outdir "$OUT_ROOT/phase2"
python scripts/neurips_grid_experiments.py --mode sitcom_noise --data_root "$DATA_ROOT" --image_list_file "$SPLIT_DIR/dev_10.txt" --outdir "$OUT_ROOT/phase2"
for mode_prefix in sitcom_lr sitcom_noise; do
  LATEST_RUN_DIR=$(ls -td "$OUT_ROOT/phase2"/${mode_prefix}_* 2>/dev/null | head -n1 || true)
  if [ -n "$LATEST_RUN_DIR" ]; then
    python scripts/neurips_postprocess_grid.py --run_dir "$LATEST_RUN_DIR"
  fi
done
