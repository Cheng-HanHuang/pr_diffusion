#!/bin/bash --login
#SBATCH --job-name=prdiff_neurips_p0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-dip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/prdiffusion_images}"
OUT_ROOT="${OUT_ROOT:-$HOME/out_prdiff_neurips}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
python scripts/neurips_canonical_compare.py --data_root "$DATA_ROOT" --outdir "$OUT_ROOT/phase0" \
  --images "00004.jpg,09375.jpg,09671.jpg,10277.jpg,19500.jpg" --radii "0.1,0.2,0.5" --seeds "100,101,102,103,104,105,106,107,108,109" --sitcom_variant unmasked
