#!/bin/bash --login
#SBATCH --job-name=prdiff_neurips_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --gpus=h200:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
CONDA_ENV="${CONDA_ENV:-dip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/prdiffusion_images}"
OUT_ROOT="${OUT_ROOT:-$HOME/out_prdiff_neurips}"
SPLIT_DIR="${SPLIT_DIR:-docs/neurips_splits}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
python scripts/neurips_canonical_compare.py --data_root "$DATA_ROOT" --outdir "$OUT_ROOT/phase1" \
  --image_list_file "$SPLIT_DIR/validation_25.txt" --radii "0.1,0.2,0.5" --seeds "100,101,102,103,104,105,106,107,108,109" --sitcom_variant unmasked
