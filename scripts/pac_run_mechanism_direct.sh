#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-prdiff}"

: "${REPO_ROOT:?Set REPO_ROOT (e.g., source env/machine.lab.env)}"
: "${DATA_ROOT:?Set DATA_ROOT (e.g., source env/machine.lab.env)}"
: "${RUN_ROOT:?Set RUN_ROOT (e.g., source env/machine.lab.env)}"
: "${SPLIT_DIR:?Set SPLIT_DIR (e.g., source env/machine.lab.env)}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/mechanism_lab"

python scripts/neurips_grid_experiments.py \
  --mode mechanism \
  --data_root "$DATA_ROOT" \
  --image_list_file "$SPLIT_DIR/validation_10.txt" \
  --outdir "$RUN_ROOT/mechanism_lab" \
  --radius "${RADIUS:-0.5}" \
  --methods noise_picking

LATEST_RUN_DIR=$(ls -dt "$RUN_ROOT"/mechanism_lab/mechanism_* 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_RUN_DIR" ]; then
  python scripts/neurips_postprocess_grid.py --run_dir "$LATEST_RUN_DIR"
fi
