#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-prdiff}"

: "${REPO_ROOT:?Set REPO_ROOT (e.g., source env/machine.lab.env)}"
: "${DATA_ROOT:?Set DATA_ROOT (e.g., source env/machine.lab.env)}"
: "${RUN_ROOT:?Set RUN_ROOT (e.g., source env/machine.lab.env)}"
: "${SPLIT_DIR:?Set SPLIT_DIR (e.g., source env/machine.lab.env)}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/canonical_test20_lab"

python scripts/neurips_canonical_compare.py \
  --data_root "$DATA_ROOT" \
  --outdir "$RUN_ROOT/canonical_test20_lab" \
  --image_list_file "$SPLIT_DIR/test_20.txt" \
  --radii "${RADIUS:-0.5}" \
  --seeds "${SEEDS:-100,101,102,103,104,105,106,107,108,109}" \
  --sitcom_variant unmasked \
  --sitcom_lr "${SITCOM_LR:-0.02}" \
  --np_num_candidates_soft "${NP_SOFT:-5}" \
  --np_num_candidates_hard "${NP_HARD:-1}" \
  --np_proj_start "${NP_PROJ_START:-400}"

echo "Saved canonical PAC pilot outputs under: $RUN_ROOT/canonical_test20_lab"
