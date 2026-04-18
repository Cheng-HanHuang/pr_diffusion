#!/usr/bin/env bash
set -euo pipefail

# ---- env bootstrap ----
source "$(conda info --base)/etc/profile.d/conda.sh"
source env/machine.lab.env
conda activate "${CONDA_ENV:-prdiff}"
cd "$REPO_ROOT"

# ---- user-tunable knobs ----
RADIUS="${RADIUS:-0.5}"
SEEDS="${SEEDS:-100,101,102,103,104,105,106,107,108,109}"

# GPU mapping (change these for your machine)
GPU_P4="${GPU_P4:-0}"
GPU_P5="${GPU_P5:-1}"
GPU_P6="${GPU_P6:-2}"
GPU_P7="${GPU_P7:-3}"

# Split files (change if needed)
SPLIT_P4="${SPLIT_P4:-$SPLIT_DIR/validation_25.txt}"
SPLIT_P5="${SPLIT_P5:-$SPLIT_DIR/validation_25.txt}"
SPLIT_P6="${SPLIT_P6:-$SPLIT_DIR/test_50.txt}"
SPLIT_P7="${SPLIT_P7:-$SPLIT_DIR/test_20.txt}"

# Output roots
OUT_P4="${OUT_P4:-$RUN_ROOT/phase4_budget_lab}"
OUT_P5="${OUT_P5:-$RUN_ROOT/phase5_mechanism_lab}"
OUT_P6="${OUT_P6:-$RUN_ROOT/phase6_main_lab}"
OUT_P7="${OUT_P7:-$RUN_ROOT/phase7_masked_ablation_lab}"

# Logs + pid tracking
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RUN_ROOT}/launch_logs_${STAMP}"
PID_FILE="${LOG_DIR}/pids.txt"
mkdir -p "$LOG_DIR" "$OUT_P4" "$OUT_P5" "$OUT_P6" "$OUT_P7"
: > "$PID_FILE"

run_bg () {
  local name="$1"; shift
  local gpu="$1"; shift
  local log="${LOG_DIR}/${name}.log"

  echo "Launching ${name} on GPU ${gpu} ..."
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "$*" > "$log" 2>&1 &
  local pid=$!
  echo "${name},${pid},gpu=${gpu},log=${log}" | tee -a "$PID_FILE"
}

# Phase 4: budget (grid mode)
run_bg "phase4_budget" "$GPU_P4" \
"source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
 source env/machine.lab.env && conda activate \"\${CONDA_ENV:-prdiff}\" && cd \"\$REPO_ROOT\" && \
 python scripts/neurips_grid_experiments.py \
   --mode budget \
   --data_root \"\$DATA_ROOT\" \
   --image_list_file \"$SPLIT_P4\" \
   --outdir \"$OUT_P4\" \
   --radius \"$RADIUS\" \
   --seeds \"$SEEDS\" && \
 LATEST=\$(ls -dt \"$OUT_P4\"/budget_* 2>/dev/null | head -n 1 || true) && \
 [ -n \"\$LATEST\" ] && python scripts/neurips_postprocess_grid.py --run_dir \"\$LATEST\" || true"

# Phase 5: mechanism
run_bg "phase5_mechanism" "$GPU_P5" \
"source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
 source env/machine.lab.env && conda activate \"\${CONDA_ENV:-prdiff}\" && cd \"\$REPO_ROOT\" && \
 python scripts/neurips_grid_experiments.py \
   --mode mechanism \
   --data_root \"\$DATA_ROOT\" \
   --image_list_file \"$SPLIT_P5\" \
   --outdir \"$OUT_P5\" \
   --radius \"$RADIUS\" \
   --methods noise_picking \
   --seeds \"$SEEDS\" && \
 LATEST=\$(ls -dt \"$OUT_P5\"/mechanism_* 2>/dev/null | head -n 1 || true) && \
 [ -n \"\$LATEST\" ] && python scripts/neurips_postprocess_grid.py --run_dir \"\$LATEST\" || true"

# Phase 6: main (canonical, SITCOM unmasked)
run_bg "phase6_main" "$GPU_P6" \
"source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
 source env/machine.lab.env && conda activate \"\${CONDA_ENV:-prdiff}\" && cd \"\$REPO_ROOT\" && \
 python scripts/neurips_canonical_compare.py \
   --data_root \"\$DATA_ROOT\" \
   --outdir \"$OUT_P6\" \
   --image_list_file \"$SPLIT_P6\" \
   --radii \"$RADIUS\" \
   --seeds \"$SEEDS\" \
   --sitcom_variant unmasked \
   --sitcom_lr 0.02 \
   --np_num_candidates_soft 5 \
   --np_num_candidates_hard 1 \
   --np_proj_start 400"

# Phase 7: SITCOM masked ablation
run_bg "phase7_masked_ablation" "$GPU_P7" \
"source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
 source env/machine.lab.env && conda activate \"\${CONDA_ENV:-prdiff}\" && cd \"\$REPO_ROOT\" && \
 python scripts/neurips_canonical_compare.py \
   --data_root \"\$DATA_ROOT\" \
   --outdir \"$OUT_P7\" \
   --image_list_file \"$SPLIT_P7\" \
   --radii \"$RADIUS\" \
   --seeds \"$SEEDS\" \
   --sitcom_variant masked \
   --sitcom_lr 0.02 \
   --np_num_candidates_soft 5 \
   --np_num_candidates_hard 1 \
   --np_proj_start 400"

echo
echo "Launched all jobs."
echo "PID file: $PID_FILE"
echo "Logs dir:  $LOG_DIR"
echo
echo "To monitor:"
echo "  tail -f \"$LOG_DIR\"/*.log"
echo "  ps -fp \$(awk -F, '{print \$2}' \"$PID_FILE\" | tr '\n' ' ')"
echo
echo "To stop all:"
echo "  awk -F, '{print \$2}' \"$PID_FILE\" | xargs -r kill"
