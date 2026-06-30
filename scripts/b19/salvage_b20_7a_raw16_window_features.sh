#!/usr/bin/env bash
set -euo pipefail

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
BASE=/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver

ID="${ID:-00046}"
MEAS_SEED="${MEAS_SEED:-5001}"
NUM_RUNS="${NUM_RUNS:-16}"
RUN_SEEDS=(${RUN_SEEDS:-4400 4500 4600 4700 4900 5500})

cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate daps

for RS in "${RUN_SEEDS[@]}"; do
  echo
  echo "================ salvage run_seed=$RS ================"

  ROOT="$REPO/external/daps/results/b20_7a_rawtraj_${NUM_RUNS}S_meas${MEAS_SEED}_runseed${RS}/b20_7a_rawtraj_${ID}_meas${MEAS_SEED}_runseed${RS}_${NUM_RUNS}S"
  MEAS="$BASE/measurements/ffhq${ID}_phase_noise005_meas${MEAS_SEED}.pt"
  METRICS="$ROOT/metrics.json"

  STEP_OUT="$BASE/B20_7A_daps_${ID}_meas${MEAS_SEED}_runseed${RS}_${NUM_RUNS}S_rawtraj_step_features.csv"
  WINDOW_OUT="$BASE/B20_7A_daps_${ID}_meas${MEAS_SEED}_runseed${RS}_${NUM_RUNS}S_rawtraj_window_features.csv"
  LOG="$BASE/B20_7A_daps_${ID}_meas${MEAS_SEED}_runseed${RS}_${NUM_RUNS}S_rawtraj_window_features.log"

  if [ -f "$WINDOW_OUT" ]; then
    echo "[skip] window exists: $WINDOW_OUT"
    continue
  fi

  if [ ! -f "$METRICS" ]; then
    echo "[error] missing metrics: $METRICS"
    continue
  fi

  if [ ! -f "$MEAS" ]; then
    echo "[error] missing measurement: $MEAS"
    continue
  fi

  echo "[root] $ROOT"
  echo "[raw-ish files]"
  find "$ROOT" -maxdepth 4 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.npz' \) | head -20

  # Candidate raw dirs:
  # 1. common names
  # 2. dirs with many raw-ish files
  mapfile -t DETECTED_DIRS < <(
    find "$ROOT" -maxdepth 4 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.npz' \) -printf '%h\n' \
      | sort \
      | uniq -c \
      | sort -nr \
      | awk '{$1=""; sub(/^ /,""); print}'
  )

  CANDIDATES=(
    "$ROOT/raw"
    "$ROOT/raw_data"
    "$ROOT/raw_traj"
    "$ROOT/traj"
    "$ROOT/trajs"
    "$ROOT/trajectory"
    "$ROOT/trajectories"
    "$ROOT"
  )

  SUCCESS=0

  for RAW_DIR in "${CANDIDATES[@]}" "${DETECTED_DIRS[@]}"; do
    if [ ! -d "$RAW_DIR" ]; then
      continue
    fi

    echo "[try raw_dir] $RAW_DIR"

    TMP_STEP="${STEP_OUT}.tmp"
    TMP_WINDOW="${WINDOW_OUT}.tmp"
    TMP_LOG="${LOG}.tmp"

    rm -f "$TMP_STEP" "$TMP_WINDOW" "$TMP_LOG"

    set +e
    python scripts/b19/analyze_daps_rawtraj_early_features.py \
      --daps_root external/daps \
      --raw_dir "$RAW_DIR" \
      --measurement_path "$MEAS" \
      --metrics_json "$METRICS" \
      --out_step_csv "$TMP_STEP" \
      --out_window_csv "$TMP_WINDOW" \
      --noise 0.05 \
      --oversample 2.0 \
      --good_threshold 25.0 > "$TMP_LOG" 2>&1
    STATUS=$?
    set -e

    if [ "$STATUS" -eq 0 ] && [ -f "$TMP_WINDOW" ]; then
      mv "$TMP_STEP" "$STEP_OUT"
      mv "$TMP_WINDOW" "$WINDOW_OUT"
      mv "$TMP_LOG" "$LOG"
      echo "[success] raw_dir=$RAW_DIR"
      echo "[write] $STEP_OUT"
      echo "[write] $WINDOW_OUT"
      SUCCESS=1
      break
    else
      echo "[failed raw_dir] $RAW_DIR"
      tail -20 "$TMP_LOG" 2>/dev/null || true
    fi
  done

  if [ "$SUCCESS" -ne 1 ]; then
    echo "[error] no raw_dir worked for run_seed=$RS"
  fi
done
