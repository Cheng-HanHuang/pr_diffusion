#!/usr/bin/env bash
# R0-N N1: PAC-only, fail-closed CPU test and analysis launcher.
# Usage: bash scripts/track_r/r0n_1_cpu_block.sh <committed-HEAD>
# Run after N0 has committed and pushed the original aggregate tables.
r0n_1_cpu_block() {
  local EXPECTED="${1:-}"
  local ROOT=/egr/research-pac/huang248
  local WT="$ROOT/pr_diffusion_track_r"
  local BASE="$ROOT/outputs/pr_diffusion/track_r"
  local CACHE="$ROOT/outputs/.cache/track_r"
  local PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
  local STAMP RUN LOG RC PID
  if [ -z "$EXPECTED" ]; then echo "STOP_N1_MISSING_HEAD"; return 3; fi
  if [ ! -x "$PY" ]; then echo "STOP_N1_PYTHON"; return 4; fi
  if [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "STOP_N1_WRONG_HEAD_OR_DIRTY"; return 5
  fi
  if [ ! -f "$WT/docs/track_r/executors/r0/returns/R0_20261023/SHA256SUMS.txt" ]; then
    echo "STOP_N1_N0_TABLES_NOT_COMMITTED"; return 6
  fi
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  RUN="$BASE/r0n_n1_$STAMP"
  LOG="$BASE/logs/r0n_n1_$STAMP.log"
  if [ -e "$RUN" ]; then echo "STOP_N1_RUN_EXISTS"; return 7; fi
  mkdir -p "$BASE/logs" "$CACHE/home" "$CACHE/tmp" "$CACHE/config" \
    "$CACHE/data" "$CACHE/state" "$CACHE/torch" "$CACHE/hf" "$CACHE/pip" || return $?
  export PYTHONPATH="$WT"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
  export HOME="$CACHE/home" TMPDIR="$CACHE/tmp"
  export XDG_CACHE_HOME="$CACHE" XDG_CONFIG_HOME="$CACHE/config"
  export XDG_DATA_HOME="$CACHE/data" XDG_STATE_HOME="$CACHE/state"
  export TORCH_HOME="$CACHE/torch" HF_HOME="$CACHE/hf"
  export PIP_CACHE_DIR="$CACHE/pip"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  cd "$WT" || return 8
  echo "N1_PINNED_HEAD=$EXPECTED"
  echo "N1_DIRTY_STATUS=$(git status --porcelain)"
  (
    cd "$WT/docs/track_r/executors/r0/returns/R0_20261023" || exit 9
    sha256sum -c SHA256SUMS.txt
  )
  RC=$?
  echo "R0N_N0_SHA_CHECK_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tests/track_r
  RC=$?
  echo "R0N_N1_CPU_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  CUDA_VISIBLE_DEVICES="" nohup "$PY" -u scripts/track_r/r0n_1_letter_tables_cpu.py \
    --expected-head "$EXPECTED" --out "$RUN" > "$LOG" 2>&1 &
  PID=$!
  printf '%s\n' "$RUN" > "$BASE/LATEST_R0N_N1_RUN"
  printf '%s\n' "$PID" > "$BASE/logs/r0n_n1_$STAMP.pid"
  echo "R0N_N1_PID=$PID"
  echo "R0N_N1_RUN=$RUN"
  echo "R0N_N1_LOG=$LOG"
}
r0n_1_cpu_block "$1"
