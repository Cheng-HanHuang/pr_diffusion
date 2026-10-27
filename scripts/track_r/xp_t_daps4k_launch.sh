#!/usr/bin/env bash
# Guarded GT-free DAPS-4K native calibration: 2 smoke then 16 main W1 jobs.
# Upstream native tree is pristine sparse e7a77d0; no original source edit.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_t_daps4k_20261027T011832Z"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: DAPS4K_WRONG_OR_DIRTY_XP_HEAD"
    return 8
  fi
  if [ ! -f "$RUN/XP_T_DAPS4K_PREP.json" ] ||
     [ ! -f "$RUN/XP_T_DAPS4K_TASKS.json" ] ||
     [ -e "$RUN/XP_T_DAPS4K_STATUS.json" ] ||
     [ -e "$RUN/gpu" ] ||
     [ -e "$RUN/XP_T_DAPS4K_SUPERVISOR.log" ] ||
     [ -e "$RUN/XP_T_DAPS4K_SUPERVISOR_PID.json" ]; then
    echo "ACTION NEEDED: DAPS4K_FROZEN_INPUT_MISSING_OR_ALREADY_STARTED"
    return 8
  fi
  echo "=== DAPS4K WORKER / SUPERVISOR REGRESSIONS (CPU ONLY) ==="
  "$PY" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_t_daps4k_worker.py' -q
  local RC=$?
  echo "DAPS4K_WORKER_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: DAPS4K_WORKER_REGRESSIONS"
    return 8
  fi
  echo "=== NATIVE SOURCE, Y, SEED, CHECKPOINT PREFLIGHT ==="
  for INDEX in 0 9 17; do
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPYCACHEPREFIX="$RUN/pycache/worker_$(printf '%02d' "$INDEX")" \
      "$DAPS_PY" "$WT/scripts/track_r/xp_t_daps4k_worker_gpu.py" \
      --expected-head "$EXPECTED" --index "$INDEX" --gpu 0 \
      --preflight-only
    RC=$?
    echo "DAPS4K_NATIVE_PREFLIGHT_INDEX_$INDEX RC=$RC"
    if [ "$RC" -ne 0 ]; then
      echo "ACTION NEEDED: DAPS4K_NATIVE_PREFLIGHT"
      return 8
    fi
  done
  echo "=== FOUR PHYSICAL GPU / ONE WORKER PER DEVICE ==="
  nohup "$PY" -u "$WT/scripts/track_r/xp_t_daps4k_supervisor.py" \
    --expected-head "$EXPECTED" \
    --required-vram-mib 32000 \
    --poll-seconds 10 \
    --gpu-hour-cap 6.0 \
    > "$RUN/XP_T_DAPS4K_SUPERVISOR.log" 2>&1 < /dev/null &
  local PID=$!
  echo "DAPS4K_SUPERVISOR_PID=$PID"
  echo "DAPS4K_STATUS=$RUN/XP_T_DAPS4K_STATUS.json"
  echo "DAPS4K_LOG=$RUN/XP_T_DAPS4K_SUPERVISOR.log"
  echo "DAPS4K_STOP_AFTER_ACTIVE=$RUN/STOP_AFTER_ACTIVE"
  echo "DAPS4K_GPU_HOUR_CAP=6.0"
  echo "DAPS4K_WORKERS_PER_DEVICE=1"
  echo "DAPS4K_SMOKE_GATE=FIRST_FFHQ_AND_IMAGENET"
  echo "DAPS4K_SUBMITTED"
  return 0
}
go
RC=$?
echo "DAPS4K_LAUNCH_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
