#!/usr/bin/env bash
# Launch guarded 4-GPU XP-T SITCOM 18-root native W=1 calibration.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_t_sitcom_20261026T233451Z"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
SITCOM_PY="$ROOT/conda-envs/sitcom_ode_bw/bin/python"
HEAD="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$HEAD" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$HEAD" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XPT_SITCOM_WRONG_OR_DIRTY_HEAD"
    return 8
  fi
  if [ ! -f "$RUN/XP_T_SITCOM_PREP.json" ] ||
     [ ! -f "$RUN/XP_T_SITCOM_TASKS.json" ] ||
     [ -e "$RUN/XP_T_SITCOM_STATUS.json" ] ||
     [ -e "$RUN/gpu" ] ||
     [ -e "$RUN/XP_T_SITCOM_SUPERVISOR.log" ]; then
    echo "ACTION NEEDED: XPT_SITCOM_MISSING_PREP_OR_ALREADY_STARTED"
    return 8
  fi
  echo "=== CPU REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" \
     -p 'test_track_r_xp_*.py' -q
  RC=$?
  echo "XPT_SITCOM_CPU_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XPT_SITCOM_TESTS"
    return 8
  fi
  echo "=== THREE NATIVE GT-FREE PREFLIGHTS (NO GPU INFERENCE) ==="
  mkdir -p "$ROOT/outputs/.cache/track_r/tmp"
  for INDEX in 0 9 17; do
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPYCACHEPREFIX="$RUN/pycache/worker_$(printf '%02d' "$INDEX")" \
       "$SITCOM_PY" "$WT/scripts/track_r/xp_t_sitcom_worker_gpu.py" \
          --expected-head "$HEAD" --index "$INDEX" --gpu 0 \
          --preflight-only
    RC=$?
    echo "XPT_SITCOM_PREFLIGHT_INDEX_$INDEX RC=$RC"
    if [ "$RC" -ne 0 ]; then
      echo "ACTION NEEDED: XPT_SITCOM_PREFLIGHT"
      return 8
    fi
  done
  echo "=== FOUR-PHYSICAL-GPU W1 SUPERVISOR ==="
  nohup "$PY" -u "$WT/scripts/track_r/xp_t_sitcom_supervisor.py" \
    --expected-head "$HEAD" \
    --required-vram-mib 24000 \
    --poll-seconds 10 --gpu-hour-cap 2.0 \
    > "$RUN/XP_T_SITCOM_SUPERVISOR.log" 2>&1 < /dev/null &
  PID=$!
  echo "XPT_SITCOM_SUPERVISOR_PID=$PID"
  echo "XPT_SITCOM_STATUS=$RUN/XP_T_SITCOM_STATUS.json"
  echo "XPT_SITCOM_LOG=$RUN/XP_T_SITCOM_SUPERVISOR.log"
  echo "XPT_SITCOM_STOP=$RUN/STOP_AFTER_ACTIVE"
  echo "XPT_SITCOM_MAX_GPU_H=2.0"
  echo "XPT_SITCOM_WORKERS_PER_GPU=1"
  echo "XPT_SITCOM_SUBMITTED"
}
go
RC=$?
echo "XPT_SITCOM_LAUNCH_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
