#!/usr/bin/env bash
# Launch exactly eight historical GT-free SITCOM replay roots with GPU admission.
# No GT path ever supplied to the worker. CPU scoring follows only on success.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
OUT="$BASE/xp_sitcom_replay_20261026T223925Z"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
BRANCH=codex/prdiff-track-r-xp-20261026
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT"
export XDG_CACHE_HOME="$ROOT/outputs/.cache/track_r"
export TMPDIR="$ROOT/outputs/.cache/track_r/tmp"

launch() {
  if [ -z "$HEAD" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$HEAD" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: SITCOM_LAUNCH_WRONG_DIRTY_CHECKOUT"
    return 8
  fi
  if [ ! -f "$OUT/XP_SITCOM_REPLAY_PREP.json" ] ||
     [ ! -f "$OUT/XP_SITCOM_REPLAY_TASKS.json" ] ||
     [ -e "$OUT/XP_SITCOM_REPLAY_STATUS.json" ]; then
    echo "ACTION NEEDED: SITCOM_LAUNCH_MISSING_PREP_OR_ALREADY_RUN"
    return 8
  fi
  echo "=== CPU REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  RC=$?
  echo "SITCOM_LAUNCH_CPU_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: SITCOM_LAUNCH_CPU_TEST_FAILED"
    return 8
  fi
  echo "=== FIVE NATIVE SOURCE / Y / RNG-IDENTITY PREFLIGHTS (NO SAMPLING) ==="
  SITCOM_PY="$ROOT/conda-envs/sitcom_ode_bw/bin/python"
  for IDX in 0 4 5 6 7; do
    MODE=b24
    OUTPART="$OUT/gpu/b24_$IDX"
    if [ "$IDX" -eq 0 ]; then
      MODE=b22
      OUTPART="$OUT/gpu/b22"
    fi
    CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX="$OUT/pycache" \
      "$SITCOM_PY" "$WT/scripts/track_r/xp_sitcom_replay_gpu.py" \
      --expected-head "$HEAD" --mode "$MODE" --index "$IDX" \
      --out "$OUTPART" --gpu 0 --preflight-only
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "ACTION NEEDED: SITCOM_NATIVE_PREFLIGHT_$IDX"
      return 8
    fi
  done
  echo "=== START GUARDED NATIVE SITCOM REPLAY SUPERVISOR ==="
  mkdir -p "$ROOT/outputs/.cache/track_r/tmp"
  nohup "$PY" -u "$WT/scripts/track_r/xp_sitcom_replay_supervisor.py" \
     --expected-head "$HEAD" --poll-seconds 10 \
     --estimated-vram-mib 32000 --gpu-hour-cap 2.0 \
     > "$OUT/XP_SITCOM_REPLAY_SUPERVISOR.log" 2>&1 < /dev/null &
  PID=$!
  echo "XP_SITCOM_SUPERVISOR_PID=$PID"
  echo "XP_SITCOM_STATUS=$OUT/XP_SITCOM_REPLAY_STATUS.json"
  echo "XP_SITCOM_LOG=$OUT/XP_SITCOM_REPLAY_SUPERVISOR.log"
  echo "XP_SITCOM_STOP_SENTINEL=$OUT/STOP_AFTER_ACTIVE"
  echo "XP_SITCOM_BUDGET_GPU_H_CAP=2.0"
  echo "SITCOM_LAUNCH_SUBMITTED"
  return 0
}
launch
RC=$?
echo "SITCOM_LAUNCH_SCRIPT_RC=$RC"
echo "=== XP STATUS ==="
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
