#!/usr/bin/env bash
# Explicit pre-GPU recovery of failed serialization launch. No cleanup/overwrite.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_t_sitcom_20261026T233451Z"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XPT_SITCOM_RECOVERY_GIT"
    return 8
  fi
  echo "=== CPU REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_*.py' -q
  local RC=$?
  echo "XPT_SITCOM_RECOVERY_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XPT_SITCOM_RECOVERY_CPU_TESTS"
    return 8
  fi
  echo "=== STRICT PRE-GPU ABORT AUDIT ==="
  "$PY" "$WT/scripts/track_r/xp_t_sitcom_recovery_cpu.py" \
    --expected-head "$EXPECTED"
  RC=$?
  echo "XPT_SITCOM_RECOVERY_AUDIT_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XPT_SITCOM_RECOVERY_UNSAFE"
    return 8
  fi
  if [ -e "$RUN/XP_T_SITCOM_RECOVERY_SUPERVISOR.log" ]; then
    echo "ACTION NEEDED: XPT_SITCOM_RECOVERY_ALREADY_SUBMITTED"
    return 8
  fi
  echo "=== RESUME FROZEN 18 ROOTS / FOUR GPUs W1 ==="
  nohup "$PY" -u "$WT/scripts/track_r/xp_t_sitcom_supervisor.py" \
    --expected-head "$EXPECTED" --recover \
    --required-vram-mib 24000 --poll-seconds 10 --gpu-hour-cap 2.0 \
    > "$RUN/XP_T_SITCOM_RECOVERY_SUPERVISOR.log" 2>&1 < /dev/null &
  local PID=$!
  echo "XP_T_SITCOM_RECOVERY_SUPERVISOR_PID=$PID"
  echo "XP_T_SITCOM_RECOVERY_LOG=$RUN/XP_T_SITCOM_RECOVERY_SUPERVISOR.log"
  echo "XP_T_SITCOM_STATUS=$RUN/XP_T_SITCOM_STATUS.json"
  echo "XP_T_SITCOM_RECOVERY_SUBMITTED"
  return 0
}
go
RC=$?
echo "XPT_SITCOM_RECOVERY_LAUNCH_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
