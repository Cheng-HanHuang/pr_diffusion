#!/usr/bin/env bash
# One PAC block starts XP-T FAST ImageNet batch1 after freezing all 100 y on CPU.
# No shell strict modes or terminal exit; do not change git after nohup launch.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE"
mkdir -p "$BASE" "$CACHE/tmp"

run_once() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XP-T IMAGENET DIRTY/MISMATCHED CHECKOUT"
    return 8
  fi
  echo "=== CPU REGRESSION TESTS ==="
  CUDA_VISIBLE_DEVICES="" "$PY" -m unittest discover \
    -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP-T IMAGENET REGRESSION TESTS"
    return 8
  fi
  echo "=== CPU COMMON IMAGENET100 MEASUREMENTS ==="
  RUN="$BASE/xp_t_fast_imagenet_$TAG"
  CUDA_VISIBLE_DEVICES="" "$PY" "$WT/scripts/track_r/xp_t_imagenet_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$RUN"
  PREP_RC=$?
  echo "IMAGENET_MEASUREMENT_PREP_RC=$PREP_RC"
  if [ "$PREP_RC" -ne 0 ]; then
    echo "ACTION NEEDED: IMAGENET PHYSICS/MEASUREMENT PREPARATION"
    return 8
  fi
  echo "=== LAUNCH NINE GT-FREE FAST IMAGENET ROOTS ==="
  nohup "$PY" "$WT/scripts/track_r/xp_t_fast_supervisor.py" \
    --expected-head "$EXPECTED" --out "$RUN" \
    --poll-seconds 10 --estimated-vram-mib 32000 \
    > "$RUN/XP_T_FAST_SUPERVISOR.log" 2>&1 < /dev/null &
  PID=$!
  if [ "$PID" -le 0 ]; then
    echo "ACTION NEEDED: XP-T SUPERVISOR PID"
    return 8
  fi
  echo "XP_T_SUPERVISOR_PID=$PID"
  echo "XP_T_OUTPUT=$RUN"
  echo "XP_T_STATUS=$RUN/XP_T_FAST_STATUS.json"
  echo "XP_T_SUPERVISOR_LOG=$RUN/XP_T_FAST_SUPERVISOR.log"
  echo "XP_T_STOP_SENTINEL=$RUN/STOP_AFTER_ACTIVE"
  echo "OK"
  return 0
}
run_once
RC=$?
echo "XP_T_IMAGENET_LAUNCH_RC=$RC"
echo "=== XP STATUS DIGEST ==="
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
