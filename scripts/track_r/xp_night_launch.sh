#!/usr/bin/env bash
# One CPU prep, one detached 4-GPU overnight supervisor:
# validate 3 distinct stage smokes first; main only if every smoke passes.
# Full has 11h controller wall cap, 46 GPU-h aggregate cap; no GT or scoring.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

if [ -z "$EXPECTED" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "ACTION NEEDED: XP_NIGHT_WRONG_OR_DIRTY_HEAD"
  exit 8
fi
echo "=== CPU-ONLY NIGHTLY SCIENCE / SAFETY TESTS ==="
"$PY" -m unittest discover -s "$WT/tests" \
 -p 'test_track_r_xp_night_*.py' -q
echo "=== FREEZE 100 IMAGENET / 180 FFHQ SIGNED-Y TASKS ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_overnight_$STAMP"
"$PY" "$WT/scripts/track_r/xp_night_prepare_cpu.py" \
 --expected-head "$EXPECTED" --out "$RUN"
test -f "$RUN/XP_NIGHT_PREP.json"
test ! -e "$RUN/XP_NIGHT_CONTROLLER_STATE.json"
test ! -e "$RUN/XP_NIGHT_ACTIVE_OWNER.json"
test ! -e "$RUN/gpu"
test ! -e "$RUN/XP_NIGHT_AUTOPILOT.log"
echo "=== DETACH GUARDED SMOKE -> 11H MAIN (ONLY ON ALL SMOKES PASS) ==="
nohup bash -c '
  set -uo pipefail
  ROOT=/egr/research-pac/huang248
  WT="$ROOT/pr_diffusion_track_r"
  PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$WT"
  RUN="$1"
  HEAD="$2"
  echo "=== XP NIGHT SMOKE: D1_P1 + A64 + D2 ==="
  "$PY" -u "$WT/scripts/track_r/xp_night_supervisor.py" \
    --root "$RUN" --expected-head "$HEAD" \
    --mode smoke --hours 11.0 --gpu-hours 46 --poll 10
  RC=$?
  echo "XP_NIGHT_SMOKE_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: STOP_ALL_AUTOPILOT_AFTER_SMOKE_FAILURE"
    exit "$RC"
  fi
  echo "=== XP NIGHT MAIN: D1_P1 -> A64 -> D2; D1_P2 SEALED/BLOCKED ==="
  "$PY" -u "$WT/scripts/track_r/xp_night_supervisor.py" \
    --root "$RUN" --expected-head "$HEAD" \
    --mode main --hours 11.0 --gpu-hours 46 --poll 10
  RC=$?
  echo "XP_NIGHT_MAIN_RC=$RC"
  echo "XP_NIGHT_OUTCOME_EMBARGO=ACTIVE"
  exit "$RC"
' bash "$RUN" "$EXPECTED" > "$RUN/XP_NIGHT_AUTOPILOT.log" 2>&1 < /dev/null &
PID=$!
echo "XP_NIGHT_AUTOPILOT_PID=$PID"
echo "XP_NIGHT_RUN_ROOT=$RUN"
echo "XP_NIGHT_LOG=$RUN/XP_NIGHT_AUTOPILOT.log"
echo "XP_NIGHT_STATUS=$RUN/XP_NIGHT_CONTROLLER_STATE.json"
echo "XP_NIGHT_STOP_FILE=$RUN/STOP_AFTER_ACTIVE"
echo "XP_NIGHT_ALL_SMOKES_REQUIRED=true"
echo "XP_NIGHT_MAIN_WALL_CAP_H=11"
echo "XP_NIGHT_COMBINED_GPUH_CAP=46"
echo "XP_NIGHT_READY_CRITICAL=D1_P1,A64,D2"
echo "XP_NIGHT_DEFERRED=D1_P2,C,C2,F8"
echo "XP_NIGHT_DATA_PROTECTION=NO_GT_NO_PSNR_NO_PROTECTED"
echo "DO_NOT_CHANGE_GIT_HEAD_WHILE_AUTOPILOT_RUNNING=true"
echo "XP_NIGHT_SUBMITTED"
