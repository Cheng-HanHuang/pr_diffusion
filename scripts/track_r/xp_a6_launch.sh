#!/usr/bin/env bash
# CPU-only freeze + detached unattended 4-GPU A6 critical production.
# Preflight, run each stage's full distinct smoke, then auto-main if PASS.
# No B16: its B22 sequential post-model RNG lineage is not yet certified.
# D1_P2 and EPRIME remain sealed; no GT, metrics or protected cohort.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
if [ -z "$HEAD" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$HEAD" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "ACTION NEEDED: A6_WRONG_OR_DIRTY_CHECKOUT"
  exit 8
fi
echo "=== A6 CPU SYNTHETIC SCIENCE/SCHEDULER TESTS ==="
"$PY" -m unittest discover -s "$WT/tests" \
  -p 'test_track_r_xp_a6_*.py' -q
echo "=== A6 FREEZE SIGNED Y + NATIVE SETTINGS ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a6_$STAMP"
"$PY" "$WT/scripts/track_r/xp_a6_prepare_cpu.py" \
   --expected-head "$HEAD" --out "$RUN"
test -f "$RUN/XP_A6_PREP.json"
test ! -e "$RUN/gpu"
test ! -e "$RUN/XP_A6_CONTROLLER_STATE.json"
test ! -e "$RUN/XP_A6_AUTOPILOT.log"
echo "=== AUTO SMOKE -> MAIN (11-HOUR WALL / 40 GPU-H BUDGET) ==="
nohup bash -c '
  set -uo pipefail
  ROOT=/egr/research-pac/huang248
  WT="$ROOT/pr_diffusion_track_r"
  PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="$WT"
  RUN="$1";HEAD="$2"
  echo "=== STAGE-SPECIFIC SMOKES: 24 D1P2 + 8 D3 + 8 EPRIME ==="
  "$PY" -u "$WT/scripts/track_r/xp_a6_supervisor.py" \
    --run "$RUN" --expected-head "$HEAD" \
    --mode smoke --hours 11 --gpu-hours 40 --poll 10
  RC=$?
  echo "XP_A6_SMOKE_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_A6_STOP_AFTER_SMOKE_FAILURE"
    exit "$RC"
  fi
  echo "=== MAIN: sealed D1 P2 + SITCOM ImageNet D3 + DAPS4K FFHQ EPRIME ==="
  "$PY" -u "$WT/scripts/track_r/xp_a6_supervisor.py" \
    --run "$RUN" --expected-head "$HEAD" \
    --mode main --hours 11 --gpu-hours 40 --poll 10
  RC=$?
  echo "XP_A6_MAIN_RC=$RC"
  echo "XP_A6_STILL_SEALED=D1P2,EPRIME,A16,A64"
  exit "$RC"
' bash "$RUN" "$HEAD" > "$RUN/XP_A6_AUTOPILOT.log" 2>&1 < /dev/null &
PID=$!
echo "XP_A6_AUTOPILOT_PID=$PID"
echo "XP_A6_RUN_ROOT=$RUN"
echo "XP_A6_LOG=$RUN/XP_A6_AUTOPILOT.log"
echo "XP_A6_STATE=$RUN/XP_A6_CONTROLLER_STATE.json"
echo "XP_A6_STOP_FILE=$RUN/STOP_AFTER_ACTIVE"
echo "XP_A6_STAGES=D1_P2,D3,EPRIME"
echo "XP_A6_SMOKE_FIRST=true"
echo "XP_A6_11H_WALL_CAP=true"
echo "XP_A6_TOTAL_GPU_HOUR_CAP=40"
echo "XP_A6_TIME_AWARE=GPU_TELEMETRY_ONLY"
echo "XP_A6_TIME_UNAWARE=ALL_CANDIDATE_GENERATION"
echo "XP_A6_OUTCOMES_SCORING=false"
echo "XP_A6_B16_BLOCKED_BY_SEQUENTIAL_HISTORY=true"
echo "DO_NOT_CHANGE_GIT_HEAD_WHILE_AUTOPILOT_RUNNING=true"
echo "XP_A6_AUTOPILOT_SUBMITTED"
