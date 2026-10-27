#!/usr/bin/env bash
# A7 isolated scheduler-only E' resume:
# original A6 runner/worktree/manifests/seeds/environments EXACTLY pinned.
# CPU audit 1645 native receipts then detached resume. No GT scoring.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
A7WT="$ROOT/pr_diffusion_track_r_a7_cpu"
RUN="$ROOT/outputs/pr_diffusion/track_r/xp_a6_20261027T165112Z"
GENERATION_HEAD=b7af2f7135beb6e8367d83f028e0fbaf163bc7bd
A7_HEAD="$1"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$A7WT"

if [ -z "$A7_HEAD" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$GENERATION_HEAD" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ] ||
   [ "$(git -C "$A7WT" rev-parse HEAD)" != "$A7_HEAD" ] ||
   [ -n "$(git -C "$A7WT" status --porcelain)" ] ||
   [ -e "$RUN/STOP_AFTER_ACTIVE" ]; then
    echo "ACTION NEEDED: PINNED_INPUT_OR_A7_CHECKOUT_OR_STOP"
    exit 8
fi

echo "=== SYNTHETIC POLICY TESTS (CPU ONLY) ==="
"$PY" -m unittest discover -s "$A7WT/tests" \
  -p 'test_track_r_xp_a7_resume_policy.py' -q
echo "=== REHASH AND REVALIDATE ALL 1645 COMPLETED NATIVE ROOTS ==="
"$PY" -u "$A7WT/scripts/track_r/xp_a7_resume_preflight_cpu.py" \
  --a7-head "$A7_HEAD"

echo "=== GPU SNAPSHOT; NEVER TERMINATE FOREIGN PROCESSES ==="
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RUN/XP_A7_RESUME_$STAMP.log"
test ! -e "$LOG"
echo "=== A7 DETACHED SAME-RUN 32H WALL-CAPPED RESUME ==="
nohup "$PY" -u \
    "$A7WT/scripts/track_r/xp_a7_resume_supervisor.py" \
    --run "$RUN" \
    --expected-head "$GENERATION_HEAD" \
    --mode main \
    --hours 32 \
    --gpu-hours 40 \
    --poll 15 \
    >"$LOG" 2>&1 </dev/null &
PID=$!
echo "A7_RESUME_SUBMITTED_PID=$PID"
echo "A7_RESUME_LOG=$LOG"
echo "A7_RESUME_STATE=$RUN/XP_A6_CONTROLLER_STATE.json"
echo "A7_RESUME_JOURNAL=$RUN/XP_A6_RESUME_JOURNAL.jsonl"
echo "A7_STOP_FILE=$RUN/STOP_AFTER_ACTIVE"
echo "A7_WORKER_HEAD=$GENERATION_HEAD"
echo "A7_CONTROLLER_HEAD=$A7_HEAD"
echo "A7_CO_TENANCY_CAP=2_GPU_IF_FOREIGN"
echo "A7_CO_TENANCY_WAIT_NO_30MIN_EXIT=true"
echo "A7_MAX_WALL_HOURS=32"
echo "A7_GPU_HOUR_CAP_TOTAL=40"
echo "A7_NO_GT_NO_OUTCOME_SCORE=true"
echo "A7_SAFE_RESUME_SUBMITTED"
