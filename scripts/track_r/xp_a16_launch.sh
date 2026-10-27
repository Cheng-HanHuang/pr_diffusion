#!/usr/bin/env bash
# XP-A16 guarded four-GPU FAST candidate generation, GT-free, no scoring.
# Requires immutable 2,160-root XP-A16 manifest at known PAC path.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_a16_20261027T022827Z"
CPU="$ROOT/conda-envs/prdiff_track_r/bin/python"
FAST="$ROOT/conda-envs/fast_dips_a15_blackwell/bin/python"
EXPECTED="$1"
MANIFEST_SHA=796429527d01963668a67c5e39949803cb8cc5ecfc35d01dcb20c5ca1e978432
PREP_HEAD=f67bd915dab3ce39be96264ca6a1ec002ad623ff
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XP_A16_WRONG_OR_DIRTY_CHECKOUT"
    return 8
  fi
  if [ ! -f "$RUN/XP_A16_PREP.json" ] ||
     [ ! -f "$RUN/XP_A16_TASKS.json" ] ||
     [ -e "$RUN/gpu" ] || [ -e "$RUN/XP_A16_STATUS.json" ] ||
     [ -e "$RUN/XP_A16_SUPERVISOR.log" ] ||
     [ -e "$RUN/XP_A16_SUPERVISOR_PID.json" ]; then
    echo "ACTION NEEDED: XP_A16_MISSING_FROZEN_MANIFEST_OR_ALREADY_STARTED"
    return 8
  fi
  echo "=== XP-A16 CPU SAFETY AND SCHEDULING TESTS ==="
  "$CPU" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_a16_*.py' -q
  local RC=$?
  echo "XP_A16_GPU_GATE_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_A16_CPU_TEST_FAILURE"
    return 8
  fi
  echo "=== NATIVE FAST GT-FREE SOURCE/CHECKPOINT/Y PREFLIGHT ==="
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPYCACHEPREFIX="$RUN/preflight_cache" \
    "$FAST" "$WT/scripts/track_r/xp_fast_gpu.py" \
      --expected-head "$EXPECTED" \
      --roots "$RUN/XP_A16_TASKS.json" \
      --roots-sha "$MANIFEST_SHA" \
      --roots-source-head "$PREP_HEAD" \
      --out "$RUN/gpu/chunk_0000_12" \
      --start 0 --count 12 --physical-gpu 0 \
      --dataset ffhq --oversample 2.0 --preflight-only
  RC=$?
  echo "XP_A16_NATIVE_PREFLIGHT_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_A16_NATIVE_PREFLIGHT"
    return 8
  fi
  echo "=== START 4-GPU SUPERVISOR (SMOKE 24 THEN MAIN 2136) ==="
  nohup "$CPU" -u "$WT/scripts/track_r/xp_a16_supervisor.py" \
     --expected-head "$EXPECTED" \
     --poll-seconds 10 --vram-mib 32000 --gpu-hour-cap 12.0 \
     > "$RUN/XP_A16_SUPERVISOR.log" 2>&1 < /dev/null &
  echo "XP_A16_SUPERVISOR_PID=$!"
  echo "XP_A16_RUN_ROOT=$RUN"
  echo "XP_A16_STATUS=$RUN/XP_A16_STATUS.json"
  echo "XP_A16_LOG=$RUN/XP_A16_SUPERVISOR.log"
  echo "XP_A16_STOP_AFTER_ACTIVE=$RUN/STOP_AFTER_ACTIVE"
  echo "XP_A16_GT_EVALUATION_ALLOWED=false"
  echo "XP_A16_OUTCOME_EMBARGO=ACTIVE"
  echo "XP_A16_SUBMITTED"
}
go
RC=$?
echo "XP_A16_LAUNCH_RC=$RC"
"$CPU" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
