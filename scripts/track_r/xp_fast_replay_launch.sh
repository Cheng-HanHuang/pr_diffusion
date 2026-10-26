#!/usr/bin/env bash
# CPU prepare and asynchronous native FAST-DIPS replay gate; 8 frozen R0 roots.
# No GT in worker or manifest. No shell flags changing caller state.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
CPU="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-xp-20261026
EXPECTED="$1"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE"
mkdir -p "$BASE" "$CACHE/tmp"

main() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XP_FAST_REPLAY_UNCLEAN_HEAD"
    return 8
  fi
  echo "=== UNIT TESTS ==="
  CUDA_VISIBLE_DEVICES="" "$CPU" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: FAST_REPLAY_CPU_TEST"
    return 8
  fi
  OUT="$BASE/xp_fast_replay_$TAG"
  echo "=== FROZEN MANIFEST PREPARATION (CPU, no GT) ==="
  CUDA_VISIBLE_DEVICES="" "$CPU" "$WT/scripts/track_r/xp_fast_replay_cpu.py" \
    prepare --expected-head "$EXPECTED" --out "$OUT"
  PREP_RC=$?
  echo "REPLAY_PREP_RC=$PREP_RC"
  if [ "$PREP_RC" -ne 0 ]; then
    echo "ACTION NEEDED: FAST_REPLAY_PREP"
    return 8
  fi
  echo "=== GUARDED ASYNCHRONOUS REPLAY (one GPU, 8 roots) ==="
  nohup "$CPU" "$WT/scripts/track_r/xp_fast_replay_supervisor.py" \
    --expected-head "$EXPECTED" --out "$OUT" \
    --estimated-vram-mib 32000 --poll-seconds 10 \
    > "$OUT/XP_FAST_SUPERVISOR.log" 2>&1 < /dev/null &
  PID=$!
  if [ "$PID" -le 0 ]; then
    echo "ACTION NEEDED: FAST_REPLAY_LAUNCH_PID"
    return 8
  fi
  echo "FAST_REPLAY_SUPERVISOR_PID=$PID"
  echo "FAST_REPLAY_OUTPUT=$OUT"
  echo "FAST_REPLAY_READONLY_STATUS=$OUT/XP_FAST_REPLAY_STATUS.json"
  echo "FAST_REPLAY_LOG=$OUT/XP_FAST_SUPERVISOR.log"
  echo "FAST_REPLAY_STOP_SENTINEL=$OUT/STOP_AFTER_ACTIVE"
  echo "OK"
}

main
LAUNCH_RC=$?
echo "FAST_REPLAY_LAUNCH_RC=$LAUNCH_RC"
"$CPU" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
