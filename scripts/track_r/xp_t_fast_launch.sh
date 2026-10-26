#!/usr/bin/env bash
# Launch bounded XP-T FAST FFHQ calibration (9 new independent GT-free roots).
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
CPU="$ROOT/conda-envs/prdiff_track_r/bin/python"
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
    echo "ACTION NEEDED: XPT_FAST_GIT_PREFLIGHT"
    return 8
  fi
  echo "=== SYNTHETIC CPU TESTS ==="
  CUDA_VISIBLE_DEVICES="" "$CPU" -m unittest discover -s "$WT/tests" \
     -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_T_TESTS"
    return 8
  fi
  echo "=== PREPARE GT-FREE 9-ROOT MANIFEST ==="
  OUT="$BASE/xp_t_fast_$TAG"
  CUDA_VISIBLE_DEVICES="" "$CPU" "$WT/scripts/track_r/xp_t_fast_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$OUT"
  PREP_RC=$?
  echo "XPT_PREP_RC=$PREP_RC"
  if [ "$PREP_RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_T_PREP"
    return 8
  fi
  echo "=== LAUNCH ONE-GPU CALIBRATION SUPERVISOR ==="
  nohup "$CPU" "$WT/scripts/track_r/xp_t_fast_supervisor.py" \
    --expected-head "$EXPECTED" --out "$OUT" \
    --estimated-vram-mib 32000 --poll-seconds 10 \
    > "$OUT/XP_T_FAST_SUPERVISOR.log" 2>&1 < /dev/null &
  PID=$!
  if [ "$PID" -le 0 ]; then
    echo "ACTION NEEDED: XP_T_NO_SUPERVISOR_PID"
    return 8
  fi
  echo "XP_T_SUPERVISOR_PID=$PID"
  echo "XP_T_OUTPUT=$OUT"
  echo "XP_T_STATUS=$OUT/XP_T_FAST_STATUS.json"
  echo "XP_T_LOG=$OUT/XP_T_FAST_SUPERVISOR.log"
  echo "XP_T_STOP_SENTINEL=$OUT/STOP_AFTER_ACTIVE"
  echo "OK"
  return 0
}
main
RC=$?
echo "XP_T_LAUNCH_RC=$RC"
"$CPU" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
