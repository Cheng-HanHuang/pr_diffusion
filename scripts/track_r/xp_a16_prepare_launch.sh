#!/usr/bin/env bash
# CPU-only XP-A16 freeze; no GPU, no GT and no new outcome evaluation.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT"
go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XP_A16_GIT_PREFLIGHT"
    return 8
  fi
  echo "=== XP-A16 SYNTHETIC REGRESSIONS ==="
  "$PY" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_a16_prepare.py' -q
  local RC=$?
  echo "XP_A16_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_A16_TESTS"
    return 8
  fi
  local RUN="$BASE/xp_a16_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "=== FREEZE HISTORICAL-EXCHANGEABLE FAST RUNS 4–15 ==="
  "$PY" "$WT/scripts/track_r/xp_a16_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$RUN"
  RC=$?
  echo "XP_A16_PREP_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP_A16_PREP"
    return 8
  fi
  echo "XP_A16_TASKS_PATH=$RUN/XP_A16_TASKS.json"
  echo "XP_A16_GPU_STARTED=false"
  echo "XP_A16_OUTCOME_EMBARGO=ACTIVE"
  return 0
}
go
RC=$?
echo "XP_A16_PREP_SCRIPT_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
