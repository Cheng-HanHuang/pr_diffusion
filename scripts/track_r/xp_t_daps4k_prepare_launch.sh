#!/usr/bin/env bash
# CPU-only DAPS-4K source/checkpoint/shared-y manifest freeze. No GPU.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
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
    echo "ACTION NEEDED: DAPS4K_PREP_WRONG_OR_DIRTY_GIT"
    return 8
  fi
  echo "=== XP CPU REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  local RC=$?
  echo "DAPS4K_CPU_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: DAPS4K_TESTS"
    return 8
  fi
  local OUT="$BASE/xp_t_daps4k_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "=== VERIFY PINNED DAPS-4K AND COMMON SIGNED MEASUREMENTS ==="
  "$PY" "$WT/scripts/track_r/xp_t_daps4k_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$OUT"
  RC=$?
  echo "DAPS4K_PREP_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: DAPS4K_PREP_INTEGRITY"
    return 8
  fi
  echo "DAPS4K_RUN_ROOT=$OUT"
  echo "DAPS4K_GPU_LAUNCHED=false"
  return 0
}
go
RC=$?
echo "DAPS4K_PREP_LAUNCH_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
