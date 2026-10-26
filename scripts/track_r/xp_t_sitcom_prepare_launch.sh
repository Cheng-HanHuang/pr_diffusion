#!/usr/bin/env bash
# CPU-only historical/common-y gate for SITCOM XP-T 3x3 FFHQ + ImageNet.
# The 18 native GPU timing roots are NOT launched in this stage.
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
    echo "ACTION NEEDED: XPT_SITCOM_DIRTY_OR_WRONG_CHECKOUT"
    return 8
  fi
  echo "=== CPU XP REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  local RC=$?
  echo "XPT_SITCOM_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XPT_SITCOM_CPU_TEST"
    return 8
  fi
  local RUN="$BASE/xp_t_sitcom_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "=== FREEZE 18 GT-FREE SITCOM TIMING INPUTS ==="
  "$PY" "$WT/scripts/track_r/xp_t_sitcom_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$RUN"
  RC=$?
  echo "XPT_SITCOM_PREP_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: XPT_SITCOM_INPUT_PREP_FAILED"
    return 8
  fi
  echo "XP_T_SITCOM_RUN_ROOT=$RUN"
  echo "XP_T_SITCOM_TASKS=$RUN/XP_T_SITCOM_TASKS.json"
  echo "XP_T_SITCOM_GPU_LAUNCHED=false"
  echo "OK"
}
go
RC=$?
echo "XPT_SITCOM_PREP_SCRIPT_RC=$RC"
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
exit "$RC"
