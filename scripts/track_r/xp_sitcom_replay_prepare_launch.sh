#!/usr/bin/env bash
# Track R XP-SITCOM historical replay manifest: CPU only, no GT file reads.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CPU="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: SITCOM_REPLAY_CHECKOUT"
    return 8
  fi
  echo "=== XP CPU REGRESSION TESTS ==="
  "$CPU" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_*.py' -q
  RC=$?
  echo "SITCOM_CPU_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: SITCOM_CPU_REGRESSION"
    return 8
  fi
  echo "=== LOCK HISTORICAL B22/DEV80 SITCOM REPLAY ==="
  RUN="$BASE/xp_sitcom_replay_$(date -u +%Y%m%dT%H%M%SZ)"
  "$CPU" "$WT/scripts/track_r/xp_sitcom_replay_prepare_cpu.py" \
    --expected-head "$EXPECTED" --out "$RUN"
  RC=$?
  echo "SITCOM_REPLAY_PREP_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: SITCOM_REPLAY_PREP"
    return 8
  fi
  echo "XP_SITCOM_REPLAY_ROOT=$RUN"
  echo "XP_SITCOM_REPLAY_MANIFEST=$RUN/XP_SITCOM_REPLAY_TASKS.json"
  echo "XP_SITCOM_REPLAY_OFFLINE=$RUN/XP_SITCOM_REPLAY_OFFLINE_HISTORICAL.json"
  echo "OK"
}
go
RC=$?
echo "SITCOM_PREP_SCRIPT_RC=$RC"
exit "$RC"
