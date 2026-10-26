#!/usr/bin/env bash
# CPU-only posthoc timing and shared-ImageNet-y publication; no CUDA, no GPU rerun.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
IN="$BASE/xp_t_fast_imagenet_20261026T212512Z"
BRANCH=codex/prdiff-track-r-xp-20261026
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
EXPECTED="$1"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT"

go() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: REAUDIT_WRONG_DIRTY_CHECKOUT"
    return 8
  fi
  echo "=== CPU TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: REAUDIT_TESTS"
    return 8
  fi
  echo "=== INDEPENDENT CPU TIMING EVIDENCE AUDIT ==="
  AUDIT="$BASE/xp_t_timing_audit_$TAG"
  "$PY" "$WT/scripts/track_r/xp_t_timing_reaudit_cpu.py" \
    --expected-head "$EXPECTED" --out "$AUDIT"
  RC=$?
  echo "REAUDIT_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: ORIGINAL_UNVERIFIED_TIMING_RETAINS_STATUS"
    return 8
  fi
  echo "=== PIN SHARED IMAGENET Y WITHOUT GT ==="
  Y="$IN/XP_IMAGENET100_Y_MANIFEST.json"
  EXPECTED_Y_SHA=26619959e3a17aeb2c0ad413f87b1764ecd8bc7aabfc0415c862cbaf27873b78
  ACTUAL_Y_SHA="$(sha256sum "$Y" 2>/dev/null | cut -d' ' -f1)"
  if [ "$ACTUAL_Y_SHA" != "$EXPECTED_Y_SHA" ]; then
    echo "ACTION NEEDED: COMMON_IMAGENET_Y_SHA_DRIFT"
    return 8
  fi
  TARGET="$WT/docs/track_r/executors/xp/returns/XP_T_FAST_TIMING_AUDIT_$TAG"
  mkdir -p "$TARGET"
  if cp "$AUDIT/XP_T_FAST_TIMING_REAUDIT.json" "$TARGET/" &&
     cp "$Y" "$TARGET/" &&
     (
       cd "$TARGET" &&
       sha256sum XP_T_FAST_TIMING_REAUDIT.json XP_IMAGENET100_Y_MANIFEST.json \
         > SHA256SUMS.txt
     ) &&
     git -C "$WT" add "docs/track_r/executors/xp/returns/XP_T_FAST_TIMING_AUDIT_$TAG" &&
     git -C "$WT" commit -m "data(track-r): audit XP-T timing and pin 100 shared ImageNet y identifiers" &&
     git -C "$WT" push origin "$BRANCH"; then
    echo "XP_T_AUDIT_PUBLISHED_COMMIT=$(git -C "$WT" rev-parse HEAD)"
    return 0
  fi
  echo "ACTION NEEDED: REAUDIT_PUBLISH_FAILED"
  return 8
}

go
RC=$?
echo "AUDIT_SCRIPT_RC=$RC"
echo "=== XP DIGEST ==="
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
