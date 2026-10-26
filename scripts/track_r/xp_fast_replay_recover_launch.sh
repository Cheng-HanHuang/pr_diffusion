#!/usr/bin/env bash
# CPU-only repair of finished FAST replay scorer. Never launch CUDA.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_fast_replay_20261026T203415Z"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-xp-20261026
EXPECTED="$1"
SOURCE_HEAD=314180b314674a24d5ddc40a0186b24b6082a5e0
TAG="$(date -u +%Y%m%dT%H%M%SZ)"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export TMPDIR="$ROOT/outputs/.cache/track_r/tmp"
export XDG_CACHE_HOME="$ROOT/outputs/.cache/track_r"
mkdir -p "$TMPDIR"

main() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: RECOVERY_UNCLEAN_HEAD"
    return 8
  fi

  echo "=== CPU REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_*.py' -q
  RC=$?
  echo "TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: RECOVERY_TEST_FAILED"
    return 8
  fi

  echo "=== CPU-ONLY ORIGINAL ROOTS RE-SCORE ==="
  "$PY" "$WT/scripts/track_r/xp_fast_replay_recover_cpu.py" \
    --expected-head "$EXPECTED" --source-head "$SOURCE_HEAD"
  RC=$?
  echo "REPLAY_RECOVER_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "ACTION NEEDED: REPLAY_GATE_NOT_ACCEPTED"
    return 8
  fi

  echo "=== SAFE PUBLICATION OF REPLAY RECEIPTS ==="
  TARGET="$WT/docs/track_r/executors/xp/returns/XP_FAST_REPLAY_RECOVER_$TAG"
  mkdir -p "$TARGET"
  if cp "$RUN/XP_FAST_REPLAY_RECOVERY_STATUS.json" "$TARGET/" &&
     cp "$RUN/XP_FAST_REPLAY_OFFLINE_RECEIPT.json" "$TARGET/" &&
     cp "$RUN/XP_FAST_REPLAY_TIMING.json" "$TARGET/" &&
     cp "$RUN/XP_FAST_REPLAY_STATUS.json" "$TARGET/" &&
     cp "$RUN/gpu/R0_5_CHUNK_COMPLETE.json" "$TARGET/" &&
     (
       cd "$TARGET" &&
       sha256sum XP_FAST_REPLAY_RECOVERY_STATUS.json \
         XP_FAST_REPLAY_OFFLINE_RECEIPT.json \
         XP_FAST_REPLAY_TIMING.json XP_FAST_REPLAY_STATUS.json \
         R0_5_CHUNK_COMPLETE.json > SHA256SUMS.txt
     ) &&
     git -C "$WT" add "docs/track_r/executors/xp/returns/XP_FAST_REPLAY_RECOVER_$TAG" &&
     git -C "$WT" commit -m "data(track-r): publish native FAST replay CPU recovery" &&
     git -C "$WT" push origin "$BRANCH"; then
    echo "FAST_REPLAY_RECOVERY_PUBLISHED_COMMIT=$(git -C "$WT" rev-parse HEAD)"
    return 0
  fi
  echo "ACTION NEEDED: RECOVERY_PUBLISH_FAILURE"
  return 8
}

main
RC=$?
echo "RECOVERY_SCRIPT_RC=$RC"
echo "=== XP STATUS DIGEST ==="
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
