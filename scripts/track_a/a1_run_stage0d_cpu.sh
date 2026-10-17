#!/usr/bin/env bash
# A1.0d CPU-only synthetic restart-plan checks. Never invokes an inference entrypoint.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
TEST_LOG="$BASE/A1_0D_RESTART_SYNTHETIC_TESTS.log"
SUMMARY="$BASE/A1_0D_RESTART_SYNTHETIC_SUMMARY.json"
ZIP="$BASE/A1_0D_RESTART_SYNTHETIC_RETURN.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_0D_STOP: missing pinned expected HEAD"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_0D_STOP: dedicated checkout/interpreter missing"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_0D_STOP: unexpected branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_0D_STOP: pinned source HEAD mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_0D_STOP: tracked files modified"
  STATUS=2
elif [ -e "$TEST_LOG" ] || [ -e "$SUMMARY" ] || [ -e "$ZIP" ]; then
  echo "A1_0D_STOP: existing synthetic report would be overwritten"
  STATUS=2
else
  mkdir -p "$BASE"
  MKRC=$?
  if [ "$MKRC" -ne 0 ]; then
    STATUS=2
  else
    echo "A1_0D_HEAD=$(git -C "$REPO" rev-parse HEAD)"
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1       "$PY" -B -m unittest       scripts.track_a.test_a1_protocol       scripts.track_a.test_a1_stage0_preflight       scripts.track_a.test_a1_restart_plan -v > "$TEST_LOG" 2>&1
    TRC=$?
    echo "A1_0D_TEST_RC=$TRC"
    if [ "$TRC" -ne 0 ]; then
      STATUS=3
      tail -80 "$TEST_LOG"
    else
      PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1         "$PY" -B -m scripts.track_a.a1_restart_plan > "$SUMMARY" 2>&1
      SRC=$?
      echo "A1_0D_SUMMARY_RC=$SRC"
      if [ "$SRC" -ne 0 ]; then
        STATUS=4
      else
        cat "$SUMMARY"
      fi
    fi
  fi
fi

if [ -f "$TEST_LOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$TEST_LOG" "$SUMMARY" "$ZIP" <<'PY'
import hashlib,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
test,summary,archive=map(Path,sys.argv[1:])
with ZipFile(archive,"x",compression=ZIP_DEFLATED) as z:
    for p in (test,summary):
        if p.is_file():
            z.write(p,p.name)
print("A1_0D_RETURN_ZIP",archive)
print("A1_0D_RETURN_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PRC=$?
  echo "A1_0D_PACK_RC=$PRC"
  if [ "$PRC" -ne 0 ]; then
    STATUS=5
  fi
fi
echo "A1_0D_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
