#!/usr/bin/env bash
# A1.0 V2 read-only source/metadata inventory (preserves V1 STOP). No set -e/-u/pipefail, no exit.
# All outputs are restricted to dedicated A1-owned namespace.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_0_preflight_20261017_v2"
TEST_LOG="$BASE/A1_0_SYNTHETIC_TESTS_V2.log"
PRE_LOG="$BASE/A1_0_PREFLIGHT_V2.log"
RETURN_ZIP="$BASE/A1_0_PREFLIGHT_RETURN_V2.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_STOP: A1_EXPECTED_HEAD must be set by the pinned caller"
  STATUS=2
elif [ ! -x "$PY" ] || [ ! -d "$REPO/.git" ]; then
  echo "A1_STOP: dedicated checkout or interpreter unavailable"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_STOP: unexpected checkout branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_STOP: checkout HEAD differs from pinned expected SHA"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_STOP: modified tracked checkout source"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$TEST_LOG" ] || [ -e "$PRE_LOG" ] || [ -e "$RETURN_ZIP" ]; then
  echo "A1_STOP: preflight artifacts already exist; preserve them"
  STATUS=2
else
  mkdir -p "$BASE"
  MKRC=$?
  if [ "$MKRC" -ne 0 ]; then
    echo "A1_STOP: cannot create dedicated A1 base"
    STATUS=2
  else
    echo "A1_HEAD=$(git -C "$REPO" rev-parse HEAD)"
    echo "A1_STAGE=synthetic_tests"
    PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m unittest \
      scripts.track_a.test_a1_protocol \
      scripts.track_a.test_a1_stage0_preflight -v > "$TEST_LOG" 2>&1
    TEST_RC=$?
    echo "A1_SYNTHETIC_TEST_RC=$TEST_RC"
    if [ "$TEST_RC" -ne 0 ]; then
      echo "A1_STOP: synthetic code tests failed; no PAC metadata/source inventory started"
      tail -80 "$TEST_LOG"
      STATUS=3
    else
      echo "A1_STAGE=read_only_source_and_exposed_metadata_preflight"
      PYTHONDONTWRITEBYTECODE=1 "$PY" -B \
        "$REPO/scripts/track_a/a1_stage0_preflight.py" \
        --checkout "$REPO" --expected-head "$A1_EXPECTED_HEAD" \
        --output "$OUT" > "$PRE_LOG" 2>&1
      PRE_RC=$?
      echo "A1_PREFLIGHT_RC=$PRE_RC"
      tail -30 "$PRE_LOG"
      if [ "$PRE_RC" -ne 0 ]; then
        echo "A1_STOP: source/checkpoint/metadata gate is not passed; NO GPU permission"
        STATUS=4
      elif [ ! -f "$OUT/A1_0_PREFLIGHT.json" ]; then
        echo "A1_STOP: source report missing"
        STATUS=5
      fi
    fi
  fi
fi
# The ZIP contains only new A1 metadata/logs and is produced even on STOP.
if [ -f "$TEST_LOG" ] && [ ! -e "$RETURN_ZIP" ]; then
  "$PY" -B - "$BASE" "$OUT" "$RETURN_ZIP" <<'PY'
import hashlib,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
base,out,dst=map(Path,sys.argv[1:])
files=[base/"A1_0_SYNTHETIC_TESTS_V2.log",base/"A1_0_PREFLIGHT_V2.log"]
files=[x for x in files if x.is_file()]
if (out/"A1_0_PREFLIGHT.json").is_file():
    files.append(out/"A1_0_PREFLIGHT.json")
with ZipFile(dst,"x",compression=ZIP_DEFLATED) as z:
    for path in files:
        z.write(path,path.relative_to(base))
print("A1_V2_RETURN_ZIP",dst)
print("A1_V2_RETURN_SHA256",hashlib.sha256(dst.read_bytes()).hexdigest())
PY
  ZRC=$?
  echo "A1_PACK_RC=$ZRC"
  if [ "$ZRC" -ne 0 ]; then
    STATUS=6
  fi
fi
echo "A1_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
