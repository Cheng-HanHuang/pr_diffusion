#!/usr/bin/env bash
# A1.1b planning and source/probe preflight ONLY. Never invokes native GPU processes.
# Keep terminal alive on any failure; no set -e, set -u, pipefail, or exit.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
TESTLOG="$BASE/A1_1B_CPU_TESTS_V1.log"
PLAN="$BASE/A1_1B_CPU_SOURCE_PLAN_V1.json"
ZIP="$BASE/A1_1B_CPU_PLAN_RETURN_V1.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_1B_STOP: expected source head not supplied"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_1B_STOP: dedicated checkout/interpreter unavailable"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_1B_STOP: unexpected source branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_1B_STOP: source HEAD differs from pinned commit"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_1B_STOP: modified tracked source"
  STATUS=2
elif [ -e "$TESTLOG" ] || [ -e "$PLAN" ] || [ -e "$ZIP" ]; then
  echo "A1_1B_STOP: existing plan output; preserve prior receipt"
  STATUS=2
else
  mkdir -p "$BASE"
  echo "A1_1B_CPU_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B -m unittest \
    scripts.track_a.test_a1_protocol \
    scripts.track_a.test_a1_restart_plan \
    scripts.track_a.test_a1_1b_native_one_each -v > "$TESTLOG" 2>&1
  TRC=$?
  echo "A1_1B_CPU_TEST_RC=$TRC"
  if [ "$TRC" -ne 0 ]; then
    STATUS=3
    tail -60 "$TESTLOG"
  else
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B "$REPO/scripts/track_a/a1_1b_native_one_each.py" \
        --mode plan --expected-head "$A1_EXPECTED_HEAD" > "$PLAN" 2>&1
    PRC=$?
    echo "A1_1B_CPU_PLAN_RC=$PRC"
    if [ "$PRC" -ne 0 ]; then
      STATUS=4
      tail -35 "$PLAN"
    else
      tail -4 "$PLAN"
    fi
  fi
fi
if [ -f "$TESTLOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$TESTLOG" "$PLAN" "$ZIP" <<'PY'
import hashlib,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
test,plan,archive=map(Path,sys.argv[1:])
with ZipFile(archive,"x",compression=ZIP_DEFLATED) as z:
    for p in (test,plan):
        if p.is_file():z.write(p,p.name)
print("A1_1B_CPU_ZIP",archive)
print("A1_1B_CPU_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PACK_RC=$?
  echo "A1_1B_CPU_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=5; fi
fi
echo "A1_1B_CPU_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
