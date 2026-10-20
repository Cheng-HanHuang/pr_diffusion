#!/usr/bin/env bash
# A1.2: one-time CPU byte verification of 8 EXPOSED historical B22 images/obs.
# This script DOES NOT run solvers or generate new measurements. No protected B24.
# No set -e/-u/pipefail and no shell exit.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_2_exposed_b22_inputs_v1"
TESTLOG="$BASE/A1_2_INPUT_FREEZE_CPU_TESTS_V1.log"
RUNLOG="$BASE/A1_2_INPUT_FREEZE_RUN_V1.log"
ZIP="$BASE/A1_2_INPUT_FREEZE_METADATA_RETURN_V1.zip"
STATUS=0
if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_2_STOP: required pinned git head missing"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_2_STOP: checkout or Python unavailable"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ] || \
     [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_2_STOP: wrong git source"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_2_STOP: modified tracked A1 files"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$TESTLOG" ] || [ -e "$RUNLOG" ] || [ -e "$ZIP" ]; then
  echo "A1_2_STOP: previous input-freeze output present, preserve it"
  STATUS=2
else
  mkdir -p "$BASE"
  echo "A1_2_INPUT_FREEZE_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B -m unittest scripts.track_a.test_a1_2_freeze_exposed_inputs -v \
      > "$TESTLOG" 2>&1
  TEST_RC=$?
  echo "A1_2_INPUT_FREEZE_TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    STATUS=3
    tail -55 "$TESTLOG"
  else
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B "$REPO/scripts/track_a/a1_2_freeze_exposed_inputs.py" \
        --expected-head "$A1_EXPECTED_HEAD" > "$RUNLOG" 2>&1
    FREEZE_RC=$?
    echo "A1_2_INPUT_FREEZE_RC=$FREEZE_RC"
    tail -40 "$RUNLOG"
    if [ "$FREEZE_RC" -ne 0 ]; then STATUS=4; fi
  fi
fi
# Archive tiny metadata logs only. Do NOT transfer image or measurement bytes.
if [ -f "$TESTLOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$BASE" "$OUT" "$TESTLOG" "$RUNLOG" "$ZIP" <<'PY'
import hashlib,sys
from zipfile import ZipFile,ZIP_DEFLATED
from pathlib import Path
base,out,test,log,archive=map(Path,sys.argv[1:])
items=[test]
if log.is_file():items.append(log)
if out.is_dir():
    for p in sorted(out.rglob("item.json")):items.append(p)
    p=out/"A1_2_INPUT_FREEZE_COMPLETE.json"
    if p.is_file():items.append(p)
with ZipFile(archive,"x",compression=ZIP_DEFLATED) as z:
    for p in items:
        if p.stat().st_size>100000:raise RuntimeError(f"unexpected oversized metadata {p}")
        z.write(p,p.relative_to(base))
print("A1_2_INPUT_FREEZE_ZIP",archive)
print("A1_2_INPUT_FREEZE_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PACK_RC=$?
  echo "A1_2_INPUT_FREEZE_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=5; fi
fi
echo "A1_2_INPUT_FREEZE_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
