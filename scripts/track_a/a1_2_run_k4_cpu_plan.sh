#!/usr/bin/env bash
# A1.2: CPU preflight for 8 exposed B22 images x 16 independent native trajectories.
# NO GPU execution; A1.2 --mode execute is disabled until an explicit later lease.
# No set -e/-u/pipefail or exit. Preserve terminal on errors.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
TESTLOG="$BASE/A1_2_K4_CPU_TESTS_V1.log"
PLANLOG="$BASE/A1_2_K4_CPU_PLAN_V1.json"
ZIP="$BASE/A1_2_K4_CPU_PLAN_RETURN_V1.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ] || [ ! -x "$PY" ] || [ ! -d "$REPO/.git" ]; then
  echo "A1_2_K4_STOP: source or Python unavailable"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ] || \
     [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_2_K4_STOP: source branch or commit mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_2_K4_STOP: modified tracked checkout"
  STATUS=2
elif [ -e "$TESTLOG" ] || [ -e "$PLANLOG" ] || [ -e "$ZIP" ]; then
  echo "A1_2_K4_STOP: prior planning receipt exists; preserve it"
  STATUS=2
else
  mkdir -p "$BASE"
  echo "A1_2_K4_SOURCE_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B -m unittest \
    scripts.track_a.test_a1_protocol \
    scripts.track_a.test_a1_2_k4_exposed_b22 -v > "$TESTLOG" 2>&1
  TEST_RC=$?
  echo "A1_2_K4_TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    STATUS=3
    tail -50 "$TESTLOG"
  else
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B "$REPO/scripts/track_a/a1_2_k4_exposed_b22.py" \
        --mode plan --expected-head "$A1_EXPECTED_HEAD" > "$PLANLOG" 2>&1
    PLAN_RC=$?
    echo "A1_2_K4_PLAN_RC=$PLAN_RC"
    if [ "$PLAN_RC" -ne 0 ]; then
      STATUS=4
      tail -50 "$PLANLOG"
    else
      "$PY" -B - "$PLANLOG" <<'PY'
import json,sys
from pathlib import Path
doc=json.loads(Path(sys.argv[1]).read_text())
print("A1_2_K4_PLAN_STATUS",doc["status"])
print("A1_2_K4_PLAN_IMAGES",len(doc["selected_images"]))
print("A1_2_K4_PLAN_CANDIDATES",doc["candidate_count"])
print("A1_2_K4_PLAN_GPU_RESERVED",doc["gpu_run_reserved"])
PY
    fi
  fi
fi
if [ -f "$TESTLOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$TESTLOG" "$PLANLOG" "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,sys
test,plan,zpath=map(Path,sys.argv[1:])
with ZipFile(zpath,"x",compression=ZIP_DEFLATED) as z:
    for p in (test,plan):
        if p.is_file():z.write(p,p.name)
print("A1_2_K4_CPU_PLAN_ZIP",zpath)
print("A1_2_K4_CPU_PLAN_SHA256",hashlib.sha256(zpath.read_bytes()).hexdigest())
PY
  PACK_RC=$?
  echo "A1_2_K4_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=5; fi
fi
echo "A1_2_K4_CPU_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
