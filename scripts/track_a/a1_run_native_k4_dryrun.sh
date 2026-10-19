#!/usr/bin/env bash
# A1.0d CPU-only synthetic K4 / source check. No CUDA/real data/inference.
# Never use set -e/-u/pipefail and never exits an interactive shell.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
TEST_LOG="$BASE/A1_0D_SYNTHETIC_TESTS_V1.log"
RUN_LOG="$BASE/A1_0D_NATIVE_K4_DRYRUN_V1.log"
JSON="$BASE/A1_0D_NATIVE_K4_DRYRUN_V1.json"
ZIP="$BASE/A1_0D_NATIVE_K4_DRYRUN_V1_RETURN.zip"
STATUS=0
if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_STOP: pinned A1_EXPECTED_HEAD required"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_STOP: dedicated A1 source or env missing"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_STOP: unexpected A1 branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_STOP: source commit mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_STOP: tracked source modification"
  STATUS=2
elif [ -e "$TEST_LOG" ] || [ -e "$RUN_LOG" ] || [ -e "$JSON" ] || [ -e "$ZIP" ]; then
  echo "A1_STOP: stage0d output already exists; preserve it"
  STATUS=2
else
  mkdir -p "$BASE"
  MK_RC=$?
  if [ "$MK_RC" -ne 0 ]; then
    echo "A1_STOP: cannot create dedicated A1 output parent"
    STATUS=2
  else
    echo "A1_0D_HEAD=$(git -C "$REPO" rev-parse HEAD)"
    echo "A1_0D_STAGE=synthetic_unit_tests"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" \
      "$PY" -B -m unittest \
        scripts.track_a.test_a1_protocol \
        scripts.track_a.test_a1_stage0_preflight \
        scripts.track_a.test_a1_native_k4_dryrun -v > "$TEST_LOG" 2>&1
    TEST_RC=$?
    echo "A1_0D_TEST_RC=$TEST_RC"
    if [ "$TEST_RC" -ne 0 ]; then
      tail -55 "$TEST_LOG"
      STATUS=3
    else
      echo "A1_0D_STAGE=cpu_native_source_seed_dryrun"
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" \
        "$PY" -B "$REPO/scripts/track_a/a1_native_k4_dryrun.py" \
          --output "$JSON" --expected-head "$A1_EXPECTED_HEAD" \
          > "$RUN_LOG" 2>&1
      DRY_RC=$?
      echo "A1_0D_DRYRUN_RC=$DRY_RC"
      tail -30 "$RUN_LOG"
      if [ "$DRY_RC" -ne 0 ]; then
        STATUS=4
      fi
    fi
  fi
fi
# Preserves logs on STOP; ZIP contains no GT or image data.
if [ -f "$TEST_LOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$BASE" "$ZIP" <<'PY'
import hashlib,sys
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile
base,arch=map(Path,sys.argv[1:])
items=[base/x for x in ("A1_0D_SYNTHETIC_TESTS_V1.log","A1_0D_NATIVE_K4_DRYRUN_V1.log","A1_0D_NATIVE_K4_DRYRUN_V1.json")]
items=[p for p in items if p.is_file()]
with ZipFile(arch,"x",ZIP_DEFLATED) as f:
    for p in items:
        f.write(p,p.name)
print("A1_0D_RETURN_ZIP",arch)
print("A1_0D_RETURN_SHA256",hashlib.sha256(arch.read_bytes()).hexdigest())
PY
  ZIP_RC=$?
  echo "A1_0D_PACK_RC=$ZIP_RC"
  if [ "$ZIP_RC" -ne 0 ]; then STATUS=5; fi
fi
echo "A1_0D_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
