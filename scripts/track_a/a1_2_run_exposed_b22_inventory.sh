#!/usr/bin/env bash
# CPU-only inventory of 8 explicitly EXPOSED B22 FFHQ images, not prospective evaluation.
# No GT/measurement tensor reads, no inference, no B24 protected confirmation.
# No set -e, set -u, pipefail, or exit; terminal remains open on STOP.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_2_exposed_b22_dev8_inventory_v1"
TEST="$BASE/A1_2_EXPOSED_B22_CPU_TESTS_V1.log"
LOG="$BASE/A1_2_EXPOSED_B22_INVENTORY_V1.log"
ZIP="$BASE/A1_2_EXPOSED_B22_INVENTORY_RETURN_V1.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ] || [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_2_STOP: missing source pin, checkout or Python"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_2_STOP: unexpected checkout branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_2_STOP: source commit mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_2_STOP: tracked source dirty"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$TEST" ] || [ -e "$LOG" ] || [ -e "$ZIP" ]; then
  echo "A1_2_STOP: output already exists, preserve first receipt"
  STATUS=2
else
  mkdir -p "$BASE"
  echo "A1_2_INVENTORY_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -B -m unittest scripts.track_a.test_a1_2_exposed_b22_inventory -v \
      > "$TEST" 2>&1
  TEST_RC=$?
  echo "A1_2_INVENTORY_TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    STATUS=3
    tail -55 "$TEST"
  else
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B "$REPO/scripts/track_a/a1_2_exposed_b22_inventory.py" \
        --expected-head "$A1_EXPECTED_HEAD" > "$LOG" 2>&1
    INVENTORY_RC=$?
    echo "A1_2_INVENTORY_RC=$INVENTORY_RC"
    tail -40 "$LOG"
    if [ "$INVENTORY_RC" -ne 0 ]; then STATUS=4; fi
  fi
fi

if [ -f "$TEST" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$TEST" "$LOG" "$OUT/A1_2_EXPOSED_DEV8_INVENTORY.json" "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,sys
test,log,manifest,archive=map(Path,sys.argv[1:])
with ZipFile(archive,"x",ZIP_DEFLATED) as z:
    for p in (test,log,manifest):
        if p.is_file():z.write(p,p.name)
print("A1_2_INVENTORY_RETURN_ZIP",archive)
print("A1_2_INVENTORY_RETURN_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PACK_RC=$?
  echo "A1_2_INVENTORY_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=5; fi
fi
echo "A1_2_INVENTORY_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
