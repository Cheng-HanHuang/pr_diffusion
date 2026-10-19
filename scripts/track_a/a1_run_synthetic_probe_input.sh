#!/usr/bin/env bash
# A1.1a procedural probe and frozen native DAPS observation, CPU-only.
# No real FFHQ/locked B24 input access; no diffusion/model checkpoint.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/daps/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_1a_synthetic_probe_v1"
TESTLOG="$BASE/A1_1A_SYNTHETIC_INPUT_TESTS_V1.log"
RUNLOG="$BASE/A1_1A_NATIVE_MEASUREMENT_V1.log"
ZIP="$BASE/A1_1A_PROBE_METADATA_RETURN_V1.zip"
STATUS=0

if [ -z "$A1_EXPECTED_HEAD" ]; then
  echo "A1_1A_STOP: expected source head not supplied"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_1A_STOP: dedicated checkout or daps Python not found"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_1A_STOP: wrong branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_1A_STOP: pinned checkout mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_1A_STOP: modified tracked checkout"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$TESTLOG" ] || [ -e "$RUNLOG" ] || [ -e "$ZIP" ]; then
  echo "A1_1A_STOP: old A1.1a artifacts exist, preserve them"
  STATUS=2
else
  mkdir -p "$BASE"
  MKRC=$?
  if [ "$MKRC" -ne 0 ]; then
    STATUS=3
  else
    echo "A1_1A_HEAD=$(git -C "$REPO" rev-parse HEAD)"
    echo "A1_1A_STAGE=CPU_SYNTHETIC_INPUT_TEST"
    PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
      "$PY" -B -m unittest scripts.track_a.test_a1_synthetic_probe_input -v > "$TESTLOG" 2>&1
    TRC=$?
    echo "A1_1A_TEST_RC=$TRC"
    if [ "$TRC" -ne 0 ]; then
      STATUS=4
      tail -75 "$TESTLOG"
    else
      echo "A1_1A_STAGE=NATIVE_PHASE_RETRIEVAL_MEASUREMENT_CPU"
      PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
        "$PY" -B "$REPO/scripts/track_a/a1_synthetic_probe_input.py" \
        --output "$OUT" --expected-head "$A1_EXPECTED_HEAD" > "$RUNLOG" 2>&1
      RRC=$?
      echo "A1_1A_MEASUREMENT_RC=$RRC"
      tail -50 "$RUNLOG"
      if [ "$RRC" -ne 0 ] || [ ! -f "$OUT/A1_1A_PROBE_MANIFEST.json" ]; then
        STATUS=5
      fi
    fi
  fi
fi

# A1.1a ZIP only includes JSON metadata and small logs, no image/GT/measurement bytes.
if [ -f "$TESTLOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$TESTLOG" "$RUNLOG" "$OUT/A1_1A_PROBE_MANIFEST.json" "$ZIP" <<'PY'
import hashlib,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
test,log,manifest,archive=map(Path,sys.argv[1:])
with ZipFile(archive,"x",compression=ZIP_DEFLATED) as z:
    for p in (test,log,manifest):
        if p.is_file():
            z.write(p,p.name)
print("A1_1A_RETURN_ZIP",archive)
print("A1_1A_RETURN_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PRC=$?
  echo "A1_1A_PACK_RC=$PRC"
  if [ "$PRC" -ne 0 ]; then STATUS=6; fi
fi
echo "A1_1A_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
