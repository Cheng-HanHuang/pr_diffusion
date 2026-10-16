#!/usr/bin/env bash
# A0 Track A — bounded exposed B21/B22 metadata-only analysis. No set -e,
# no GPU, image/model/FFT/measurement reads, no confirmation305 access.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a0_native_selector"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015"
B21="$ROOT/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401"
B22="$ROOT/outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135"
B24="$ROOT/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z"
COR="$ROOT/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z"
OUT="$BASE/A0_B22_METADATA_REPLAY_V1"
TLOG="$BASE/A0_B22_SYNTHETIC_TESTS.log"
RLOG="$BASE/A0_B22_METADATA_REPLAY.log"
ARCHIVE="$BASE/A0_B22_METADATA_REPLAY_RETURN.zip"

if [ ! -x "$PY" ] || [ ! -d "$REPO/.git" ] || [ ! -d "$BASE" ]; then
  echo "A0_STOP: interpreter, A0 checkout, or existing inventory base missing"
elif [ -e "$OUT" ] || [ -e "$TLOG" ] || [ -e "$RLOG" ] || [ -e "$ARCHIVE" ]; then
  echo "A0_STOP: A0 B22 output/log/archive already exists; preserve and inspect"
elif [ ! -f "$BASE/A0_SITCOM_PATCH_AUDIT.json" ]; then
  echo "A0_STOP: bounded B22 SITCOM patch evidence missing"
else
  echo "A0_CHECKOUT_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  echo "A0_CHECKOUT_BRANCH=$(git -C "$REPO" branch --show-current)"
  echo "A0_B22_STAGE=synthetic_unittest"
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m unittest \
    scripts.track_a.test_a0_selectors -v > "$TLOG" 2>&1
  TRC=$?
  echo "A0_SYNTHETIC_TEST_RC=$TRC"
  if [ "$TRC" -ne 0 ]; then
    echo "A0_STOP: synthetic tests failed; no historical analysis launched"
    tail -80 "$TLOG"
  else
    echo "A0_B22_STAGE=metadata_only_replay"
    PYTHONDONTWRITEBYTECODE=1 "$PY" -B \
      "$REPO/scripts/track_a/a0_historical_compare.py" \
      --cohort B22 \
      --spec "$REPO/configs/track_a/a0_analysis_spec.json" \
      --inventory "$BASE/A0_SOURCE_AND_ARTIFACT_MANIFEST.json" \
      --source-reconcile "$BASE/A0_SITCOM_SOURCE_RECHECK.json" \
      --patch-audit "$BASE/A0_SITCOM_PATCH_AUDIT.json" \
      --b21 "$B21" --b22 "$B22" --b24 "$B24" --corrected "$COR" \
      --out "$OUT" > "$RLOG" 2>&1
    ARC=$?
    echo "A0_B22_REPLAY_RC=$ARC"
    tail -50 "$RLOG"
    if [ "$ARC" -eq 0 ] && [ -f "$OUT/A0_REPLAY_CHECKS.json" ]; then
      echo "A0_B22_STAGE=pack_results"
      "$PY" -B - "$BASE" "$ARCHIVE" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
import sys
base=Path(sys.argv[1]); archive=Path(sys.argv[2])
files = [base/"A0_B22_SYNTHETIC_TESTS.log",
         base/"A0_B22_METADATA_REPLAY.log",
         base/"A0_SITCOM_PATCH_AUDIT.json",
         base/"A0_SITCOM_SOURCE_RECHECK.json",
         base/"A0_SOURCE_AND_ARTIFACT_MANIFEST.json"]
files += sorted((base/"A0_B22_METADATA_REPLAY_V1").glob("*"))
if not files or not all(p.is_file() for p in files):
    raise RuntimeError("missing A0 result artifact")
with ZipFile(archive, mode="x", compression=ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(base))
print("A0_RETURN_ZIP",archive)
print("A0_RETURN_ZIP_SHA256",sha256(archive.read_bytes()).hexdigest())
PY
      PACKRC=$?
      echo "A0_PACK_RC=$PACKRC"
    fi
  fi
fi
