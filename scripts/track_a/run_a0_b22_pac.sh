#!/usr/bin/env bash
# Track A A0 — B22-only CPU metadata audit, V2; preserves V1 failure evidence.
# No shell exit/set -e/set -u/pipefail; caller terminal remains open on failure.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a0_native_selector"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a0_native_selector_audit_20261015"
B21="$ROOT/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401"
B22="$ROOT/outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135"
B24="$ROOT/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z"
COR="$ROOT/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z"
OUT="$BASE/A0_B22_METADATA_REPLAY_V2"
TLOG="$BASE/A0_B22_SYNTHETIC_TESTS_V2.log"
RLOG="$BASE/A0_B22_METADATA_REPLAY_V2.log"
ARCHIVE="$BASE/A0_B22_METADATA_REPLAY_V2_RETURN.zip"
STATUS=0

if [ ! -x "$PY" ] || [ ! -d "$REPO/.git" ] || [ ! -d "$BASE" ]; then
  echo "A0_STOP: interpreter, A0 checkout, or existing inventory base missing"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$TLOG" ] || [ -e "$RLOG" ] || [ -e "$ARCHIVE" ]; then
  echo "A0_STOP: V2 output/log/archive exists; preserve and inspect"
  STATUS=2
elif [ ! -f "$BASE/A0_SITCOM_PATCH_AUDIT.json" ]; then
  echo "A0_STOP: bounded B22 SITCOM patch evidence missing"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a0-native-selector-audit" ]; then
  echo "A0_STOP: unexpected checkout branch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A0_STOP: A0 source checkout has modified tracked files"
  STATUS=2
else
  echo "A0_CHECKOUT_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  echo "A0_CHECKOUT_BRANCH=$(git -C "$REPO" branch --show-current)"
  echo "A0_B22_STAGE=synthetic_unittest"
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m unittest \
    scripts.track_a.test_a0_selectors -v > "$TLOG" 2>&1
  TRC=$?
  echo "A0_SYNTHETIC_TEST_RC=$TRC"
  if [ "$TRC" -ne 0 ]; then
    echo "A0_STOP: synthetic tests failed; historical analysis was NOT launched"
    tail -85 "$TLOG"
    STATUS=3
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
    tail -40 "$RLOG"
    if [ "$ARC" -ne 0 ]; then
      echo "A0_STOP: metadata provenance/schema/replay failed, no result is a PASS"
      STATUS=4
    elif [ ! -f "$OUT/A0_REPLAY_CHECKS.json" ]; then
      echo "A0_STOP: replay checks absent"
      STATUS=5
    fi
  fi
fi

# Pack available safe A0 logs/metadata, including failure log on STOP.
# Never packages reconstruction, source repositories, or confirmation data.
if [ -f "$TLOG" ] && [ ! -e "$ARCHIVE" ]; then
  "$PY" -B - "$BASE" "$OUT" "$TLOG" "$RLOG" "$ARCHIVE" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
import sys
base,out,tlog,rlog,archive=map(Path,sys.argv[1:])
files=[f for f in (tlog,rlog,base/"A0_SITCOM_PATCH_AUDIT.json",
                   base/"A0_SITCOM_SOURCE_RECHECK.json",
                   base/"A0_SOURCE_AND_ARTIFACT_MANIFEST.json") if f.is_file()]
if out.is_dir():
    files += [p for p in sorted(out.iterdir()) if p.is_file()
              and p.suffix in (".csv",".json",".md")]
if not files: raise RuntimeError("no A0 metadata/log output")
with ZipFile(archive,mode="x",compression=ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(base))
print("A0_B22_V2_RETURN_ZIP", archive)
print("A0_B22_V2_RETURN_SHA256", sha256(archive.read_bytes()).hexdigest())
PY
  PACKRC=$?
  echo "A0_PACK_RC=$PACKRC"
  if [ "$PACKRC" -ne 0 ]; then
    STATUS=6
  fi
fi
echo "A0_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
