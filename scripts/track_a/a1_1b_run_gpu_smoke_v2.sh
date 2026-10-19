#!/usr/bin/env bash
# A1.1b V2 procedural four-method GPU smoke, ONLY after confirmed renewed RUN RESERVATION.
# Executes one full candidate/method, not K4; no FFHQ/confirmation data.
# No set -e/-u/pipefail or exit: preserve PAC terminal on failure.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_1b_one_each_v2"
LOG="$BASE/A1_1B_GPU_NATIVE_ONEEACH_V2.log"
ZIP="$BASE/A1_1B_GPU_NATIVE_RETURN_V2.zip"
STATUS=0
GPU="$A1_PHYSICAL_GPU"
ACK="$A1_RUN_RESERVATION_ACK"
if [ -z "$A1_EXPECTED_HEAD" ] || [ "$ACK" != "TRACK_A_A1_1B_PROCEDURAL_GPU_SMOKE_APPROVED" ]; then
  echo "A1_1B_STOP: source pin and explicit coordinator RUN RESERVATION acknowledgement needed"
  STATUS=2
elif [ "$GPU" != "3" ]; then
  echo "A1_1B_STOP: proposed reservation only covers physical GPU 3"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_1B_STOP: checkout or interpreter missing"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "codex/prdiff-track-a-a1-oracle-budget-20261016" ]; then
  echo "A1_1B_STOP: wrong branch"
  STATUS=2
elif [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_1B_STOP: source commit mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_1B_STOP: tracked source dirty"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$LOG" ] || [ -e "$ZIP" ]; then
  echo "A1_1B_STOP: existing smoke output; preserve for debugging, do not rerun"
  STATUS=2
else
  echo "A1_1B_V2_GPU_HEAD=$(git -C "$REPO" rev-parse HEAD)"
  echo "A1_1B_V2_STAGE=FOUR_SEQUENTIAL_NATIVE_SINGLE_CANDIDATES"
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -B "$REPO/scripts/track_a/a1_1b_native_one_each.py" \
      --mode execute --expected-head "$A1_EXPECTED_HEAD" \
      --physical-gpu "$GPU" \
      --reservation-ack "$ACK" > "$LOG" 2>&1
  RUN_RC=$?
  echo "A1_1B_V2_NATIVE_RC=$RUN_RC"
  tail -60 "$LOG"
  if [ "$RUN_RC" -ne 0 ]; then STATUS=3; fi
fi
# Return only bounded metadata, run logs and timing JSON, never images or tensor payloads.
if [ -f "$LOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$BASE" "$OUT" "$LOG" "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib,sys
base,out,log,archive=map(Path,sys.argv[1:])
items=[log]
if out.is_dir():
    for pattern in ("RUN_IDENTITY.json","A1_1B_COMPLETE.json","A1_1B_STOP.json",
                    "DAPS_COMPLETE.json","SITCOM_COMPLETE.json",
                    "NP_LF_COMPLETE.json","NP_S2_COMPLETE.json"):
        path=out/pattern
        if path.is_file():items.append(path)
    for p in sorted(out.rglob("cuda_timing.json")):
        if p.is_file():items.append(p)
    for p in sorted(out.rglob("run.log")):
        if p.is_file():items.append(p)
    for p in sorted(out.rglob("result.json")):
        if p.is_file():items.append(p)
with ZipFile(archive,"x",compression=ZIP_DEFLATED) as z:
    for p in items:
        if p.stat().st_size>5_000_000:
            raise RuntimeError(f"refuse to package unexpected oversized source {p}")
        z.write(p,p.relative_to(base))
print("A1_1B_V2_GPU_RETURN_ZIP",archive)
print("A1_1B_V2_GPU_RETURN_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  PACK_RC=$?
  echo "A1_1B_V2_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=4; fi
fi
echo "A1_1B_V2_GPU_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
