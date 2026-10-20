#!/usr/bin/env bash
# A1.2 B22 already-exposed DEV8: 128 independent native K4 GPU trajectories.
# ONE physical GPU3, sequential, <= 3h, explicit dated run authorization.
# No set -e/-u/pipefail or exit: keep PAC interactive terminal after STOP.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
BRANCH="codex/prdiff-track-a-a1-oracle-budget-20261016"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$BASE/a1_2_exposed_b22_k4_v1"
LOG="$BASE/A1_2_K4_GPU_MAIN_V1.log"
ZIP="$BASE/A1_2_K4_GPU_METADATA_RETURN_V1.zip"
CPU="$BASE/A1_2_K4_CPU_PLAN_RETURN_V1.zip"
CPU_SHA="7777d36f78464a0782d3cde446f88f53fd7b98d0a2b1fde70afff767e0d4dd20"
UUID="GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a"
RESERVATION="TRACK_A_A1_2_EXPOSED_B22_K4_GPU3_20261019_NIGHT"
STATUS=0
NOWUTC=$(date -u +%Y%m%d%H%M%S)
echo "A1_2_GPU_ATTEMPT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$A1_EXPECTED_HEAD" = "" ] || [ "$A1_RUN_RESERVATION_ACK" != "$RESERVATION" ]; then
  echo "A1_2_STOP: need pinned source and manual exact [RUN RESERVATION] acknowledgement"
  STATUS=2
elif [ "$A1_PHYSICAL_GPU" != "3" ]; then
  echo "A1_2_STOP: reservation only covers physical GPU3"
  STATUS=2
elif [ "$NOWUTC" -lt 20261020023000 ] || [ "$NOWUTC" -gt 20261020050000 ]; then
  echo "A1_2_STOP: launch outside 2026-10-19 22:30 to Oct20 01:00 EDT approved START window"
  echo "A1_2_NOTE: no log/ZIP/output written, renew reservation instead"
  STATUS=2
elif [ ! -d "$REPO/.git" ] || [ ! -x "$PY" ]; then
  echo "A1_2_STOP: dedicated checkout/Python missing"
  STATUS=2
elif [ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ] || \
     [ "$(git -C "$REPO" rev-parse HEAD)" != "$A1_EXPECTED_HEAD" ]; then
  echo "A1_2_STOP: source branch/commit mismatch"
  STATUS=2
elif [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "A1_2_STOP: tracked source dirty"
  STATUS=2
elif [ -e "$OUT" ] || [ -e "$LOG" ] || [ -e "$ZIP" ]; then
  echo "A1_2_STOP: this output namespace already contains a run, preserve it"
  STATUS=2
elif [ ! -f "$CPU" ] || [ "$(sha256sum "$CPU" | awk '{print $1}')" != "$CPU_SHA" ]; then
  echo "A1_2_STOP: independently verified prior CPU 128-candidate plan receipt absent/drifted"
  STATUS=2
else
  GPU=$(nvidia-smi --id=3 \
    --query-gpu=uuid,memory.free,memory.used \
    --format=csv,noheader,nounits 2>/dev/null)
  GPU_RC=$?
  echo "A1_2_GPU3_SNAPSHOT=$GPU"
  APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null)
  APPS_RC=$?
  echo "A1_2_GPU_COMPUTE_APP_QUERY_RC=$APPS_RC"
  if [ "$GPU_RC" -ne 0 ] || [ "$APPS_RC" -ne 0 ]; then
    echo "A1_2_STOP: GPU telemetry not readable"
    STATUS=2
  elif echo "$APPS" | grep -F -q "$UUID"; then
    echo "A1_2_STOP: GPU3 has active compute-app process, give Track B priority"
    STATUS=2
  elif [ "$(echo "$GPU" | awk -F, '{gsub(/ /,"",$1);print $1}')" != "$UUID" ]; then
    echo "A1_2_STOP: physical GPU3 UUID mismatch"
    STATUS=2
  else
    echo "A1_2_GPU_HEAD=$A1_EXPECTED_HEAD"
    echo "A1_2_STAGE=128_SEQUENTIAL_NATIVE_K4_EXPOSED_DEV_ONLY"
    PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES=3 \
      "$PY" -B "$REPO/scripts/track_a/a1_2_k4_exposed_b22.py" \
        --mode execute --expected-head "$A1_EXPECTED_HEAD" \
        --physical-gpu 3 --reservation-ack "$RESERVATION" 2>&1 | tee "$LOG"
    GPU_RC=\${PIPESTATUS[0]}
    echo "A1_2_K4_NATIVE_RC=$GPU_RC"
    if [ "$GPU_RC" -ne 0 ]; then STATUS=3; fi
  fi
fi

# ZIP: small metadata, partial receipts, and bounded latest four log tails.
# NO face PNGs, GT/image/measurement/reconstruction tensors.
if [ -f "$LOG" ] && [ ! -e "$ZIP" ]; then
  "$PY" -B - "$BASE" "$OUT" "$LOG" "$ZIP" <<'PY'
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,sys
base,out,log,archive=map(Path,sys.argv[1:])
files=[log]
if out.is_dir():
    for name in ("RUN_IDENTITY.json","A1_2_COMPLETE.json","A1_2_STOP.json",
                 "A1_2_OFFLINE_ORACLE_SUMMARY.json"):
        p=out/name
        if p.is_file():files.append(p)
    for pat in ("*_COMPLETE.json","cuda_timing.json","result.json"):
        files.extend(sorted(out.rglob(pat)))
logs=sorted(out.rglob("run.log"),key=lambda p:p.stat().st_mtime)[-4:] if out.is_dir() else []
with ZipFile(archive,"x",compression=ZIP_DEFLATED,compresslevel=6) as z:
    seen=set()
    for p in files:
        if p in seen:continue
        seen.add(p)
        if p.stat().st_size>12_000_000:
            raise RuntimeError(f"refuse abnormally large metadata {p}")
        z.write(p,p.relative_to(base))
    for p in logs:
        z.writestr(str(p.relative_to(base))+".tail.txt",p.read_bytes()[-32768:])
print("A1_2_K4_GPU_METADATA_ZIP",archive)
print("A1_2_K4_GPU_METADATA_SHA256",hashlib.sha256(archive.read_bytes()).hexdigest())
print("A1_2_K4_GPU_RECORD_COUNT",len(files))
PY
  PACK_RC=$?
  echo "A1_2_K4_GPU_PACK_RC=$PACK_RC"
  if [ "$PACK_RC" -ne 0 ]; then STATUS=4; fi
fi
echo "A1_2_K4_GPU_FINAL_STATUS=$STATUS"
test "$STATUS" -eq 0
