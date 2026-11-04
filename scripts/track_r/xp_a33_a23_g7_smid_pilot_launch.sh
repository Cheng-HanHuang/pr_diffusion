#!/usr/bin/env bash
# A23 bounded 2-image GT-only S_mid global2 numerical pilot.
# One GPU at a time; no inverse-problem sampling or other completed stage reruns.
# STOP on science/equivalence/source/admission gate; no automatic retry.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/track_r/xp_a33_a23_g7_smid_pilot_launch.sh <exact-A23-Git-head>" >&2
  exit 8
fi
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/daps/bin/python"
export PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
# Serialize global Track R GPU campaign controllers.
exec 9>"$BASE/XP_A12_CAMPAIGN_CONTROLLER.lock"
if ! flock -n 9; then
  echo "A23_G7_PILOT_PARKED_TRACK_R_GPU_CONTROLLER_ACTIVE"
  exit 0
fi
echo "=== CPU-only source/protocol and numerical mock tests ==="
export CUDA_VISIBLE_DEVICES=""
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a23_source_cpu"
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a33_a23_g7_pilot.py' -v
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a33_a23_g7_smid_pilot_$STAMP"
test ! -e "$RUN"
mkdir -m 700 "$RUN"
mkdir "$RUN/logs" "$RUN/pycache"
# Fresh ledger reconciliation required before future production.
"$PY" - "$BASE" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
old=base/"xp_a12_g7_prod_a22_20261104T025355Z/XP_A22_G7_C2_FINAL_STATUS.json"
j=json.loads(old.read_text())
assert j["status"]=="A22_G7_C2_280_COMPLETE"
assert j["active"]==[]
assert abs(float(j["project_gpu_h"])-244.45709476042322)<1e-8
assert abs(float(j["gpu_occupancy_h"])-2.6265298946173345)<1e-8
for f in base.glob("xp_*/XP_*FINAL_STATUS.json"):
    try:
        d=json.loads(f.read_text())
    except (OSError,ValueError):
        continue
    current=d.get("project_gpu_h")
    if isinstance(current,(int,float)) and current>j["project_gpu_h"]+1e-8:
        raise RuntimeError("STOP_A23_G7_LEDGER_ADVANCED "+str(f))
print("A23_G7_LEDGER_HISTORICAL_BASELINE_CHECK_PASS")
PY
echo "=== Two prospective source-only GT checks; CUDA hidden ==="
for index in 0 180; do
    "$PY" -u scripts/track_r/xp_a33_a23_g7_smid_pilot_gpu.py \
      --expected-head "$HEAD" --run "$RUN" \
      --index "$index" --gpu 0 --preflight-only
done
echo "=== Read-only co-tenant GPU admission ==="
GPU=$("$PY" - <<'PY'
import subprocess,math,sys
s=subprocess.check_output([
 "nvidia-smi",
 "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
 "--format=csv,noheader,nounits"],text=True,timeout=30)
eligible=[]
reserve=math.ceil(16384*1.25+8192)
for row in s.splitlines():
    fields=[x.strip() for x in row.split(",")]
    if len(fields)!=5:raise RuntimeError("STOP_A23_GPU_INVENTORY_FORMAT")
    i,total,used,free,util=map(int,fields)
    if 0<=i<=3 and free>=reserve and used+reserve<=int(.8*total):
        eligible.append((util,-free,i))
if not eligible:
    print("STOP_NO_COTENANT_SAFE_GPU",file=sys.stderr)
    raise SystemExit(8)
print(min(eligible)[2])
PY
) || {
  echo "A23_G7_PILOT_PARKED_NO_SAFE_GPU"
  exit 0
}
echo "A23_G7_CO_TENANT_CANDIDATE_GPU=$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== Native A20 vs A21 pilot on frozen FFHQ and ImageNet GT ==="
for index in 0 180; do
  export PYTHONPYCACHEPREFIX="$RUN/pycache/image_$index"
  echo "A23_G7_PILOT_START index=$index gpu=$GPU"
  timeout --signal=TERM --kill-after=15 1800 \
      "$PY" -u scripts/track_r/xp_a33_a23_g7_smid_pilot_gpu.py \
      --expected-head "$HEAD" --run "$RUN" \
      --index "$index" --gpu "$GPU" \
      > "$RUN/logs/PILOT_$index.log" 2>&1 || {
        echo "A23_G7_S_MID_PILOT_STOP index=$index"
        tail -n 60 "$RUN/logs/PILOT_$index.log"
        exit 8
      }
  tail -n 4 "$RUN/logs/PILOT_$index.log"
done
export CUDA_VISIBLE_DEVICES=""
"$PY" - "$RUN" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1])
a=json.loads((r/"G7_S_MID_GLOBAL2_PILOT_000.json").read_text())
b=json.loads((r/"G7_S_MID_GLOBAL2_PILOT_180.json").read_text())
assert a["status"]==b["status"]=="A23_G7_S_MID_GLOBAL2_TWO_IMAGE_PILOT_PASS"
assert (a["dataset"],b["dataset"])==("ffhq","imagenet")
assert a["B2_B8_choice_equal"] and b["B2_B8_choice_equal"]
used=a["gpu_occupancy_h"]+b["gpu_occupancy_h"]
assert used<1.0 and used<7.3734701053826655
print("A23_G7_S_MID_GLOBAL2_TWO_DATASET_PILOT_PASS",used)
print("A23_G7_PILOT_RUN_ROOT",r)
print("A23_NO_PRODUCTION_GPU_GENERATION_OR_FULL_G7_CLAIM")
PY
