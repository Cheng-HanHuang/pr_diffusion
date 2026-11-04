#!/usr/bin/env bash
# Autonomous A23 S_mid global2 only: full 280 GT diagnostic (278 new + 2 A33)
# CPU preflights -> co-tenant-safe union-budgeted GPU controller -> CPU readout
# -> separate GitHub result branch publication. Never repeats G7 C2/CLIP or
# launches any XP-K/new selector/protocol v2. Fail closed on science gates.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: bash scripts/track_r/xp_a34_a23_g7_smid_launch.sh <exact-A34-commit>" >&2
    exit 8
fi
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/daps/bin/python"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a34_gt_cpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$WT"

echo "=== Exact source pin; non-dirty A34 worktree ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test -f docs/track_r/executors/xp/theory_inputs/XP_A23_G7_GT_SOURCE_LOCK.json
test -f scripts/track_r/xp_a34_a23_g7_smid_controller_cpu.py
test -f scripts/track_r/xp_a34_a23_g7_smid_prod_gpu.py
test -f scripts/track_r/xp_a34_a23_g7_smid_readout_cpu.py
test -f scripts/track_r/xp_a34_a23_g7_smid_publish_cpu.py
echo "A34_SOURCE_PIN_PASS=$HEAD"

echo "=== Real CPU test suite and Bash syntax ==="
bash -n scripts/track_r/xp_a34_a23_g7_smid_launch.sh
"$PY" -m unittest discover -s tests \
    -p 'test_track_r_xp_a34_a23_g7_prod.py' -v

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a34_a23_g7_smid_prod_$STAMP"
PREFLIGHT="$BASE/xp_a34_a23_g7_smid_prod_preflight_$STAMP"
test ! -e "$RUN"
test ! -e "$PREFLIGHT"

echo "=== Check frozen A33 pilot, A22 C2 ledger, no new GPU ==="
"$PY" -u scripts/track_r/xp_a34_a23_g7_smid_controller_cpu.py \
    --expected-head "$HEAD" --run "$RUN" --preflight-only

echo "=== Native source and FFHQ/ImageNet checkpoint preflight, CUDA hidden ==="
mkdir -m 700 "$PREFLIGHT"
mkdir "$PREFLIGHT/results" "$PREFLIGHT/pycache"
for i in 0 180; do
    "$PY" -u scripts/track_r/xp_a34_a23_g7_smid_prod_gpu.py \
        --expected-head "$HEAD" --run "$PREFLIGHT" \
        --index "$i" --gpu 0 --preflight-only
done

echo "=== Controller owns global Track R GPU ledger lock, no foreign signals ==="
echo "A34_G7_PRODUCTION_RUN=$RUN"
echo "A34_G7_S_MID_ONLY_REUSE_TWO_PILOTS=0,180"
# The native controller itself checks the global lock and live per-card
# admission before launching one child per safely eligible physical GPU.
"$PY" -u scripts/track_r/xp_a34_a23_g7_smid_controller_cpu.py \
    --expected-head "$HEAD" --run "$RUN"

echo "=== Read terminal stage status; do not publish a partial result ==="
if [ ! -f "$RUN/XP_A34_G7_S_MID_FINAL_STATUS.json" ]; then
    echo "A34_G7_PARTIAL_SAFETY_STOP_RETAIN_RUN=$RUN"
    echo "A34_G7_NO_PUBLICATION_OF_INCOMPLETE_280_GRID"
    exit 0
fi

"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
s=json.loads((run/"XP_A34_G7_S_MID_FINAL_STATUS.json").read_text())
assert s["status"]=="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
assert s["n_complete"]==280
assert s["G7_total_charged_gpu_h"]<10.0
assert s["project_charged_gpu_h"]<400.0
print("A34_G7_GT_ALL_280_COMPLETED_PROJECT_H",
      s["project_charged_gpu_h"])
PY

echo "=== GT-only upright-retention analysis, CPU only ==="
"$PY" -u scripts/track_r/xp_a34_a23_g7_smid_readout_cpu.py \
    --expected-head "$HEAD" --run "$RUN"

echo "=== Independent SHA-verified results branch, CPU only ==="
"$PY" -u scripts/track_r/xp_a34_a23_g7_smid_publish_cpu.py \
    --expected-head "$HEAD" --run "$RUN"

echo "A34_G7_S_MID_GLOBAL2_AUTO_COMPLETED_AND_PUBLISHED"
echo "A34_G7_NEXT_REGISTERED_METHOD=S_mid_all8"
echo "A34_G7_NO_FULL_G7_CLAIM"
echo "A34_G7_CP4_SUPPLEMENT_ONLY"
