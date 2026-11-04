#!/usr/bin/env bash
# A23 single autonomous G7 GT-only queue: S_mid all8 -> 2-forward -> UnivFD.
# Native charged canaries, persistent GPU workers, CPU UnivFD, per-stage
# independent SHA publication. No GPU job starts before CPU tests/preflights.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: bash scripts/track_r/xp_a35_g7_launch.sh <exact-A35-commit>" >&2
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a35_cpu"
cd "$WT"

echo "=== Frozen A35 exact Git source and prereg structure ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test -f docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json
test -f scripts/track_r/xp_a35_g7_scoring.py
test -f scripts/track_r/xp_a35_g7_worker.py
test -f scripts/track_r/xp_a35_g7_autopilot_cpu.py
test -f scripts/track_r/xp_a35_g7_readout_publish_cpu.py
bash -n scripts/track_r/xp_a35_g7_launch.sh
echo "A35_SOURCE_PIN_PASS $HEAD"

echo "=== CPU-only A35 deterministic math and safety regression ==="
"$PY" -m unittest discover -s tests \
    -p 'test_track_r_xp_a35_g7_remaining.py' -v

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a35_g7_remaining_$STAMP"
PREF="$BASE/xp_a35_g7_remaining_preflight_$STAMP"
test ! -e "$RUN"
test ! -e "$PREF"
mkdir -m 700 "$PREF"

echo "=== Preflight both native dataset models and pinned CPU UnivFD source ==="
for METHOD in S_mid_all8 2_forward UnivFD; do
    mkdir "$PREF/$METHOD"
    mkdir "$PREF/$METHOD/tasks" "$PREF/$METHOD/results"
    mkdir "$PREF/$METHOD/logs" "$PREF/$METHOD/pycache"
    echo '[0]' > "$PREF/$METHOD/tasks/ffhq.json"
    echo '[180]' > "$PREF/$METHOD/tasks/imagenet.json"
    for DATASET in ffhq imagenet; do
        "$PY" -u scripts/track_r/xp_a35_g7_worker.py \
            --expected-head "$HEAD" \
            --run "$PREF" \
            --method "$METHOD" \
            --dataset "$DATASET" \
            --tasks "$PREF/$METHOD/tasks/$DATASET.json" \
            --preflight-only
    done
done

echo "=== Source A34 publication and immutable G7 ledger preflight ==="
"$PY" -u scripts/track_r/xp_a35_g7_autopilot_cpu.py \
    --expected-head "$HEAD" \
    --run "$RUN" \
    --preflight-only

echo "=== Launch single ordered controller; no methods skipped ==="
echo "A35_G7_RUN_ROOT=$RUN"
echo "A35_G7_PREVIOUS_FINAL_H=5.1570269519018215"
echo "A35_G7_TOTAL_GPU_CEILING_H=10"
echo "A35_G7_NEXT_ORDER=S_mid_all8,2_forward,UnivFD"
"$PY" -u scripts/track_r/xp_a35_g7_autopilot_cpu.py \
    --expected-head "$HEAD" \
    --run "$RUN"

echo "=== Final A35 terminal marker ==="
"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
f=r/"A35_G7_AUTOPILOT_FINAL_STATUS.json"
assert f.is_file()
s=json.loads(f.read_text())
assert s["G7_total_gpu_h"]<10.0
assert s["project_gpu_h"]<400.0
if s["status"]=="A35_G7_THREE_REMAINING_METHODS_PUBLISHED":
    assert len(s["not_run"])==0
    assert all(s["method_results"][k]["published"]
               for k in ("S_mid_all8","2_forward","UnivFD"))
    print("A35_G7_ALL_REMAINING_METHODS_PUBLISHED",r,flush=True)
else:
    print("A35_G7_PARTIAL_SAFE_STOP",s["status"],
          "not_run",s["not_run"],
          "run",r,flush=True)
    print("A35_G7_DO_NOT_RESTART_FROM_SCRATCH",flush=True)
PY
