#!/usr/bin/env bash
# A16.5 ImageNet-disjoint XP-N: 90 image-aligned native FAST chunks,
# 6400 independent roots; preregistered v5-code gate or Nov 2 lapse.
# Schedules into tmux, never launches GPU work before the code-freeze gate.
# CPU scoring and publication remain separate behind the v5-output seal.
set -euo pipefail
[[ $# == 3 && "$1" == "launch" ]] || {
  echo "Usage: bash $0 launch PINNED_HEAD TASKS_DIR" >&2
  exit 8
}
HEAD="$2"
TASKS="$3"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$PY"
cd "$WT"

echo "===== A18 CPU REGRESSION AND AST ====="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a18_planner_a16.py' -v

echo "===== REQUIRE ONE NATIVE D1 PIXEL-EXACT REPLAY RECEIPT ====="
REPLAY="$("$PY" - "$BASE" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
rs=[]
for p in base.glob("xp_a18_xpn_replay_*/XP_N_D1_REPLAY_RECEIPT.json"):
    r=json.loads(p.read_text())
    if r.get("status")=="A18_XPN_D1_REPLAY2_PIXEL_EXACT_PASS":
        assert r["n_roots"]==2 and r["native_worker_rc"]==0
        assert r["visibility_unknown"] is False
        assert all(z["pixels_exact"] for z in r["pixel_comparisons"])
        rs.append(p.parent)
assert len(rs)==1, f"Expected one valid replay, got {rs}"
print(rs[0])
PY
)"
echo "A18_XPN_REPLAY=$REPLAY"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a18_xpn_prod_$STAMP"
LOG="$BASE/XP_A18_XPN_AUTOPILOT_$STAMP.log"
SESSION="prdiff_a18_xpn_$STAMP"
test ! -e "$RUN" && test ! -e "$LOG"

echo "===== PROSPECTIVE SOURCE/BUDGET AND REGISTRY PREFLIGHT ====="
"$PY" scripts/track_r/xp_a18_xpn_autopilot.py \
    --expected-head "$HEAD" --tasks "$TASKS" \
    --replay "$REPLAY" --run "$RUN" --preflight-only

echo "===== FOUR-GPU A10 CO-TENANCY INVENTORY ====="
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

echo "===== AUTOPILOT LAUNCH IN DETACHED TMUX ====="
tmux new-session -d -s "$SESSION" -c "$WT" \
  "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 '$PY' -u -m scripts.track_r.xp_a18_xpn_autopilot --expected-head '$HEAD' --tasks '$TASKS' --replay '$REPLAY' --run '$RUN' > '$LOG' 2>&1"

echo "A18_XPN_TMUX_SESSION=$SESSION"
echo "A18_XPN_LOG=$LOG"
echo "A18_XPN_RUN=$RUN"
echo "A18_XPN_OPERATOR_STOP_FILE=$RUN/STOP_XPN"
echo "A18_XPN_PROSPECTIVE_V5_GATE_AND_6400_NATIVE_ROOTS_SUBMITTED"
