#!/usr/bin/env bash
# A18 parallel lanes: CP-N n16 CPU return NOW; 28h GPU continuation
# independently with SHA-reconciled ledger. One native chunk smoke first.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: bash $0 PINNED_A22_HEAD" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"

echo "=== A22 FULL CPU SOURCE TESTS ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a18_planner_a16.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a19_planner_a17.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a20_recovery.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a22_planner_a18.py' -v
bash -n scripts/track_r/xp_a22_launch_a18.sh \
 scripts/track_r/xp_a22_cpn_part1_publish.sh

echo "=== 28H EXACT PRIOR COST AND NATIVE ROOT SHA PREFLIGHT ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a22_xpn_continue_$STAMP"
GLOG="$BASE/XP_A22_XPN_28H_CONTINUE_$STAMP.log"
CLOG="$BASE/XP_A22_CPN_PART1_$STAMP.log"
GPU_SESSION="prdiff_a22_xpn_28h_$STAMP"
CPU_SESSION="prdiff_a22_cpn_part1_$STAMP"
test ! -e "$OUT" && test ! -e "$GLOG" && test ! -e "$CLOG"
"$PY" scripts/track_r/xp_a22_xpn_continue_gpu.py \
  --expected-head "$HEAD" --out "$OUT" --preflight-only

echo "=== FULL 100-IMAGE/16-RUN PANEL RECEIPT PREFLIGHT ==="
"$PY" scripts/track_r/xp_a22_cpn_part1_cpu.py \
  --expected-head "$HEAD" --out "$BASE/xp_a22_cpn_part1_preflight_uncreated" \
  --preflight-only

echo "=== SUBMIT CPU CP-N PART1 (NO GPU AND NO V5 K16 TEST) ==="
tmux new-session -d -s "$CPU_SESSION" -c "$WT" \
 "bash '$WT/scripts/track_r/xp_a22_cpn_part1_publish.sh' '$HEAD' > '$CLOG' 2>&1"
echo "=== SUBMIT NATIVE 28H CONTINUATION (ONE SMOKE BEFORE FANOUT) ==="
tmux new-session -d -s "$GPU_SESSION" -c "$WT" \
 "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 '$PY' -u -m scripts.track_r.xp_a22_xpn_continue_gpu --expected-head '$HEAD' --out '$OUT' > '$GLOG' 2>&1"

echo "A22_XPN_GPU_TMUX=$GPU_SESSION"
echo "A22_XPN_GPU_LOG=$GLOG"
echo "A22_XPN_STATE=$OUT/XP_N_CONTINUATION_STATE.json"
echo "A22_XPN_OPERATOR_STOP_FILE=$OUT/STOP_XPN_CONTINUATION"
echo "A22_CPN_PART1_CPU_TMUX=$CPU_SESSION"
echo "A22_CPN_PART1_CPU_LOG=$CLOG"
echo "A22_A18_GPU_AND_INDEPENDENT_CPU_CP_N_SUBMITTED"
