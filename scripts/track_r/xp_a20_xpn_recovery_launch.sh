#!/usr/bin/env bash
# Resume the exact original XP-N 6,400-root native campaign after A10
# external GPU-VRAM interference. No scientific or seed changes.
# XP-I chosen-metric lane is already independent. v5 is triggered ONLY
# after 400 prefix candidate receipts complete and then published before
# any >=4 scoring. This launcher does not score held indices.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 PINNED_RECOVERY_HEAD" >&2; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
TASKS="$BASE/xp_a18_xpn_tasks_20261031T224645Z"
ORIGINAL="$BASE/xp_a18_xpn_prod_20261031T225748Z"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$TASKS" && test -f "$ORIGINAL/XP_N_FINAL_STATUS.json"
cd "$WT"

echo "=== A18/A19/A20 CPU SOURCE QA ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a18_planner_a16.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a19_planner_a17.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a20_recovery.py' -v
bash -n scripts/track_r/xp_a20_xpn_recovery_launch.sh \
  scripts/track_r/xp_a20_v5_after_recovery.sh \
  scripts/track_r/xp_a19_xpn_v5_autopilot.sh
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RECOVER="$BASE/xp_a20_xpn_recovery_$STAMP"
RLOG="$BASE/XP_A20_XPN_RECOVERY_$STAMP.log"
VLOG="$BASE/XP_A20_XPN_V5_AFTER_RECOVERY_$STAMP.log"
RTMUX="prdiff_a20_xpn_resume_$STAMP"
VTMUX="prdiff_a20_v5_resume_$STAMP"
test ! -e "$RECOVER" && test ! -e "$RLOG" && test ! -e "$VLOG"
echo "=== READ-ONLY RECEIPT AND GPU-ADMISSION PREFLIGHT ==="
"$PY" scripts/track_r/xp_a20_xpn_recover_gpu.py \
   --expected-head "$HEAD" --out "$RECOVER" --preflight-only
echo "=== SUBMIT PINNED ORIGINAL-HEAD RECOVERY ==="
tmux new-session -d -s "$RTMUX" -c "$WT" \
 "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 '$PY' -u -m scripts.track_r.xp_a20_xpn_recover_gpu --expected-head '$HEAD' --out '$RECOVER' > '$RLOG' 2>&1"
echo "=== SUBMIT V5 CPU LANE LINKED TO THE RECOVERING PREFIX ==="
tmux new-session -d -s "$VTMUX" -c "$WT" \
 "bash '$WT/scripts/track_r/xp_a20_v5_after_recovery.sh' '$HEAD' '$ORIGINAL' '$RECOVER' '$TASKS' > '$VLOG' 2>&1"

echo "A20_XPN_RECOVERY_TMUX=$RTMUX"
echo "A20_XPN_RECOVERY_LOG=$RLOG"
echo "A20_XPN_RECOVERY_STATE=$RECOVER/XP_N_RECOVERY_STATE.json"
echo "A20_XPN_V5_TMUX=$VTMUX"
echo "A20_XPN_V5_LOG=$VLOG"
echo "A20_XPN_OPERATOR_STOP_FILE=$RECOVER/STOP_XPN_RECOVERY"
echo "A20_XPN_RECEIPT_PRESERVING_AUTOPILOT_SUBMITTED"
