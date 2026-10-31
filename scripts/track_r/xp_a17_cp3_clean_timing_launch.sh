#!/usr/bin/env bash
# A15 authorized CPU controller, six serial native GPU timing processes:
# 3 PC-DAPS bags + 3 DAPS-4K runs on exactly three same-y FFHQ IDs.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test -n "$HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"
"$PY" scripts/track_r/xp_a17_cp3_clean_timing_cpu.py \
 --expected-head "$HEAD" --preflight-only
if test -f "$BASE/XP_A17_CP3_TIMING_LATEST_LOG.txt";then
 echo "STOP_A17_TIMING_LAUNCH_ALREADY_EXISTS";exit 8
fi
SESSION="prdiff_a17_clean_$(date -u +%Y%m%dT%H%M%SZ)"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BASE/xp_a17_cp3_timing_$STAMP"
LOG="$BASE/XP_A17_CP3_TIMING_$STAMP.log"
test ! -e "$OUT" && test ! -e "$LOG"
if pgrep -af '[p]ython.*xp_a17_cp3_clean_timing_cpu.py';then
 echo "STOP_A17_TIMING_OTHER_CONTROLLER";exit 8
fi
CMD="'$PY' scripts/track_r/xp_a17_cp3_clean_timing_cpu.py --expected-head '$HEAD' --out '$OUT'; "
CMD+="echo A17_CP3_CLEAN_TIMING_SESSION_COMPLETE"
tmux new-session -d -s "$SESSION" \
 "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 bash -c $(printf %q "set -euo pipefail; cd '$WT'; $CMD") > '$LOG' 2>&1"
printf '%s\n' "$LOG" > "$BASE/XP_A17_CP3_TIMING_LATEST_LOG.txt"
printf '%s\n' "$OUT" > "$BASE/XP_A17_CP3_TIMING_LATEST_OUT.txt"
printf '%s\n' "$SESSION" > "$BASE/XP_A17_CP3_TIMING_LATEST_SESSION.txt"
echo "A17_CP3_CLEAN_TIMING_DETACHED_LAUNCHED"
echo "LOG=$LOG"
echo "OUT=$OUT"
echo "SESSION=$SESSION"
echo "NO_CP3_MATCHED_NFE_CLAIM"
