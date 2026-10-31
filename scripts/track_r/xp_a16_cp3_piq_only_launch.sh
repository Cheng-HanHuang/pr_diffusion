#!/usr/bin/env bash
# CPU-only SSIM/VGG-LPIPS evaluation from already produced native tensors.
# Does not rerun CLIP/GPU stages or publish CP3.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
VENDOR="$BASE/.vendor/piq_0p8"
HEAD="$1"
test -n "$HEAD"
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$VENDOR:$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$VENDOR/piq"
test ! -e "$BASE/XP_A16_CP3_PIQ_LATEST_LOG.txt" || {
  echo "STOP_CP3_PIQ_ALREADY_LAUNCHED";exit 8;
}
if pgrep -af '[x]p_a16_cp3_piq_metrics_cpu.py|[x]p_a16_cp3_piq_only_launch.sh'; then
 echo "STOP_CP3_COMPETING_METRICS";exit 8
fi
cd "$WT"
echo "===== INDEPENDENT OFFICIAL METRIC PREFLIGHTS ====="
for stage in G4 G5 G3; do
 "$PY" scripts/track_r/xp_a16_cp3_piq_metrics_cpu.py \
    --expected-head "$HEAD" --stage "$stage" \
    --stamp 20261030T210259Z --preflight-only
done
echo "===== LAUNCH READ-ONLY NATIVE FLOAT METRICS ====="
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$BASE/XP_A16_CP3_PIQ_$STAMP.log"
SESSION="prdiff_a16_cp3_piq_$STAMP"
test ! -e "$LOG"
CMD="set -euo pipefail; cd '$WT'; "
for stage in G4 G5 G3; do
 printf -v DEST '%s/xp_a16_cp3_piq_%s_%s' "$BASE" "$stage" "$STAMP"
 CMD+="echo '===== OFFICIAL PIQ STAGE $stage ====='; "
 CMD+="'$PY' scripts/track_r/xp_a16_cp3_piq_metrics_cpu.py --expected-head '$HEAD' --stage '$stage' --stamp 20261030T210259Z --out '$DEST'; "
done
CMD+="echo 'CP3_PIQ_THREE_NATIVE_FLOAT_STAGES_PASS__CP3_RELEASE_STILL_BLOCKED'; "
tmux new-session -d -s "$SESSION" \
 "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$VENDOR:$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 bash -c $(printf %q "$CMD") > '$LOG' 2>&1"
printf '%s\n' "$LOG" > "$BASE/XP_A16_CP3_PIQ_LATEST_LOG.txt"
printf '%s\n' "$SESSION" > "$BASE/XP_A16_CP3_PIQ_LATEST_SESSION.txt"
echo "A16_CP3_PIQ_ONLY_LAUNCHED"
echo "LOG=$LOG"
echo "SESSION=$SESSION"
echo "NO_CP3_PLANNER_RETURN_UNTIL_MATCHED_NFE_AND_FINAL_AUDIT"
