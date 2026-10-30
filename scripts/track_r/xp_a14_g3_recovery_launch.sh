#!/usr/bin/env bash
# Recovery of G3 after an admission-only path check failed.
# Reuses CPU-certified native signed-y task manifest. No replay of G2/G4/G5,
# no old G3 outputs overwritten and no changes to solver configuration.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
OLD_PREP="$BASE/xp_a12_g3_prepare_20261030T153655Z"
OLD_RUN="$BASE/xp_a12_g3_prod_20261030T153655Z"
HEAD="$1"
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo STOP_EXPECTED_SHA40;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$(cat "$BASE/XP_A14_G3_LATEST_RUN.txt")" = "$OLD_RUN"
test ! -e "$OLD_RUN"
test ! -e "$BASE/XP_A14_G3_RECOVERY_LATEST_RUN.txt"
test -f "$OLD_PREP/XP_A12_G3_PREP_RECEIPT.json"
test -f "$OLD_PREP/XP_A12_G3_180_BAGS.json"
test "$(sha256sum "$OLD_PREP/XP_A12_G3_180_BAGS.json" | awk '{print $1}')" = \
  3ad52d84216ccd57c14d8b59c3a643a9b89f59adfce908dac2f43d4d1639ff84
test "$(git -C "$ROOT/external/PC_DAPS_xp" rev-parse HEAD)" = \
  59b2f0461134ca83bb99d61c648f38cd4a9cd4a0
if pgrep -af '[x]p_a12_campaign_autopilot.py|[x]p_a14_g3_after_clone_autopilot.py|[x]p_a12_g6_clean_autopilot.py|[x]p_a12_g7_c2_autopilot.py';then
 echo STOP_CONCURRENT_TRACK_R_CONTROLLER;exit 8
fi
if find "$BASE" -maxdepth 1 -type d -name 'xp_a12_g3_prod_*' -print -quit | grep -q .;then
 echo STOP_EXISTING_G3_OUTPUTS;exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PRIOR_RUN=$(cat "$BASE/XP_A13_G2G5_LATEST_RUN.txt")
PRIOR="$PRIOR_RUN/XP_A12_CAMPAIGN_STATE.json"
test -f "$PRIOR"
cd "$WT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a12_g3_prod_$STAMP"
LOG="$BASE/XP_A14_G3_RECOVERY_$STAMP.log"
SESSION="prdiff_a14_g3_recovery_$STAMP"
test ! -e "$RUN" && test ! -e "$LOG"
echo "===== A14 G3 ADMISSION DRY-RUN: NO GPU, NO NEW OUTPUT ====="
"$PY" scripts/track_r/xp_a14_g3_after_clone_autopilot.py \
 --expected-head "$HEAD" --prior-campaign "$PRIOR" \
 --prep "$OLD_PREP" --run "$RUN" --preflight-only
test ! -e "$RUN"
echo "===== LAUNCH FROZEN TWO-BAG G3 SMOKE / CONDITIONAL PRODUCTION ====="
printf -v CMD '%q ' \
 "$PY" scripts/track_r/xp_a14_g3_after_clone_autopilot.py \
 --expected-head "$HEAD" --prior-campaign "$PRIOR" \
 --prep "$OLD_PREP" --run "$RUN"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $CMD > '$LOG' 2>&1"
printf '%s\n' "$RUN" > "$BASE/XP_A14_G3_RECOVERY_LATEST_RUN.txt"
echo "A14_G3_RECOVERY_SESSION=$SESSION"
echo "A14_G3_RECOVERY_LOG=$LOG"
echo "A14_G3_RECOVERY_RUN=$RUN"
echo "A14_G3_RECOVERY_AUTOPILOT_LAUNCHED"
