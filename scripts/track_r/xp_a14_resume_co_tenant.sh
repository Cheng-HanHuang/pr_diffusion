#!/usr/bin/env bash
# A14 co-tenant recovery: charge prior G2 smoke; G4/G5 first, then G2 pilot.
# Does not repeat G1b or modify the terminated earlier G2 run.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_EXPECTED_SHA40;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
# The earlier controller ended after two valid G2 smoke roots.
# It must remain immutable; A14 bills its cost once inside the new controller.
OLD="$BASE/xp_a12_campaign_prod_20261030T002042Z"
test -f "$OLD/XP_A12_CAMPAIGN_STATE.json"
test "$(cat "$BASE/XP_A13_G2G5_LATEST_RUN.txt")" = "$OLD"
if pgrep -af '[x]p_a12_campaign_autopilot.py|[x]p_a13_g1b_autopilot.py'; then
 echo STOP_DUPLICATE_A14_CONTROLLER; exit 8
fi
if find "$BASE" -maxdepth 1 -type d -name 'xp_a12_campaign_prod_*' \
   ! -path "$OLD" -print -quit | grep -q .; then
 echo STOP_ANOTHER_CAMPAIGN_EXISTS; exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd "$WT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
G2PREP="$BASE/xp_a12_g2_prepare_$STAMP"
G45PREP="$BASE/xp_a12_g45_prepare_$STAMP"
G3PREP="$BASE/xp_a12_g3_prepare_$STAMP"
G6PREP="$BASE/xp_a12_g6_prepare_$STAMP"
AUDIT="$BASE/XP_A12_G367_CONDITIONAL_AUDIT_$STAMP.json"
RUN="$BASE/xp_a12_campaign_prod_$STAMP"
G1BRUN=$(cat "$BASE/XP_A13_G1B_LATEST_RUN.txt")
test -d "$G1BRUN"
case "$G1BRUN" in "$BASE"/xp_a13_g1b_prod_*) ;; *) echo STOP_G1B_PATH;exit 8;; esac
echo '===== CPU FREEZE G2 PAIRED-NOISE 360 OBSERVATIONS ====='
"$PY" scripts/track_r/xp_a12_g2_prepare_cpu.py \
 --expected-head "$HEAD" --out "$G2PREP"
echo '===== CPU FREEZE G4/G5 DAPS-4K CONTINUATIONS AND REPLAYS ====='
"$PY" scripts/track_r/xp_a12_g45_prepare_cpu.py \
 --expected-head "$HEAD" --out "$G45PREP"
echo '===== G3 PC-DAPS: INDEPENDENT A14 CERTIFICATION, NOT A G4/G5 BLOCKER ====='
PC_AVAILABLE=false
if [ -d "$ROOT/external/PC_DAPS_xp" ]; then
 echo 'G3_SOURCE_PRESENT: separate CPU adapter certification and 2-id smoke required'
else
 echo 'G3_CODE_ABSENT: Cheng must clone the exact upstream PC-DAPS source'
 echo 'git clone https://github.com/voilalab/PC-DAPS.git /egr/research-pac/huang248/external/PC_DAPS_xp'
 echo 'git -C /egr/research-pac/huang248/external/PC_DAPS_xp checkout --detach 59b2f046'
fi
echo '===== CPU READINESS FOR G6/G7 ====='
"$PY" scripts/track_r/xp_a12_g367_readiness_cpu.py \
 --expected-head "$HEAD" --out "$AUDIT"
if "$PY" scripts/track_r/xp_a12_g6_prepare_cpu.py \
 --expected-head "$HEAD" --out "$G6PREP"; then
 printf '%s\n' "$G6PREP" > "$BASE/XP_A13_G6_PREP_LATEST.txt"
else
 echo 'ACTION NEEDED: G6_OPTIONAL_TIMING_PREP_FAILED__G2_G5_PRIORITY_CONTINUES'
fi
echo '===== WAIT FOR G1B TERMINAL RECEIPT, NO GPU OVERLAP ====='
G1BSESSION=$(basename "$G1BRUN" | sed 's/^xp_a13_g1b_prod_/prdiff_a13_g1b_/')
while [ ! -f "$G1BRUN/XP_A13_G1B_FINAL_STATUS.json" ]; do
 if ! tmux has-session -t "$G1BSESSION" 2>/dev/null;then
  echo STOP_G1B_CONTROLLER_ENDED_WITHOUT_RECEIPT;exit 8
 fi
 sleep 10
done
set -- --expected-head "$HEAD" \
 --ledger "$BASE/XP_A12_PROJECT_LEDGER_VERIFIED.json" \
 --g1b-final "$G1BRUN/XP_A13_G1B_FINAL_STATUS.json" \
 --g2-prep "$G2PREP" --g45-prep "$G45PREP" --run "$RUN" --a14
if [ "$PC_AVAILABLE" = true ]; then
 set -- "$@" --g3-prep "$G3PREP"
fi
echo '===== A13 DUAL-LEDGER CPU ADMISSION PRE-GPU CHECK ====='
"$PY" scripts/track_r/xp_a12_campaign_autopilot.py "$@" --preflight-only
echo '===== A14 CO-TENANT G4/G5 FIRST, G2 PILOT LATER ====='
SESSION=prdiff_a14_g45_g2_$STAMP
LOG="$BASE/XP_A14_G45_G2_$STAMP.log"
test ! -e "$LOG"
printf -v QUOTED '%q ' "$PY" scripts/track_r/xp_a12_campaign_autopilot.py "$@"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $QUOTED > '$LOG' 2>&1"
printf '%s\n' "$RUN" > "$BASE/XP_A13_G2G5_LATEST_RUN.txt"
echo "A13_G2G5_SESSION=$SESSION"
echo "A13_G2G5_LOG=$LOG"
echo "A13_G2G5_RUN=$RUN"
echo "A13_G2_PREP=$G2PREP"
echo "A13_G45_PREP=$G45PREP"
echo "A13_G3_PREP=$G3PREP"
echo "A13_G6_PREP=$G6PREP"
echo 'A14_CO_TENANT_AUTOPILOT_LAUNCHED'
