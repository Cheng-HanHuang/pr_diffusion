#!/usr/bin/env bash
# A13 first executable handoff: G1b FAST 3200 roots and B16 protocol CPU.
# Run ONLY from a fresh detached, clean, pinned A13 XP worktree.
# Usage: bash scripts/track_r/xp_a13_cp1_g1b_start.sh <40-char HEAD>
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
FAST="$ROOT/conda-envs/fast_dips_a15_blackwell/bin/python"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_EXPECTED_SHA40;exit 8; }
test -d "$WT"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd "$WT"
echo '===== A13 PINNED CODE CPU UNIT TESTS ====='
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a12_g2_g7.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a12_g1a*.py' -v
echo '===== REQUIRE ORIGINAL PROTOCOL COMMIT LOCALLY ====='
git cat-file -e 'c598176:docs/track_r/protocol/gtfree_protocol.py'
echo '===== A13 FROZEN G1B CPU PREPARATION ====='
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PREP="$BASE/xp_a12_g1b_prepare_$STAMP"
RUN="$BASE/xp_a13_g1b_prod_$STAMP"
B16="$BASE/xp_a13_b16_protocol_$STAMP"
"$PY" scripts/track_r/xp_a12_g1b_prepare_cpu.py \
 --expected-head "$HEAD" --out "$PREP"
"$PY" scripts/track_r/xp_a13_g1b_autopilot.py \
 --expected-head "$HEAD" --prep "$PREP" --run "$RUN" --preflight-only
echo '===== NATIVE FAST SOURCE / MODEL / Y SHA PRE-GPU CHECK ====='
MF="$PREP/XP_A12_G1B_3200_TASKS.json"
DIGEST=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["manifest_sha256"])' \
 "$PREP/XP_A12_G1B_PREP_RECEIPT.json")
for IDX in 0 3168; do
 CUDA_VISIBLE_DEVICES=0 "$FAST" scripts/track_r/xp_a12_campaign_fast_gpu.py \
  --expected-head "$HEAD" --roots "$MF" --roots-sha "$DIGEST" \
  --roots-source-head "$HEAD" --out "$RUN/production/unused_preflight_$IDX" \
  --start "$IDX" --count 32 --physical-gpu 0 --dataset imagenet \
  --oversample 2.0 --preflight-only
done
echo '===== DETACHED G1B RUN, 4 GPUS ====='
SESSION=prdiff_a13_g1b_$STAMP
LOG="$BASE/XP_A13_G1B_$STAMP.log"
test ! -e "$LOG"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 '$PY' scripts/track_r/xp_a13_g1b_autopilot.py --expected-head '$HEAD' --prep '$PREP' --run '$RUN' > '$LOG' 2>&1"
echo '===== DETACHED B16 FROZEN PROTOCOL CPU ====='
SESSIONB=prdiff_a13_b16_$STAMP
LOGB="$BASE/XP_A13_B16_PROTOCOL_$STAMP.log"
test ! -e "$LOGB"
tmux new-session -d -s "$SESSIONB" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 '$PY' scripts/track_r/xp_a13_b16_protocol_cpu.py --expected-head '$HEAD' --out '$B16' > '$LOGB' 2>&1"
printf '%s\n' "$RUN" > "$BASE/XP_A13_G1B_LATEST_RUN.txt"
printf '%s\n' "$B16" > "$BASE/XP_A13_B16_PROTOCOL_LATEST_RUN.txt"
echo "A13_G1B_SESSION=$SESSION"
echo "A13_G1B_RUN=$RUN"
echo "A13_G1B_LOG=$LOG"
echo "A13_G1B_PREP=$PREP"
echo "A13_B16_SESSION=$SESSIONB"
echo "A13_B16_PROTOCOL_OUTPUT=$B16"
echo "A13_B16_PROTOCOL_LOG=$LOGB"
echo 'A13_G1B_B16_LAUNCHED'
