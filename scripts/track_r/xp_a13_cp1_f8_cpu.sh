#!/usr/bin/env bash
# A13 CP1 F8 CPU scoring -> frozen N2 face CLIP -> v1/v3 -> protocol v1.
# Run detached alongside G1b, after B16 frozen protocol is ready.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_SOURCE_SHA;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$WT"
B16=$(cat "$BASE/XP_A13_B16_PROTOCOL_LATEST_RUN.txt")
case "$B16" in "$BASE"/xp_a13_b16_protocol_*) ;; *) echo STOP_B16_SOURCE;exit 8;; esac
while [ ! -f "$B16/XP_A13_B16_PROTOCOL_COMPLETE.json" ]; do
 SESSION=$(basename "$B16" | sed 's/^xp_a13_b16_protocol_/prdiff_a13_b16_/')
 if ! tmux has-session -t "$SESSION" 2>/dev/null;then
  echo STOP_B16_PROTOCOL_REPRODUCTION_MISSING;exit 8
 fi
 sleep 10
done
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SCORE="$BASE/xp_a13_f8_score_$STAMP"
CLIP="$BASE/xp_a13_f8_clip_$STAMP"
COMP="$BASE/xp_a13_f8_compare_$STAMP"
PROT="$BASE/xp_a13_f8_protocol_$STAMP"
echo '===== F8 CPU 520 HISTORICAL + 920 NEW ROOTS ====='
"$PY" scripts/track_r/xp_a13_f8_score_cpu.py \
 --expected-head "$HEAD" --out "$SCORE"
echo '===== F8 CPU FROZEN FACE CLIP ====='
"$PY" scripts/track_r/xp_a13_f8_faceclip_cpu.py \
 --expected-head "$HEAD" --score-dir "$SCORE" --out "$CLIP"
echo '===== F8 V1 PRIMARY V3 SECONDARY ====='
"$PY" scripts/track_r/xp_a13_f8_v1_v3_compare_cpu.py \
 --expected-head "$HEAD" --score-dir "$SCORE" --clip-dir "$CLIP" \
 --out "$COMP"
echo '===== F8 FROZEN GT-FREE PROTOCOL V1 + H1/H2/H3 ====='
"$PY" scripts/track_r/xp_a13_f8_protocol_cpu.py \
 --expected-head "$HEAD" --score-dir "$SCORE" --clip-dir "$CLIP" \
 --b16-protocol "$B16" --out "$PROT"
printf '%s\n' "$SCORE" > "$BASE/XP_A13_F8_SCORE_LATEST.txt"
printf '%s\n' "$CLIP" > "$BASE/XP_A13_F8_CLIP_LATEST.txt"
printf '%s\n' "$COMP" > "$BASE/XP_A13_F8_COMPARE_LATEST.txt"
printf '%s\n' "$PROT" > "$BASE/XP_A13_F8_PROTOCOL_LATEST.txt"
echo "A13_CP1_B16_PROTOCOL=$B16"
echo "A13_CP1_F8_SCORE=$SCORE"
echo "A13_CP1_F8_CLIP=$CLIP"
echo "A13_CP1_F8_FORECAST_COMPARE=$COMP"
echo "A13_CP1_F8_PROTOCOL=$PROT"
echo 'A13_CP1_F8_CPU_ANALYSIS_COMPLETE'
