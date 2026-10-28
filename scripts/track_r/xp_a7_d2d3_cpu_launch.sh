#!/usr/bin/env bash
# CPU-only K<=4 ImageNet metrics for D2 DAPS4K and D3 SITCOM.
# Gate: 3 policy tests, exact pinned 400/400 receipt audit for each stage,
# and a real single-image CPU LPIPS/SSIM/PSNR smoke per family.
# Isolated from the running A6 original worktree. No GPU, no GT for >K4.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
A7WT="$ROOT/pr_diffusion_track_r_a7_cpu"
BASE="$ROOT/outputs/pr_diffusion/track_r"
DEPS="$BASE/a7_cpu_dependencies"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
PINNED=b7af2f7135beb6e8367d83f028e0fbaf163bc7bd
if [ -z "$HEAD" ] ||
   [ "$(git -C "$A7WT" rev-parse HEAD)" != "$HEAD" ] ||
   [ -n "$(git -C "$A7WT" status --porcelain)" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$PINNED" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ] ||
   [ ! -f "$DEPS/piq/__init__.py" ]; then
   echo "ACTION NEEDED: A7_CPU_K4_PREFLIGHT_REPO_OR_PACKAGE"
   exit 8
fi
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEPS:$A7WT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "=== VERIFY K4 POLICY REGRESSIONS ==="
"$PY" -m unittest discover -s "$A7WT/tests" \
  -p 'test_track_r_xp_a7_d2d3_score.py' -q

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
D2OUT="$BASE/xp_a7_d2d3_$STAMP""_D2"
D3OUT="$BASE/xp_a7_d2d3_$STAMP""_D3"
LOG="$BASE/xp_a7_d2d3_$STAMP""_cpu.log"

test ! -e "$D2OUT"
test ! -e "$D3OUT"
test ! -e "$LOG"

echo "=== HASH ALL D2 RECEIPTS; SMOKE ONE COMPLETE D2 METRIC TRIPLE ==="
"$PY" -u "$A7WT/scripts/track_r/xp_a7_d2d3_score_cpu.py" \
  --stage D2 --expected-head "$HEAD" --out "$D2OUT" --preflight-only

echo "=== HASH ALL D3 RECEIPTS; SMOKE ONE COMPLETE D3 METRIC TRIPLE ==="
"$PY" -u "$A7WT/scripts/track_r/xp_a7_d2d3_score_cpu.py" \
  --stage D3 --expected-head "$HEAD" --out "$D3OUT" --preflight-only

echo "=== LAUNCH DETACHED CPU-ONLY K4 D2 THEN D3 ==="
nohup bash -c '
   set -euo pipefail
   PY="$1"; SCRIPT="$2"; HEAD="$3"; D2="$4"; D3="$5"
   echo "A7_D2_SCORE_START"
   "$PY" -u "$SCRIPT" --stage D2 --expected-head "$HEAD" --out "$D2"
   echo "A7_D2_SCORE_RC=0"
   echo "A7_D3_SCORE_START"
   "$PY" -u "$SCRIPT" --stage D3 --expected-head "$HEAD" --out "$D3"
   echo "A7_D3_SCORE_RC=0"
   echo "A7_D2D3_K4_CPU_SCORE_PASS"
' bash "$PY" "$A7WT/scripts/track_r/xp_a7_d2d3_score_cpu.py" \
   "$HEAD" "$D2OUT" "$D3OUT" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "A7_D2D3_CPU_PID=$PID"
echo "A7_D2D3_LOG=$LOG"
echo "A7_D2_AGGREGATE=$D2OUT/XP_A7_K4_AGGREGATE.json"
echo "A7_D3_AGGREGATE=$D3OUT/XP_A7_K4_AGGREGATE.json"
echo "A7_SCORING_IS_CPU_ONLY=true"
echo "A7_D2D3_CPU_SUBMITTED"
