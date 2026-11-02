#!/usr/bin/env bash
# A18 all100 / K64 CPU-only analysis. GPU generation already complete.
# Fail closed on receipt, forecast or source mismatch. Do not rerun GPU.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 EXPECTED_A23_HEAD" >&2;exit 8; }
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
echo "=== A23 FROZEN MODEL EVALUATION UNIT TESTS ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a23_cpn_part2.py' -v
bash -n scripts/track_r/xp_a23_launch_part2.sh \
 scripts/track_r/xp_a23_cpn_part2_publish.sh
echo "=== A23 CHECK PINNED A22 FINAL 90/90 AND V5 REGISTERED BYTES ==="
"$PY" scripts/track_r/xp_a23_cpn_part2_cpu.py \
 --expected-head "$HEAD" \
 --out "$BASE/xp_a23_cpn_part2_unused_preflight" --preflight-only
echo "=== SUBMIT A23 CPU OFFLINE FULL K64 CP-N ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="prdiff_a23_cpn_part2_$STAMP"
LOG="$BASE/XP_A23_CPN_PART2_$STAMP.log"
test ! -e "$LOG"
tmux new-session -d -s "$SESSION" -c "$WT" \
 "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '$WT/scripts/track_r/xp_a23_cpn_part2_publish.sh' '$HEAD' > '$LOG' 2>&1"
echo "A23_CPN_PART2_TMUX=$SESSION"
echo "A23_CPN_PART2_LOG=$LOG"
echo "A23_CPN_PART2_CPU_ONLY_SUBMITTED"
