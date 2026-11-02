#!/usr/bin/env bash
# Detached A19 Nov4 CPU-only FAST work: anonymized 7 cases + A9.2d 11,520.
# No GPU, no new selector, does not touch running XP-I2 supervisor.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 PINNED_HEAD" >&2; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_nov4_fast.py' -v
"$PY" scripts/track_r/xp_a24_nov4_fast_cpu.py \
    --expected-head "$HEAD" --out "$BASE/.nonexistent-fast-preflight" \
    --preflight-only

if compgen -G "$BASE/xp_a24_nov4_fast_*/A24_FAST_FACECLIP_RECEIPT.json" >/dev/null; then
  echo "A24_NOV4_FAST_ALREADY_COMPLETE_NO_DUPLICATE"
  exit 0
fi
if compgen -G "$BASE/xp_a24_nov4_fast_*/image_chunks" >/dev/null; then
  echo "ACTION NEEDED: EXISTING_NOV4_FAST_PARTIAL_RUN_REQUIRES_RECONCILIATION" >&2
  exit 8
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a24_nov4_fast_$STAMP"
LOG="$BASE/XP_A24_NOV4_FAST_$STAMP.log"
# Run inside a separate shell; do not leave a background GPU job in this shell.
nohup bash -c '
set -euo pipefail
PY="$1"; WT="$2"; HEAD="$3"; RUN="$4"
cd "$WT"
"$PY" -u scripts/track_r/xp_a24_nov4_fast_cpu.py \
  --expected-head "$HEAD" --out "$RUN" --panels-only
"$PY" -u scripts/track_r/xp_a24_nov4_fast_cpu.py \
  --expected-head "$HEAD" --out "$RUN" --resume
' bash "$PY" "$WT" "$HEAD" "$RUN" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "A24_NOV4_FAST_DETACHED_PID=$PID"
echo "A24_NOV4_FAST_RUN=$RUN"
echo "A24_NOV4_FAST_LOG=$LOG"
echo "A24_NOV4_FAST_CPU_ONLY_NO_GPU_LAUNCHED"
