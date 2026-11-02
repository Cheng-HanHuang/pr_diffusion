#!/usr/bin/env bash
# A19 Nov 4 two independent CPU components, detached after full provenance QA.
# Quality-only face-CLIP and GT figure diagnostics; leaves XP-I2 GPU alone.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 PINNED_HEAD" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"

for p in test_track_r_xp_a24_nov4_fast.py test_track_r_xp_a24_nov4_d4.py; do
  "$PY" -m unittest discover -s tests -p "$p" -v
done

# Independent frozen-source preflights. A failure launches no CPU production.
"$PY" scripts/track_r/xp_a24_nov4_fast_cpu.py \
  --expected-head "$HEAD" --out "$BASE/unused_a24_fast_preflight" \
  --preflight-only
"$PY" scripts/track_r/xp_a24_nov4_d4_panels_cpu.py \
  --expected-head "$HEAD" --out "$BASE/unused_a24_d4_preflight" \
  --preflight-only

if compgen -G "$BASE/xp_a24_nov4_fast_*" >/dev/null ||
   compgen -G "$BASE/xp_a24_nov4_d4_*" >/dev/null; then
  echo "ACTION NEEDED: EXISTING_NOV4_RUN_REQUIRES_READ_ONLY_RECONCILIATION" >&2
  exit 8
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FAST="$BASE/xp_a24_nov4_fast_$STAMP"
D4="$BASE/xp_a24_nov4_d4_$STAMP"
FASTLOG="$BASE/XP_A24_NOV4_FAST_$STAMP.log"
D4LOG="$BASE/XP_A24_NOV4_D4_$STAMP.log"

nohup bash -c '
set -euo pipefail
PY="$1"; WT="$2"; HEAD="$3"; RUN="$4"
cd "$WT"
"$PY" -u scripts/track_r/xp_a24_nov4_fast_cpu.py \
  --expected-head "$HEAD" --out "$RUN" --panels-only
"$PY" -u scripts/track_r/xp_a24_nov4_fast_cpu.py \
  --expected-head "$HEAD" --out "$RUN" --resume
' bash "$PY" "$WT" "$HEAD" "$FAST" >"$FASTLOG" 2>&1 </dev/null &
FASTPID=$!

nohup "$PY" -u scripts/track_r/xp_a24_nov4_d4_panels_cpu.py \
   --expected-head "$HEAD" --out "$D4" >"$D4LOG" 2>&1 </dev/null &
D4PID=$!

echo "A24_NOV4_FAST_CPU_PID=$FASTPID"
echo "A24_NOV4_D4_CPU_PID=$D4PID"
echo "A24_NOV4_FAST_RUN=$FAST"
echo "A24_NOV4_D4_RUN=$D4"
echo "A24_NOV4_FAST_LOG=$FASTLOG"
echo "A24_NOV4_D4_LOG=$D4LOG"
echo "A24_NOV4_NO_GPU_REQUIRED"
