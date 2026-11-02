#!/usr/bin/env bash
# A19 round15 safe, work-conserving CPU freeze/readiness bootstrap.
# DOES NOT launch XP-C3 or XP-I2 GPU until prereg source and replay
# gates are registered. Never score prior candidates before code freeze.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 PINNED_A24_HEAD" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
WT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"
echo "=== A19 ROUND15 CODE QA ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_round15.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_xpi2_prep.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_i2_native.py' -v
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/XP_A24_A19_READINESS_$STAMP.json"
echo "=== INVENTORY SOURCE/FILE/CODE READINESS, CPU ONLY ==="
"$PY" scripts/track_r/xp_a24_a19_readiness_cpu.py \
  --expected-head "$HEAD" --out "$OUT"
echo "=== FREEZE ORIGINAL D1 XP-I2 IMAGE-NET NATIVE BAGS (NO GPU) ==="
PREP="$BASE/xp_a24_xpi2_prepare_$STAMP"
if "$PY" scripts/track_r/xp_a24_xpi2_prepare_cpu.py \
 --expected-head "$HEAD" --out "$PREP" --preflight-only; then
 "$PY" scripts/track_r/xp_a24_xpi2_prepare_cpu.py \
  --expected-head "$HEAD" --out "$PREP"
 echo "A24_XPI2_ORIGINAL_D1_Y_FROZEN=$PREP"
else
 echo "A24_XPI2_CPU_FREEZE_PENDING_PINNED_SOURCE_OR_D1_Y"
fi
echo "A24_READINESS=$OUT"
echo "A24_GPU_LAUNCHED=0"
echo "A24_GPU_RULE=DIRTY_ALLOWED_AFTER_SHA_REPLAY_AND_80PCT_ADMISSION"
echo "A24_XPC3_CODE_FREEZE_STILL_REQUIRED_BEFORE_ANY_CANDIDATE_SCORE"
echo "A24_ROUND15_CPU_BOOTSTRAP_COMPLETE"
