#!/usr/bin/env bash
# CPU-only A23 readiness audit; it never allocates a CUDA context.
# Run from a dedicated clean worktree of the exact committed A23 head.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/track_r/xp_a32_a23_launch_readiness.sh <exact-commit-sha>" >&2
  exit 8
fi

HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$WT"

echo "=== Exact A23 code pin / no uncommitted edits ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test "$(git rev-parse --is-inside-work-tree)" = true
test -f "$WT/scripts/track_r/xp_a32_a23_readiness_cpu.py"
test -f "$WT/docs/track_r/executors/xp/XP_A23_CP4_SUPPLEMENT_DRAFT_PENDING.json"
echo "A23_CPU_PIN_OK=$HEAD"

echo "=== Published source refs remain separate and unchanged ==="
G7="$ROOT/pr_diffusion_track_r_a22_g7_pub_20261104T035034Z"
TAB3="$ROOT/pr_diffusion_track_r_a22_tab3_pub_20261104T040346Z"
test "$(git -C "$G7" rev-parse HEAD)" = \
  9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1
test "$(git -C "$TAB3" rev-parse HEAD)" = \
  df6e8cd5c62f506343f2ff8fe6d493c4d032ea68
test -z "$(git -C "$G7" status --porcelain)"
test -z "$(git -C "$TAB3" status --porcelain)"
echo "A23_PUBLISHED_INPUT_TREES_PINNED"

echo "=== CPU-only A23 tests; no CUDA allocations ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a32_a23_*.py' -v

echo "=== Exact frozen GT/diagnostic source provenance ==="
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BASE/xp_a32_a23_readiness_$STAMP"
"$PY" -u scripts/track_r/xp_a32_a23_readiness_cpu.py \
  --expected-head "$HEAD" --out "$OUT"

echo "=== Completed safe readiness ==="
echo "A23_READINESS_ROOT=$OUT"
echo "A23_READINESS_RECEIPT=$OUT/A23_CPU_READINESS_RECEIPT.json"
echo "A23_NO_GPU_JOB_LAUNCHED"
echo "A23_SCIENTIFIC_ADAPTERS_PENDING"
echo "A23_NEXT_CPU_ENGINEERING_VIA_SCHEDULED_AUTOPILOT"
