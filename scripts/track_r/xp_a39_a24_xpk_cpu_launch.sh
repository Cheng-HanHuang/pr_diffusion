#!/usr/bin/env bash
# A39: A24 XP-K CPU-only native gate preparation. NO GPU allocation,
# no historical identity replays and no calibration reconstruction runs.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/track_r/xp_a39_a24_xpk_cpu_launch.sh <exact-A39-commit>" >&2
  exit 8
fi

HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/daps/bin/python"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a39_xpk_cpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$WT"
echo "=== A24 source pin and strictly CPU-only regression ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test -f scripts/track_r/xp_a39_a24_xpk_cpu_preflight.py
test -f scripts/track_r/xp_a32_a23_config_diff_cpu.py
test -f docs/track_r/executors/xp/returns/XP_F8_LEGACY_DAPS_8ROOT_REPLAY_EXACT_PASS_20261029.md

bash -n scripts/track_r/xp_a39_a24_xpk_cpu_launch.sh
"$PY" -m unittest discover -s tests \
  -p 'test_track_r_xp_a39_a24_xpk_cpu.py' -v

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a39_xpk_cpu_$STAMP"
echo "=== Read-only native source inventory and one-leaf screens ==="
"$PY" -u scripts/track_r/xp_a39_a24_xpk_cpu_preflight.py \
    --expected-head "$HEAD" --out "$OUT"

echo "A39_XPK_CPU_RUN=$OUT"
echo "A39_NO_GPU_JOBS_LAUNCHED"
echo "A39_NATIVE_BINDING_AND_IDENTITY_REPLAY_PENDING"
echo "A39_DAPS_F8_PREVIOUS_REPLAY_DOCUMENTED"
