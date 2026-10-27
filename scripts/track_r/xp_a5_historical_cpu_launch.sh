#!/usr/bin/env bash
# A5 immediate CPU-only historical block: A8b exact residual strings,
# frozen source/family tau, then targeted A4.4 historical DAPS config probe.
# Does not read GT images, run GPU, inspect new XP outcomes or alter S1/S2/S3.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

if [ -z "$HEAD" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$HEAD" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "ACTION NEEDED: A5_CPU_WRONG_OR_DIRTY_CHECKOUT"
  exit 8
fi
echo "=== HISTORICAL ONLY A8b TESTS ==="
"$PY" -m unittest discover -s "$WT/tests" \
  -p 'test_track_r_xp_r0n2_a8b.py' -q
echo "=== A8b FROZEN RUN-ORDERED RHO AND SOURCE/FAMILY TAU ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" "$WT/scripts/track_r/xp_r0n2_a8b_cpu.py" \
  --expected-head "$HEAD" --out "$BASE/r0n2_a8b_$STAMP"
echo "=== A4.4 HISTORICAL CONFIG PROVENANCE PROBE ==="
"$PY" "$WT/scripts/track_r/xp_a44_daps_metadata_probe_cpu.py" \
  --expected-head "$HEAD" --out "$BASE/r0n2_a44_$STAMP" ||
  echo "ACTION NEEDED: A44_CONFIG_PROBE_REVIEW; A8b output is preserved"
echo "=== END A5 CPU BLOCK ==="
echo "A8B_DIR=$BASE/r0n2_a8b_$STAMP"
echo "A44_DIR=$BASE/r0n2_a44_$STAMP"
echo "GPU_LAUNCHED=false"
echo "XP_A16_OUTCOME_EMBARGO=ACTIVE"
