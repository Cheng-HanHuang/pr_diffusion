#!/usr/bin/env bash
set -euo pipefail

# B21.2 selector-v2 first pass: clean-free rot180-aware measurement scoring.
# No GPU required.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
INPUT=${INPUT:-$B21_BASE/B21_2_candidate_recovery/candidate_recovery_rows.csv}
OUT=${OUT:-$B21_BASE/B21_2_symmetry_selector}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/score_b21_2_symmetry_selector_v2.py \
  --input_csv "$INPUT" \
  --b19_base "$B19_BASE" \
  --outdir "$OUT" \
  --report_path "$REPO/docs/b21/b21_2_symmetry_selector.md"

echo "B21.2 symmetry selector report: $REPO/docs/b21/b21_2_symmetry_selector.md"
echo "B21.2 symmetry selector artifacts: $OUT"
