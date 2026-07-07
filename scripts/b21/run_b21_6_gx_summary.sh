#!/usr/bin/env bash
set -euo pipefail

# B21.6 GX summary table from already-computed forensics artifacts.
# No GPU work.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
FORENSICS_DIR=${FORENSICS_DIR:-$B21_BASE/B21_6_hard_attractor_forensics}
OUT=${OUT:-$FORENSICS_DIR}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/summarize_b21_6_gx.py \
  --forensics_dir "$FORENSICS_DIR" \
  --outdir "$OUT" \
  --factors "0.4,0.5,0.6" \
  --report_path "$REPO/docs/b21/b21_6_gx_summary.md"

echo "B21.6 GX report: $REPO/docs/b21/b21_6_gx_summary.md"
echo "B21.6 GX artifacts: $OUT"
