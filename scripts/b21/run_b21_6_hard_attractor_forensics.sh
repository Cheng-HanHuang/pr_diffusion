#!/usr/bin/env bash
set -euo pipefail

# B21.6 no/low-GPU hard-image bad-attractor forensics.
# Uses existing PNG/CSV outputs only.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_6_hard_attractor_forensics}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/hard_attractor_forensics.py \
  --b19_base "$B19_BASE" \
  --targets "00046,00171,00480,00746,00971" \
  --bad_psnr_threshold 25.0 \
  --max_per_image 120 \
  --image_size 128 \
  --lf_radius_frac 0.12 \
  --outdir "$OUT" \
  --report_path "$REPO/docs/b21/b21_6_hard_attractor_forensics.md"

echo "B21.6 report: $REPO/docs/b21/b21_6_hard_attractor_forensics.md"
echo "B21.6 artifacts: $OUT"
