#!/usr/bin/env bash
set -euo pipefail

# B21.0 read-only measurement-integrity audit.
# Run from any shell on PAC. Does not launch reconstruction jobs.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_0_measurement_integrity_audit}

cd "$REPO"
mkdir -p "$OUT"

# Include both direct B19_20 CSVs under $B19_BASE and nested variants.
# The first version of this wrapper only searched nested B19_20*/**/*.csv paths,
# but the current PAC outputs place the B19_20 CSVs directly under $B19_BASE.
python scripts/b21/audit_measurement_integrity.py \
  --measurement_roots "$B19_BASE/measurements" \
  --measurement_patterns "ffhq*_phase_noise*_meas*.pt,**/ffhq*_phase_noise*_meas*.pt" \
  --seeds "5001-5010" \
  --expected_images 100 \
  --case_csv_globs "$B19_BASE/B19_20*.csv,$B19_BASE/B19_20*/**/*.csv,$B19_BASE/**/B19_20*/*.csv" \
  --runner_search_roots "$REPO/scripts,$REPO/docs/b19" \
  --outdir "$OUT" \
  --report_path "$REPO/docs/b21/b21_0_measurement_integrity_audit.md"

echo "B21.0 report: $REPO/docs/b21/b21_0_measurement_integrity_audit.md"
echo "B21.0 artifacts: $OUT"
