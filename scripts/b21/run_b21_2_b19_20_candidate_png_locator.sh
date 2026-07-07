#!/usr/bin/env bash
set -euo pipefail

# B21.2 no-GPU prerequisite: strict search for B19.20 candidate PNGs outside
# the replay CSVs.  The broad B20/B21 result tree contains many unrelated
# meas5001 PNGs, so this wrapper keeps only sample PNGs whose path looks like
# B19_20 / ffhq100 / runseed4400.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_2_b19_20_candidate_png_locator_strict}

ROOTS=${ROOTS:-/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps/results,/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
TARGETS=${TARGETS:-00046,00136,00154,00171,00253,00480,00746,00971}
REQUIRE_ANY=${REQUIRE_ANY:-B19_20,b19_20,ffhq100,runseed4400}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/locate_b19_20_candidate_pngs.py \
  --targets "$TARGETS" \
  --roots "$ROOTS" \
  --require_any "$REQUIRE_ANY" \
  --sample_only \
  --maybe_b19_20_only \
  --outdir "$OUT" \
  --report_path "$REPO/docs/b21/b21_2_b19_20_candidate_png_locator.md"

echo "B21.2 B19.20 PNG locator report: $REPO/docs/b21/b21_2_b19_20_candidate_png_locator.md"
echo "B21.2 B19.20 PNG locator artifacts: $OUT"
