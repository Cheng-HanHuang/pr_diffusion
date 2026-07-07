#!/usr/bin/env bash
set -euo pipefail

# B21.2 prerequisite.  This does not launch GPU work; it audits whether old
# replay/output CSVs expose the sample image paths needed for prior-score replay.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_2_candidate_path_discovery}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/discover_candidate_sample_paths.py \
  --b19_base "$B19_BASE" \
  --targets "00046,00171,00480,00746,00971,00136,00154,00253" \
  --outdir "$OUT" \
  --report_path "$REPO/docs/b21/b21_2_candidate_path_discovery.md"

echo "B21.2 path-discovery report: $REPO/docs/b21/b21_2_candidate_path_discovery.md"
echo "B21.2 path-discovery artifacts: $OUT"
