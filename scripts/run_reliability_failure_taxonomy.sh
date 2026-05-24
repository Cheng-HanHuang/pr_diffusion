#!/usr/bin/env bash
set -euo pipefail

# Analyze class-level reliability failure modes from a trace root or CSV.
#
# Usage:
#   TRACE_ROOT=/path/to/selector_validation_pool_or_trace.csv \
#   bash scripts/run_reliability_failure_taxonomy.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
TRACE_ROOT=${TRACE_ROOT:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTDIR=${OUTDIR:-}

cd "$REPO"

if [[ -z "$TRACE_ROOT" ]]; then
  TRACE_ROOT=$(ls -td "$ROOT"/np_ffhq_selector_validation_pool_* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$TRACE_ROOT" ]]; then
  echo "Could not infer TRACE_ROOT. Pass TRACE_ROOT=/path/to/root or CSV." >&2
  exit 1
fi

if [[ -z "$OUTDIR" ]]; then
  if [[ -f "$TRACE_ROOT" ]]; then
    OUTDIR="$(dirname "$TRACE_ROOT")/reliability_failure_taxonomy_$STAMP"
  else
    OUTDIR="$TRACE_ROOT/reliability_failure_taxonomy_$STAMP"
  fi
fi
mkdir -p "$OUTDIR"

python scripts/analyze_reliability_failure_taxonomy.py \
  --roots_or_traces "$TRACE_ROOT" \
  --outdir "$OUTDIR" \
  --thresholds 25,28,30 \
  --selector_stat post_winner_lf_mse_mean \
  --psnr_key raw_psnr

cat <<EOF

Reliability failure taxonomy complete.

Output directory:
  $OUTDIR

Files:
  $OUTDIR/reliability_failure_taxonomy_by_image.csv
  $OUTDIR/reliability_failure_taxonomy_counts.csv
EOF
