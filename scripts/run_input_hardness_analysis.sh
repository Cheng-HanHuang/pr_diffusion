#!/usr/bin/env bash
set -euo pipefail

# Compute pre-reconstruction input features and join them with reliability labels.
# This is intended for class-level analysis of hard images, not for tuning a
# single image.
#
# Usage:
#   TRACE_CSV=/path/to/selector_validation_diag_run_trace_summary.csv \
#   bash scripts/run_input_hardness_analysis.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
TRACE_CSV=${TRACE_CSV:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTDIR=${OUTDIR:-}
THRESHOLD=${THRESHOLD:-28}

cd "$REPO"

if [[ -z "$TRACE_CSV" ]]; then
  latest=$(ls -td "$ROOT"/np_ffhq_selector_validation_pool_* 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest" ]]; then
    TRACE_CSV="$latest/selector_validation_diag_run_trace_summary.csv"
  fi
fi
if [[ -z "$TRACE_CSV" || ! -f "$TRACE_CSV" ]]; then
  echo "Could not find TRACE_CSV. Pass TRACE_CSV=/path/to/selector_validation_diag_run_trace_summary.csv" >&2
  exit 1
fi

if [[ -z "$OUTDIR" ]]; then
  OUTDIR="$(dirname "$TRACE_CSV")/input_hardness_analysis_$STAMP"
fi
mkdir -p "$OUTDIR"

python scripts/analyze_input_hardness_properties.py \
  --trace_csv "$TRACE_CSV" \
  --data_root "$DATA_ROOT" \
  --out_csv "$OUTDIR/input_hardness_features_thr${THRESHOLD}.csv" \
  --threshold "$THRESHOLD" \
  --psnr_key raw_psnr \
  --selector_stat post_winner_lf_mse_mean

cat <<EOF

Input hardness analysis complete.

Output:
  $OUTDIR/input_hardness_features_thr${THRESHOLD}.csv

Recommended quick check:
  column -s, -t < $OUTDIR/input_hardness_features_thr${THRESHOLD}.csv | less -S
EOF
