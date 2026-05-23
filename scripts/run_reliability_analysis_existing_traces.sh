#!/usr/bin/env bash
set -euo pipefail

# Run adaptive-compute simulation and selector calibration from completed
# multi-lambda diagnostic traces.
#
# Usage from repo root on PAC:
#   bash scripts/run_reliability_analysis_existing_traces.sh
#
# Optional:
#   BATCH_ROOT=/path/to/np_ffhq_multilambda_validation_batch \
#   OUTDIR=/path/to/reliability_analysis \
#   bash scripts/run_reliability_analysis_existing_traces.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
BATCH_ROOT=${BATCH_ROOT:-$ROOT/np_ffhq_multilambda_validation_batch}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTDIR=${OUTDIR:-$BATCH_ROOT/reliability_analysis_$STAMP}

SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}
PSNR_KEY=${PSNR_KEY:-raw_psnr}
THRESHOLDS=${THRESHOLDS:-25,28,30}
PRIMARY_THRESHOLD=${PRIMARY_THRESHOLD:-25}
N_SEED_ORDERS=${N_SEED_ORDERS:-200}
ADAPTIVE_START_K=${ADAPTIVE_START_K:-2}
ADAPTIVE_ADD_K=${ADAPTIVE_ADD_K:-2}
ADAPTIVE_MAX_K=${ADAPTIVE_MAX_K:-10}
HARD_IMAGES=${HARD_IMAGES:-00028,00005,00013,00034,00027,00007,00000}

cd "$REPO"
mkdir -p "$OUTDIR"

python scripts/analyze_reliability_from_traces.py \
  --roots_or_traces "$BATCH_ROOT" \
  --outdir "$OUTDIR" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key "$PSNR_KEY" \
  --thresholds "$THRESHOLDS" \
  --primary_threshold "$PRIMARY_THRESHOLD" \
  --hard_images "$HARD_IMAGES" \
  --n_seed_orders "$N_SEED_ORDERS" \
  --adaptive_start_k "$ADAPTIVE_START_K" \
  --adaptive_add_k "$ADAPTIVE_ADD_K" \
  --adaptive_max_k "$ADAPTIVE_MAX_K" \
  --dedupe

cat <<EOF

Reliability analysis complete.

Output directory:
  $OUTDIR

Main files:
  $OUTDIR/adaptive_policy_summary.csv
  $OUTDIR/adaptive_policy_image_level.csv
  $OUTDIR/selector_calibration_global.csv
  $OUTDIR/selector_calibration_by_image.csv
  $OUTDIR/candidate_availability_by_image.csv
  $OUTDIR/hard_image_candidate_availability.csv

Suggested quick checks:
  column -s, -t < $OUTDIR/selector_calibration_global.csv | less -S
  column -s, -t < $OUTDIR/hard_image_candidate_availability.csv | less -S
  grep mean_over_orders $OUTDIR/adaptive_policy_summary.csv | column -s, -t | less -S
EOF
