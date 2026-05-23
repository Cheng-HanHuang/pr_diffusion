#!/usr/bin/env bash
set -euo pipefail

# Optional 00013 margin-recovery experiment.
#
# Run this only if the offline selector-policy sweep shows that selector
# calibration is still insufficient or that 00013 needs more candidate margin.
#
# Motivation:
#   In the remaining-hard grid, 00013 had >28 dB candidates but relatively thin
#   candidate-level success.  This focused run asks whether stronger/later
#   high-lambda branches and more candidate diversity can raise the margin.
#
# Usage from repo root on PAC:
#   bash scripts/launch_ffhq_00013_margin_recovery.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

HARD_IMAGES=${HARD_IMAGES:-00013}
SEEDS=${SEEDS:-124,125,126,127,128,129,130,131}
LAMBDAS=${LAMBDAS:-"0.05 0.08 0.1 0.12 0.15 0.18 0.2"}
PROJ_STARTS=${PROJ_STARTS:-"350 400 450"}
SOFT_VALUES=${SOFT_VALUES:-"5 8"}
HARD_VALUES=${HARD_VALUES:-"1"}

GPU_LIST=${GPU_LIST:-0,1,2,3}
MAX_PARALLEL=${MAX_PARALLEL:-4}
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_00013_margin_recovery_$STAMP}

cd "$REPO"

cat <<EOF
[00013-margin-recovery]
  OUTROOT     = $OUTROOT
  HARD_IMAGES = $HARD_IMAGES
  SEEDS       = $SEEDS
  LAMBDAS     = $LAMBDAS
  PROJ_STARTS = $PROJ_STARTS
  SOFT_VALUES = $SOFT_VALUES
  HARD_VALUES = $HARD_VALUES
  GPU_LIST    = $GPU_LIST
EOF

HARD_IMAGES="$HARD_IMAGES" \
SEEDS="$SEEDS" \
LAMBDAS="$LAMBDAS" \
PROJ_STARTS="$PROJ_STARTS" \
SOFT_VALUES="$SOFT_VALUES" \
HARD_VALUES="$HARD_VALUES" \
GPU_LIST="$GPU_LIST" \
MAX_PARALLEL="$MAX_PARALLEL" \
OUTROOT="$OUTROOT" \
bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

cat <<EOF

00013 margin-recovery experiment complete.

Output root:
  $OUTROOT

Key files:
  $OUTROOT/hard_ablation_diag_run_trace_summary.csv
  $OUTROOT/reliability_analysis/hard_image_candidate_availability.csv
  $OUTROOT/hard_ablation_post_winner_lf_mse_mean_selected_summary.csv
EOF
