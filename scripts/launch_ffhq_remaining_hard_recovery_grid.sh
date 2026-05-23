#!/usr/bin/env bash
set -euo pipefail

# Focused recovery grid for the remaining true hard FFHQ phase-retrieval cases.
#
# Motivation after hard-image steps 0/1/2:
#   - 25 dB reliability is plausible with the expanded LF/S2 pool.
#   - 28 dB reliability is still blocked by candidate-generation failures on
#     00013, 00027, and 00028.
#   - Selector improvements alone cannot fix pools where the oracle also fails.
#
# This launcher therefore spends compute on the remaining hard cases, with a
# denser lambda/projection-start grid and more seeds.  It uses all four GPUs by
# default through scripts/launch_ffhq_hard_image_reliability_ablation.sh.
#
# Default experiment:
#   images      = 00013,00027,00028
#   seeds       = 116,117,118,119,120,121,122,123
#   lambdas     = 0.03,0.05,0.08,0.10,0.12,0.15
#   proj_start  = 300,350,400
#   soft/hard   = 5/1
#
# Usage from repo root on PAC:
#   bash scripts/launch_ffhq_remaining_hard_recovery_grid.sh
#
# Smaller smoke test:
#   SEEDS=116,117 LAMBDAS="0.08 0.1 0.12" PROJ_STARTS="350" \
#   bash scripts/launch_ffhq_remaining_hard_recovery_grid.sh
#
# Optional diversity follow-up:
#   RUN_DIVERSITY=1 bash scripts/launch_ffhq_remaining_hard_recovery_grid.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

# Remaining hard cases for >=28 dB reliability.
HARD_IMAGES=${HARD_IMAGES:-00013,00027,00028}

# Main recovery grid.
SEEDS=${SEEDS:-116,117,118,119,120,121,122,123}
LAMBDAS=${LAMBDAS:-"0.03 0.05 0.08 0.1 0.12 0.15"}
PROJ_STARTS=${PROJ_STARTS:-"300 350 400"}
SOFT_VALUES=${SOFT_VALUES:-"5"}
HARD_VALUES=${HARD_VALUES:-"1"}

# Shared reconstruction settings.
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
OVERSAMPLE=${OVERSAMPLE:-2}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MAX_PARALLEL=${MAX_PARALLEL:-4}
SKIP_EXISTING=${SKIP_EXISTING:-1}

# Output root for main grid.
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_remaining_hard_recovery_grid_$STAMP}

cd "$REPO"
mkdir -p "$OUTROOT"

cat <<EOF
[remaining-hard-recovery] main grid
  OUTROOT      = $OUTROOT
  HARD_IMAGES  = $HARD_IMAGES
  SEEDS        = $SEEDS
  LAMBDAS      = $LAMBDAS
  PROJ_STARTS  = $PROJ_STARTS
  SOFT_VALUES  = $SOFT_VALUES
  HARD_VALUES  = $HARD_VALUES
  GPU_LIST     = $GPU_LIST
  MAX_PARALLEL = $MAX_PARALLEL
EOF

HARD_IMAGES="$HARD_IMAGES" \
SEEDS="$SEEDS" \
LAMBDAS="$LAMBDAS" \
PROJ_STARTS="$PROJ_STARTS" \
SOFT_VALUES="$SOFT_VALUES" \
HARD_VALUES="$HARD_VALUES" \
SCORE_RADIUS="$SCORE_RADIUS" \
PROJ_RADIUS="$PROJ_RADIUS" \
SIGMA="$SIGMA" \
NP_STEPS="$NP_STEPS" \
OVERSAMPLE="$OVERSAMPLE" \
GPU_LIST="$GPU_LIST" \
MAX_PARALLEL="$MAX_PARALLEL" \
SKIP_EXISTING="$SKIP_EXISTING" \
OUTROOT="$OUTROOT" \
bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

# Optional second grid: increase candidate diversity after the main lambda/proj
# sweep.  This is disabled by default because it is more expensive and should be
# run only if the main grid still has oracle failures.
RUN_DIVERSITY=${RUN_DIVERSITY:-0}
if [[ "$RUN_DIVERSITY" == "1" ]]; then
  DIVERSITY_OUTROOT=${DIVERSITY_OUTROOT:-$ROOT/np_ffhq_remaining_hard_diversity_grid_$STAMP}
  DIVERSITY_LAMBDAS=${DIVERSITY_LAMBDAS:-"0.05 0.08 0.1 0.12"}
  DIVERSITY_PROJ_STARTS=${DIVERSITY_PROJ_STARTS:-"350"}
  DIVERSITY_SOFT_VALUES=${DIVERSITY_SOFT_VALUES:-"8"}
  DIVERSITY_HARD_VALUES=${DIVERSITY_HARD_VALUES:-"1 2"}

  cat <<EOF

[remaining-hard-recovery] optional diversity grid
  DIVERSITY_OUTROOT     = $DIVERSITY_OUTROOT
  DIVERSITY_LAMBDAS     = $DIVERSITY_LAMBDAS
  DIVERSITY_PROJ_STARTS = $DIVERSITY_PROJ_STARTS
  DIVERSITY_SOFT_VALUES = $DIVERSITY_SOFT_VALUES
  DIVERSITY_HARD_VALUES = $DIVERSITY_HARD_VALUES
EOF

  HARD_IMAGES="$HARD_IMAGES" \
  SEEDS="$SEEDS" \
  LAMBDAS="$DIVERSITY_LAMBDAS" \
  PROJ_STARTS="$DIVERSITY_PROJ_STARTS" \
  SOFT_VALUES="$DIVERSITY_SOFT_VALUES" \
  HARD_VALUES="$DIVERSITY_HARD_VALUES" \
  SCORE_RADIUS="$SCORE_RADIUS" \
  PROJ_RADIUS="$PROJ_RADIUS" \
  SIGMA="$SIGMA" \
  NP_STEPS="$NP_STEPS" \
  OVERSAMPLE="$OVERSAMPLE" \
  GPU_LIST="$GPU_LIST" \
  MAX_PARALLEL="$MAX_PARALLEL" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  OUTROOT="$DIVERSITY_OUTROOT" \
  bash scripts/launch_ffhq_hard_image_reliability_ablation.sh
fi

cat <<EOF

Remaining-hard recovery experiment complete.

Main root:
  $OUTROOT

Main files:
  $OUTROOT/hard_ablation_diag_run_trace_summary.csv
  $OUTROOT/hard_ablation_post_winner_lf_mse_mean_selected_summary.csv
  $OUTROOT/hard_ablation_post_winner_lf_mse_mean_image_diagnostics.csv
  $OUTROOT/reliability_analysis/hard_image_candidate_availability.csv
  $OUTROOT/reliability_analysis/adaptive_policy_summary.csv
  $OUTROOT/reliability_analysis/selector_calibration_global.csv

Quick checks:
  column -s, -t < $OUTROOT/reliability_analysis/hard_image_candidate_availability.csv | less -S
  grep mean_over_orders $OUTROOT/reliability_analysis/adaptive_policy_summary.csv | column -s, -t | less -S
EOF
