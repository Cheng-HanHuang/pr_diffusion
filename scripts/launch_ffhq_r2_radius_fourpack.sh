#!/usr/bin/env bash
set -euo pipefail

# Run the four R2 high-frequency/low-ceiling sensitivity experiments concurrently.
#
# Do NOT launch four default hard-image ablation scripts manually: each default
# script uses all four GPUs.  This wrapper assigns one experiment to one GPU and
# runs each with MAX_PARALLEL=1.
#
# Experiments:
#   A: PROJ_RADIUS=0.1
#   B: PROJ_RADIUS=0.3
#   C: SCORE_RADIUS=0.4, PROJ_RADIUS=0.2
#   D: SCORE_RADIUS=0.8, PROJ_RADIUS=0.2
#
# Usage from repo root on PAC:
#   bash scripts/launch_ffhq_r2_radius_fourpack.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
PARENT_OUTROOT=${PARENT_OUTROOT:-$ROOT/r2_radius_fourpack_$STAMP}

HARD_IMAGES=${HARD_IMAGES:-00004,00025}
SEEDS_PROJ=${SEEDS_PROJ:-152,153,154,155}
SEEDS_SCORE=${SEEDS_SCORE:-156,157,158,159}
LAMBDAS_PROJ=${LAMBDAS_PROJ:-"0.03 0.05 0.1 0.15"}
PROJ_STARTS_PROJ=${PROJ_STARTS_PROJ:-"250 300 350 400 450"}
LAMBDAS_SCORE=${LAMBDAS_SCORE:-"0.03 0.05 0.1 0.15"}
PROJ_STARTS_SCORE=${PROJ_STARTS_SCORE:-"250 300 350 400"}
SOFT_VALUES=${SOFT_VALUES:-"5"}
HARD_VALUES=${HARD_VALUES:-"1"}

cd "$REPO"
mkdir -p "$PARENT_OUTROOT/logs"

launch_one() {
  local name="$1"
  local gpu="$2"
  shift 2
  local outroot="$PARENT_OUTROOT/$name"
  local log="$PARENT_OUTROOT/logs/$name.log"
  echo "[launch] $name on GPU $gpu -> $outroot"
  (
    set -euo pipefail
    HARD_IMAGES="$HARD_IMAGES" \
    GPU_LIST="$gpu" \
    MAX_PARALLEL=1 \
    OUTROOT="$outroot" \
    "$@"
  ) > "$log" 2>&1 &
}

launch_one proj_radius_0p1 0 env \
  SEEDS="$SEEDS_PROJ" \
  LAMBDAS="$LAMBDAS_PROJ" \
  PROJ_STARTS="$PROJ_STARTS_PROJ" \
  SOFT_VALUES="$SOFT_VALUES" \
  HARD_VALUES="$HARD_VALUES" \
  PROJ_RADIUS=0.1 \
  bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

launch_one proj_radius_0p3 1 env \
  SEEDS="$SEEDS_PROJ" \
  LAMBDAS="$LAMBDAS_PROJ" \
  PROJ_STARTS="$PROJ_STARTS_PROJ" \
  SOFT_VALUES="$SOFT_VALUES" \
  HARD_VALUES="$HARD_VALUES" \
  PROJ_RADIUS=0.3 \
  bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

launch_one score_radius_0p4 2 env \
  SEEDS="$SEEDS_SCORE" \
  LAMBDAS="$LAMBDAS_SCORE" \
  PROJ_STARTS="$PROJ_STARTS_SCORE" \
  SOFT_VALUES="$SOFT_VALUES" \
  HARD_VALUES="$HARD_VALUES" \
  SCORE_RADIUS=0.4 \
  PROJ_RADIUS=0.2 \
  bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

launch_one score_radius_0p8 3 env \
  SEEDS="$SEEDS_SCORE" \
  LAMBDAS="$LAMBDAS_SCORE" \
  PROJ_STARTS="$PROJ_STARTS_SCORE" \
  SOFT_VALUES="$SOFT_VALUES" \
  HARD_VALUES="$HARD_VALUES" \
  SCORE_RADIUS=0.8 \
  PROJ_RADIUS=0.2 \
  bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

wait

cat <<EOF

R2 four-pack experiments complete.

Parent output root:
  $PARENT_OUTROOT

Subruns:
  $PARENT_OUTROOT/proj_radius_0p1
  $PARENT_OUTROOT/proj_radius_0p3
  $PARENT_OUTROOT/score_radius_0p4
  $PARENT_OUTROOT/score_radius_0p8

Logs:
  $PARENT_OUTROOT/logs/*.log

Key files in each subrun:
  hard_ablation_diag_run_trace_summary.csv
  hard_ablation_post_winner_lf_mse_mean_selected_summary.csv
  hard_ablation_post_winner_lf_mse_mean_image_diagnostics.csv
  reliability_analysis/hard_image_candidate_availability.csv
EOF
