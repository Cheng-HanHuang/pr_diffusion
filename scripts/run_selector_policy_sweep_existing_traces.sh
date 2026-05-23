#!/usr/bin/env bash
set -euo pipefail

# Offline selector-policy sweep from existing trace summaries.
#
# This script does not launch GPU reconstructions.  It tests selector variants
# using already generated *_run_trace_summary.csv files.
#
# Usage from repo root on PAC:
#   TRACE_ROOT=/path/to/np_ffhq_remaining_hard_recovery_grid_xxx \
#   bash scripts/run_selector_policy_sweep_existing_traces.sh
#
# You can also pass a direct CSV file:
#   TRACE_ROOT=/path/to/hard_ablation_diag_run_trace_summary.csv \
#   bash scripts/run_selector_policy_sweep_existing_traces.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
TRACE_ROOT=${TRACE_ROOT:-}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTDIR=${OUTDIR:-}

SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}
PSNR_KEY=${PSNR_KEY:-raw_psnr}
LF_RESID_KEY=${LF_RESID_KEY:-raw_noisy_lowfreq_mag_l2}
FULL_RESID_KEY=${FULL_RESID_KEY:-raw_noisy_mag_l2}
THRESHOLDS=${THRESHOLDS:-25,28,30}

cd "$REPO"

if [[ -z "$TRACE_ROOT" ]]; then
  # Prefer newest remaining-hard run if not specified.
  TRACE_ROOT=$(ls -td "$ROOT"/np_ffhq_remaining_hard_recovery_grid_* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$TRACE_ROOT" ]]; then
  echo "Could not infer TRACE_ROOT. Please pass TRACE_ROOT=/path/to/run_root or CSV." >&2
  exit 1
fi

if [[ -z "$OUTDIR" ]]; then
  if [[ -f "$TRACE_ROOT" ]]; then
    OUTDIR="$(dirname "$TRACE_ROOT")/selector_policy_sweep_$STAMP"
  else
    OUTDIR="$TRACE_ROOT/selector_policy_sweep_$STAMP"
  fi
fi

mkdir -p "$OUTDIR"

python scripts/simulate_selector_policy_variants.py \
  --roots_or_traces "$TRACE_ROOT" \
  --outdir "$OUTDIR" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key "$PSNR_KEY" \
  --lf_resid_key "$LF_RESID_KEY" \
  --full_resid_key "$FULL_RESID_KEY" \
  --thresholds "$THRESHOLDS" \
  --dedupe

cat <<EOF

Selector policy sweep complete.

Trace root:
  $TRACE_ROOT

Output directory:
  $OUTDIR

Main files:
  $OUTDIR/selector_policy_summary.csv
  $OUTDIR/selector_policy_by_threshold.csv
  $OUTDIR/selector_policy_image_level.csv
  $OUTDIR/selector_policy_failures.csv

Quick checks:
  column -s, -t < $OUTDIR/selector_policy_summary.csv | less -S
  grep ',28' $OUTDIR/selector_policy_by_threshold.csv | column -s, -t | less -S
EOF
