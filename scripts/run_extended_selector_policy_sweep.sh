#!/usr/bin/env bash
set -euo pipefail

# Extended offline selector-policy sweep for 00013-style seed-ranking failures.
#
# Usage from repo root on PAC:
#   TRACE_ROOT=/path/to/np_ffhq_00013_margin_recovery_<timestamp> \
#   bash scripts/run_extended_selector_policy_sweep.sh
#
# If TRACE_ROOT is omitted, the newest np_ffhq_00013_margin_recovery_* folder is used.

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
  TRACE_ROOT=$(ls -td "$ROOT"/np_ffhq_00013_margin_recovery_* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$TRACE_ROOT" ]]; then
  echo "Could not infer TRACE_ROOT. Pass TRACE_ROOT=/path/to/run_root or CSV." >&2
  exit 1
fi

if [[ -z "$OUTDIR" ]]; then
  if [[ -f "$TRACE_ROOT" ]]; then
    OUTDIR="$(dirname "$TRACE_ROOT")/extended_selector_policy_sweep_$STAMP"
  else
    OUTDIR="$TRACE_ROOT/extended_selector_policy_sweep_$STAMP"
  fi
fi

mkdir -p "$OUTDIR"

python scripts/simulate_selector_policy_variants_extended.py \
  --roots_or_traces "$TRACE_ROOT" \
  --outdir "$OUTDIR" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key "$PSNR_KEY" \
  --lf_resid_key "$LF_RESID_KEY" \
  --full_resid_key "$FULL_RESID_KEY" \
  --thresholds "$THRESHOLDS" \
  --dedupe

cat <<EOF

Extended selector policy sweep complete.

Trace root:
  $TRACE_ROOT

Output directory:
  $OUTDIR

Main files:
  $OUTDIR/extended_selector_policy_summary.csv
  $OUTDIR/extended_selector_policy_by_threshold.csv
  $OUTDIR/extended_selector_policy_image_level.csv
  $OUTDIR/extended_selector_policy_failures.csv
  $OUTDIR/extended_selector_risk_diagnostics.csv

Quick checks:
  column -s, -t < $OUTDIR/extended_selector_policy_summary.csv | less -S
  grep ',28' $OUTDIR/extended_selector_policy_by_threshold.csv | column -s, -t | less -S
  column -s, -t < $OUTDIR/extended_selector_risk_diagnostics.csv | less -S
EOF
