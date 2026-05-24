#!/usr/bin/env bash
set -euo pipefail

# Combined offline selector-policy sweep over multiple hard-trace roots.
#
# This is the next validation step after the 00013 extended selector sweep.
# It asks whether the promoted 00013-fixing policies still behave well when
# evaluated jointly with the remaining-hard traces and older hard-set traces.
#
# Usage from repo root on PAC:
#   TRACE_ROOTS="/path/to/root1 /path/to/root2 /path/to/root3" \
#   bash scripts/run_extended_selector_policy_sweep_combined_hard.sh
#
# If TRACE_ROOTS is omitted, the script tries to collect newest roots from:
#   np_ffhq_00013_margin_recovery_*
#   np_ffhq_remaining_hard_recovery_grid_*
#   np_ffhq_hard_image_reliability_ablation_*

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
TRACE_ROOTS=${TRACE_ROOTS:-}
OUTDIR=${OUTDIR:-$ROOT/combined_hard_extended_selector_sweep_$STAMP}

SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}
PSNR_KEY=${PSNR_KEY:-raw_psnr}
LF_RESID_KEY=${LF_RESID_KEY:-raw_noisy_lowfreq_mag_l2}
FULL_RESID_KEY=${FULL_RESID_KEY:-raw_noisy_mag_l2}
THRESHOLDS=${THRESHOLDS:-25,28,30}

cd "$REPO"
mkdir -p "$OUTDIR"

if [[ -z "$TRACE_ROOTS" ]]; then
  roots=()
  newest_00013=$(ls -td "$ROOT"/np_ffhq_00013_margin_recovery_* 2>/dev/null | head -n 1 || true)
  newest_remaining=$(ls -td "$ROOT"/np_ffhq_remaining_hard_recovery_grid_* 2>/dev/null | head -n 1 || true)
  newest_hard=$(ls -td "$ROOT"/np_ffhq_hard_image_reliability_ablation_* 2>/dev/null | head -n 1 || true)
  [[ -n "$newest_00013" ]] && roots+=("$newest_00013")
  [[ -n "$newest_remaining" ]] && roots+=("$newest_remaining")
  [[ -n "$newest_hard" ]] && roots+=("$newest_hard")
else
  # shellcheck disable=SC2206
  roots=($TRACE_ROOTS)
fi

if [[ ${#roots[@]} -eq 0 ]]; then
  echo "No trace roots found. Pass TRACE_ROOTS='root1 root2 ...'." >&2
  exit 1
fi

cat <<EOF
[combined-hard-selector-sweep]
  OUTDIR = $OUTDIR
  roots:
EOF
printf '    %s\n' "${roots[@]}"

python scripts/simulate_selector_policy_variants_extended.py \
  --roots_or_traces "${roots[@]}" \
  --outdir "$OUTDIR" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key "$PSNR_KEY" \
  --lf_resid_key "$LF_RESID_KEY" \
  --full_resid_key "$FULL_RESID_KEY" \
  --thresholds "$THRESHOLDS" \
  --dedupe

cat <<EOF

Combined extended selector sweep complete.

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
