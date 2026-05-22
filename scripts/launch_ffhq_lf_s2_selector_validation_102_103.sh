#!/usr/bin/env bash
set -euo pipefail

# Validation run for the LF/S2 seed tie-break selector using a different seed pair.
#
# This wraps launch_ffhq_lf_s2_selector.sh with the validation seed pair 102,103
# and a separate output root so the validation does not mix with the 100,101 run.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_lf_s2_selector_validation_102_103.sh
#
# Optional overrides:
#   GPU=1 SKIP_LPIPS=1 bash scripts/launch_ffhq_lf_s2_selector_validation_102_103.sh

export SEEDS=${SEEDS:-102,103}
export MAX_IMAGES=${MAX_IMAGES:-25}
export SKIP_LPIPS=${SKIP_LPIPS:-1}
export SEED_TIE_THRESHOLD=${SEED_TIE_THRESHOLD:-0.00005}
export OUTROOT=${OUTROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_lf_s2_selector_tiebreak_validation_102_103}

bash scripts/launch_ffhq_lf_s2_selector.sh
