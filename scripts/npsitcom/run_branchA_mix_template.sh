#!/usr/bin/env bash
set -euo pipefail

# Branch A mixer template. Set SITCOM_CSV to a standardized SITCOM run-level CSV.
# Example:
#   SITCOM_CSV=/path/to/run_level_standardized.csv bash scripts/npsitcom/run_branchA_mix_template.sh

OUT=${OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610}
NP_CSV=${NP_CSV:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608/np_selector_ffhq/selector_full25_s100_103/lf_s2_selector_latest/run_level.csv}
SITCOM_CSV=${SITCOM_CSV:-}
TAG=${TAG:-branchA_np_sitcom_mix}
ALIGNMENT=${ALIGNMENT:-resolve}

if [[ -z "$SITCOM_CSV" ]]; then
  echo "Set SITCOM_CSV=/path/to/standardized_sitcom_run_level.csv" >&2
  exit 2
fi
if [[ ! -f "$NP_CSV" ]]; then
  echo "Missing NP_CSV=$NP_CSV" >&2
  echo "If the timestamp folder differs, set NP_CSV explicitly." >&2
  exit 2
fi
if [[ ! -f "$SITCOM_CSV" ]]; then
  echo "Missing SITCOM_CSV=$SITCOM_CSV" >&2
  exit 2
fi

RUN_OUT=$OUT/branchA_mix/$TAG
mkdir -p "$RUN_OUT"
python scripts/npsitcom/mix_select_candidates.py \
  --candidate_csv "np:$NP_CSV" \
  --candidate_csv "sitcom:$SITCOM_CSV" \
  --alignment "$ALIGNMENT" \
  --outdir "$RUN_OUT"
