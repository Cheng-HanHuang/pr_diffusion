#!/usr/bin/env bash
set -euo pipefail

# GPU validation run for the promoted selector/candidate-pool idea.
#
# This is not another blind search.  It validates a compact branch pool suggested
# by the hard-image studies:
#   - stable-safe branch for >=25 dB: lambda=0.15, proj_start=350, soft=8
#   - sharp 00013 branch: lambda=0.15, proj_start=450, soft=8
#   - robust sharp/mid branches: lambda=0.10/0.12, proj_start=450, soft=5
#   - 00027-like mid branch: lambda=0.03/0.05, proj_start=400/450, soft=5
#   - LF guard branch at proj_start=350
#
# After running, it applies the normal trace analysis, current selector, and the
# extended selector-policy sweep.
#
# Usage from repo root on PAC:
#   bash scripts/launch_ffhq_selector_validation_pool.sh
#
# Optional smoke test:
#   HARD_IMAGES="00013,00027,00028" SEEDS=132,133 bash scripts/launch_ffhq_selector_validation_pool.sh
#
# Full split / all images in SPLIT:
#   HARD_IMAGES="" bash scripts/launch_ffhq_selector_validation_pool.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_selector_validation_pool_$STAMP}

# Use the full hard set by default.  Pass HARD_IMAGES="" to use every image in
# SPLIT, e.g. the full FFHQ-25 split.
HARD_IMAGES=${HARD_IMAGES-00000,00005,00007,00013,00027,00028,00034}
SEEDS=${SEEDS:-132,133,134,135}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MAX_PARALLEL=${MAX_PARALLEL:-4}

SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
OVERSAMPLE=${OVERSAMPLE:-2}
SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}
SKIP_EXISTING=${SKIP_EXISTING:-1}

cd "$REPO"
mkdir -p "$OUTROOT/logs"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
declare -i job_idx=0

acquire_slot() {
  while true; do
    local running
    running=$(jobs -rp | wc -l | tr -d ' ')
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 15
  done
}

launch_cfg() {
  local tag="$1"; shift
  local gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
  local outdir="$OUTROOT/$tag"
  local log_file="$OUTROOT/logs/$tag.log"
  if [[ "$SKIP_EXISTING" == "1" ]] && compgen -G "$outdir/**/final_metrics.csv" > /dev/null; then
    echo "[skip existing] $tag"
    return 0
  fi
  echo "[launch] gpu=$gpu tag=$tag log=$log_file"
  acquire_slot

  select_args=()
  if [[ -n "$HARD_IMAGES" ]]; then
    select_args=(--select_images "$HARD_IMAGES")
  fi

  (
    set -euo pipefail
    mkdir -p "$outdir"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      --data_root "$DATA_ROOT" \
      --image_list_file "$SPLIT" \
      "${select_args[@]}" \
      --outdir "$outdir" \
      --tag "$tag" \
      --guided_model_path "$GUIDED_MODEL" \
      --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
      --variants np_canonical \
      --seeds "$SEEDS" \
      --np_steps "$NP_STEPS" \
      --score_radius "$SCORE_RADIUS" \
      --proj_radius "$PROJ_RADIUS" \
      --oversample_values "$OVERSAMPLE" \
      --measurement_noise_values "$SIGMA" \
      --clip_noisy_magnitude \
      --alignments raw,rot180,resolve \
      --skip_lpips \
      --log_every 100 \
      "$@"
  ) > "$log_file" 2>&1 &
  job_idx+=1
}

cat <<EOF
[selector-validation-pool]
  OUTROOT     = $OUTROOT
  SPLIT       = $SPLIT
  HARD_IMAGES = ${HARD_IMAGES:-<all images in split>}
  SEEDS       = $SEEDS
  GPU_LIST    = $GPU_LIST
EOF

# LF guard.
launch_cfg hard_lf_ps350_soft5_hard1 \
  --late_start 350 --soft_candidates 5 --hard_candidates 1 \
  --score_mode lf --score_reg_lambda 0.0 --score_reg_lambda_schedule constant --noise_memory_k 0

# Stable-safe fallback branch.
launch_cfg hard_s2_lam0p15_ps350_soft8_hard1 \
  --late_start 350 --soft_candidates 8 --hard_candidates 1 \
  --score_mode prev_l2 --score_reg_lambda 0.15 --score_reg_lambda_schedule pre_projection_only --noise_memory_k 0

# 00013 sharp branch found by margin recovery.
launch_cfg hard_s2_lam0p15_ps450_soft8_hard1 \
  --late_start 450 --soft_candidates 8 --hard_candidates 1 \
  --score_mode prev_l2 --score_reg_lambda 0.15 --score_reg_lambda_schedule pre_projection_only --noise_memory_k 0

# Sharp branches around high-lambda 00013 region.
for lam in 0.10 0.12; do
  tag_lam=${lam//./p}
  launch_cfg hard_s2_lam${tag_lam}_ps450_soft5_hard1 \
    --late_start 450 --soft_candidates 5 --hard_candidates 1 \
    --score_mode prev_l2 --score_reg_lambda "$lam" --score_reg_lambda_schedule pre_projection_only --noise_memory_k 0
done

# Mid-lambda guard branches for 00027-like behavior.
for spec in "0.03 400" "0.05 400" "0.05 450"; do
  read -r lam ps <<< "$spec"
  tag_lam=${lam//./p}
  launch_cfg hard_s2_lam${tag_lam}_ps${ps}_soft5_hard1 \
    --late_start "$ps" --soft_candidates 5 --hard_candidates 1 \
    --score_mode prev_l2 --score_reg_lambda "$lam" --score_reg_lambda_schedule pre_projection_only --noise_memory_k 0
done

wait

echo "[analysis] merging selector validation traces"
python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
  --root "$OUTROOT" \
  --out_prefix "$OUTROOT/selector_validation_diag"

python scripts/apply_multiconfig_selector_to_diagnostic.py \
  --root_or_trace "$OUTROOT/selector_validation_diag_run_trace_summary.csv" \
  --out_prefix "$OUTROOT/selector_validation_${SELECTOR_STAT}" \
  --selector_stat "$SELECTOR_STAT" \
  --alignments raw,rot180,resolve

python scripts/simulate_selector_policy_variants_extended.py \
  --roots_or_traces "$OUTROOT/selector_validation_diag_run_trace_summary.csv" \
  --outdir "$OUTROOT/extended_selector_policy_sweep" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key raw_psnr \
  --lf_resid_key raw_noisy_lowfreq_mag_l2 \
  --full_resid_key raw_noisy_mag_l2 \
  --thresholds 25,28,30 \
  --dedupe

python scripts/analyze_reliability_from_traces.py \
  --roots_or_traces "$OUTROOT/selector_validation_diag_run_trace_summary.csv" \
  --outdir "$OUTROOT/reliability_analysis" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key raw_psnr \
  --thresholds 25,28,30 \
  --primary_threshold 28 \
  --hard_images "${HARD_IMAGES:-ALL_SPLIT_IMAGES}" \
  --n_seed_orders 200 \
  --adaptive_start_k 2 \
  --adaptive_add_k 2 \
  --adaptive_max_k 4 \
  --dedupe

python scripts/analyze_reliability_failure_taxonomy.py \
  --roots_or_traces "$OUTROOT/selector_validation_diag_run_trace_summary.csv" \
  --outdir "$OUTROOT/reliability_failure_taxonomy" \
  --thresholds 25,28,30 \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key raw_psnr

cat <<EOF

Selector validation pool complete.

Output root:
  $OUTROOT

Main files:
  $OUTROOT/selector_validation_diag_run_trace_summary.csv
  $OUTROOT/selector_validation_${SELECTOR_STAT}_selected_summary.csv
  $OUTROOT/selector_validation_${SELECTOR_STAT}_image_diagnostics.csv
  $OUTROOT/extended_selector_policy_sweep/extended_selector_policy_summary.csv
  $OUTROOT/extended_selector_policy_sweep/extended_selector_policy_by_threshold.csv
  $OUTROOT/extended_selector_policy_sweep/extended_selector_risk_diagnostics.csv
  $OUTROOT/reliability_analysis/hard_image_candidate_availability.csv
  $OUTROOT/reliability_analysis/adaptive_policy_summary.csv
  $OUTROOT/reliability_failure_taxonomy/reliability_failure_taxonomy_by_image.csv
  $OUTROOT/reliability_failure_taxonomy/reliability_failure_taxonomy_counts.csv
EOF
