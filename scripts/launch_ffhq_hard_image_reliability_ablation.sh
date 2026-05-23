#!/usr/bin/env bash
set -euo pipefail

# Targeted hard-image candidate-generation ablation for reliability study.
#
# Goal:
#   Increase candidate-generation success probability p_x on recurring hard
#   images, rather than merely improving average PSNR.
#
# Default hard images:
#   00028, 00005, 00013, 00034, 00027, 00007, 00000
#
# Usage from repo root on PAC:
#   bash scripts/launch_ffhq_hard_image_reliability_ablation.sh
#
# Useful smaller dry run:
#   SEEDS=110,111 LAMBDAS="0.01 0.1" PROJ_STARTS="350" \
#   SOFT_VALUES="5" HARD_VALUES="1" bash scripts/launch_ffhq_hard_image_reliability_ablation.sh

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
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_hard_image_reliability_ablation_$STAMP}
HARD_IMAGES=${HARD_IMAGES:-00028,00005,00013,00034,00027,00007,00000}

# Keep this default moderate.  Increase later only after checking first results.
SEEDS=${SEEDS:-110,111,112,113}
LAMBDAS=${LAMBDAS:-"0.01 0.02 0.05 0.1"}
PROJ_STARTS=${PROJ_STARTS:-"300 350 400"}
SOFT_VALUES=${SOFT_VALUES:-"5 8"}
HARD_VALUES=${HARD_VALUES:-"1 2"}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
OVERSAMPLE=${OVERSAMPLE:-2}
SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MAX_PARALLEL=${MAX_PARALLEL:-4}
SKIP_EXISTING=${SKIP_EXISTING:-1}

cd "$REPO"
mkdir -p "$OUTROOT/logs"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

declare -i job_idx=0

aquire_slot() {
  while true; do
    local running
    running=$(jobs -rp | wc -l | tr -d ' ')
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 15
  done
}

run_one_config() {
  local gpu="$1"
  local tag="$2"
  local outdir="$3"
  local log_file="$4"
  shift 4
  local extra_args=("$@")

  if [[ "$SKIP_EXISTING" == "1" ]] && compgen -G "$outdir/**/final_metrics.csv" > /dev/null; then
    echo "[skip existing] $tag"
    return 0
  fi

  echo "[launch] gpu=$gpu tag=$tag log=$log_file"
  (
    set -euo pipefail
    mkdir -p "$outdir"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      --data_root "$DATA_ROOT" \
      --image_list_file "$SPLIT" \
      --select_images "$HARD_IMAGES" \
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
      "${extra_args[@]}"
  ) > "$log_file" 2>&1 &
}

# Baseline LF for each structural setting.  This tells us whether proj_start,
# soft, and hard changes help even without the S2 regularizer.
for proj_start in $PROJ_STARTS; do
  for soft in $SOFT_VALUES; do
    for hard in $HARD_VALUES; do
      gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
      tag="hard_lf_ps${proj_start}_soft${soft}_hard${hard}"
      outdir="$OUTROOT/$tag"
      log_file="$OUTROOT/logs/$tag.log"
      aquire_slot
      run_one_config "$gpu" "$tag" "$outdir" "$log_file" \
        --late_start "$proj_start" \
        --soft_candidates "$soft" \
        --hard_candidates "$hard" \
        --score_mode lf \
        --score_reg_lambda 0.0 \
        --score_reg_lambda_schedule constant \
        --noise_memory_k 0
      job_idx+=1
    done
  done
done

# S2 lambda grid.
for proj_start in $PROJ_STARTS; do
  for soft in $SOFT_VALUES; do
    for hard in $HARD_VALUES; do
      for lam in $LAMBDAS; do
        lam_tag=${lam//./p}
        gpu=${GPUS[$((job_idx % ${#GPUS[@]}))]}
        tag="hard_s2_lam${lam_tag}_ps${proj_start}_soft${soft}_hard${hard}"
        outdir="$OUTROOT/$tag"
        log_file="$OUTROOT/logs/$tag.log"
        aquire_slot
        run_one_config "$gpu" "$tag" "$outdir" "$log_file" \
          --late_start "$proj_start" \
          --soft_candidates "$soft" \
          --hard_candidates "$hard" \
          --score_mode prev_l2 \
          --score_reg_lambda "$lam" \
          --score_reg_lambda_schedule pre_projection_only \
          --noise_memory_k 0
        job_idx+=1
      done
    done
  done
done

wait

echo "[analysis] merging hard-image ablation traces"
python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
  --root "$OUTROOT" \
  --out_prefix "$OUTROOT/hard_ablation_diag"

python scripts/apply_multiconfig_selector_to_diagnostic.py \
  --root_or_trace "$OUTROOT/hard_ablation_diag_run_trace_summary.csv" \
  --out_prefix "$OUTROOT/hard_ablation_${SELECTOR_STAT}" \
  --selector_stat "$SELECTOR_STAT" \
  --alignments raw,rot180,resolve

python scripts/analyze_reliability_from_traces.py \
  --roots_or_traces "$OUTROOT/hard_ablation_diag_run_trace_summary.csv" \
  --outdir "$OUTROOT/reliability_analysis" \
  --selector_stat "$SELECTOR_STAT" \
  --psnr_key raw_psnr \
  --thresholds 25,28,30 \
  --primary_threshold 25 \
  --hard_images "$HARD_IMAGES" \
  --n_seed_orders 200 \
  --adaptive_start_k 2 \
  --adaptive_add_k 2 \
  --adaptive_max_k 4 \
  --dedupe

cat <<EOF

Hard-image reliability ablation complete.

Output root:
  $OUTROOT

Main analysis files:
  $OUTROOT/hard_ablation_diag_run_trace_summary.csv
  $OUTROOT/hard_ablation_${SELECTOR_STAT}_selected_summary.csv
  $OUTROOT/hard_ablation_${SELECTOR_STAT}_image_diagnostics.csv
  $OUTROOT/reliability_analysis/hard_image_candidate_availability.csv
  $OUTROOT/reliability_analysis/selector_calibration_global.csv
  $OUTROOT/reliability_analysis/adaptive_policy_summary.csv

Monitor logs while running:
  tail -f $OUTROOT/logs/*.log
EOF
