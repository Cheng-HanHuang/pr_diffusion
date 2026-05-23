#!/usr/bin/env bash
set -euo pipefail

# Full-25 multi-lambda validation batch.
#
# Runs:
#   GPU0: seeds 100,101,102,103
#   GPU1: seeds 104,105
#   GPU2: optional seeds 106,107
#   GPU3: optional seeds 108,109
#
# Each run uses configs LF + S2 lambdas 0.005, 0.02, 0.05 with full diagnostic traces,
# then applies multi-config selector and slice analysis.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_multilambda_validation_batch.sh
#
# Optional:
#   RUN_106107=0 RUN_108109=0 bash scripts/launch_ffhq_multilambda_validation_batch.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}

BATCH_ROOT=${BATCH_ROOT:-$ROOT/np_ffhq_multilambda_validation_batch}
LAMBDAS=${LAMBDAS:-"0.005 0.02 0.05"}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
LATE_START=${LATE_START:-300}
OVERSAMPLE=${OVERSAMPLE:-2}
SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}

RUN_4SEED=${RUN_4SEED:-1}
RUN_104105=${RUN_104105:-1}
RUN_106107=${RUN_106107:-1}
RUN_108109=${RUN_108109:-0}

cd "$REPO"
mkdir -p "$BATCH_ROOT/logs"

run_full25_group() {
  local gpu="$1"
  local seeds="$2"
  local label="$3"
  local outroot="$BATCH_ROOT/$label"
  local log_file="$BATCH_ROOT/logs/${label}.log"
  mkdir -p "$outroot"
  echo "[launch] label=$label gpu=$gpu seeds=$seeds log=$log_file"
  (
    set -euo pipefail
    common_args=(
      --data_root "$DATA_ROOT"
      --image_list_file "$SPLIT"
      --outdir "$outroot"
      --guided_model_path "$GUIDED_MODEL"
      --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR"
      --variants np_canonical
      --seeds "$seeds"
      --np_steps "$NP_STEPS"
      --late_start "$LATE_START"
      --soft_candidates "$SOFT"
      --hard_candidates "$HARD"
      --score_radius "$SCORE_RADIUS"
      --proj_radius "$PROJ_RADIUS"
      --oversample_values "$OVERSAMPLE"
      --measurement_noise_values "$SIGMA"
      --clip_noisy_magnitude
      --alignments raw,rot180,resolve
      --skip_lpips
      --log_every 100
    )

    echo "[$label] LF baseline"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      "${common_args[@]}" \
      --tag "${label}_lf" \
      --score_mode lf \
      --score_reg_lambda 0.0 \
      --score_reg_lambda_schedule constant \
      --noise_memory_k 0

    for lam in $LAMBDAS; do
      tag="${label}_s2_lam${lam//./p}"
      echo "[$label] S2 lambda=$lam tag=$tag"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
        "${common_args[@]}" \
        --tag "$tag" \
        --score_mode prev_l2 \
        --score_reg_lambda "$lam" \
        --score_reg_lambda_schedule pre_projection_only \
        --noise_memory_k 0
    done

    python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
      --root "$outroot" \
      --out_prefix "$outroot/${label}_diag"

    python scripts/apply_multiconfig_selector_to_diagnostic.py \
      --root_or_trace "$outroot/${label}_diag_run_trace_summary.csv" \
      --out_prefix "$outroot/${label}_${SELECTOR_STAT}" \
      --selector_stat "$SELECTOR_STAT"

    # Slice by individual seeds and reduced lambda pools.
    IFS=',' read -ra seed_arr <<< "$seeds"
    seed_args=()
    seed_args+=("all:${seeds}")
    for s in "${seed_arr[@]}"; do
      seed_args+=("s${s}:${s}")
    done
    python scripts/apply_multiconfig_selector_slices.py \
      --root_or_trace "$outroot/${label}_diag_run_trace_summary.csv" \
      --out_prefix "$outroot/${label}_slices" \
      --selector_stat "$SELECTOR_STAT" \
      --seed_sets "${seed_args[@]}" \
      --config_sets \
        "all_configs" \
        "no005:${label}_lf,${label}_s2_lam0p02,${label}_s2_lam0p05" \
        "no002:${label}_lf,${label}_s2_lam0p005,${label}_s2_lam0p05" \
        "no0050:${label}_lf,${label}_s2_lam0p005,${label}_s2_lam0p02" \
        "lf_lam005_lam002:${label}_lf,${label}_s2_lam0p005,${label}_s2_lam0p02" \
        "lf_lam005_lam0050:${label}_lf,${label}_s2_lam0p005,${label}_s2_lam0p05"

    echo "[$label] done"
  ) > "$log_file" 2>&1 &
  echo "  pid=$!"
}

if [[ "$RUN_4SEED" == "1" ]]; then
  run_full25_group 0 "100,101,102,103" "full25_s100_101_102_103_sr06_start300"
fi
if [[ "$RUN_104105" == "1" ]]; then
  run_full25_group 1 "104,105" "full25_s104_105_sr06_start300"
fi
if [[ "$RUN_106107" == "1" ]]; then
  run_full25_group 2 "106,107" "full25_s106_107_sr06_start300"
fi
if [[ "$RUN_108109" == "1" ]]; then
  run_full25_group 3 "108,109" "full25_s108_109_sr06_start300"
fi

cat <<EOF

Launched multi-lambda validation batch.

Batch root:
  $BATCH_ROOT

Monitor:
  tail -f $BATCH_ROOT/logs/*.log
  watch -n 5 nvidia-smi

Key files per run folder:
  <label>_${SELECTOR_STAT}_selected_summary.csv
  <label>_${SELECTOR_STAT}_selected_image_level.csv
  <label>_${SELECTOR_STAT}_image_diagnostics.csv
  <label>_slices_selected_summary.csv
  <label>_slices_image_diagnostics.csv
EOF
