#!/usr/bin/env bash
set -euo pipefail

# Extra-GPU overnight launcher for multi-lambda selector ablations.
#
# Intended use: while the focused baseline multi-lambda selector is running on
# one GPU, use the other GPUs for high-value follow-up tests.
#
# Default experiments:
#   GPU1: full FFHQ-25 multi-lambda selector with seeds 102,103.
#   GPU2: focused multi-lambda selector with score_radius=0.4.
#   GPU3: focused multi-lambda selector with proj_start=200.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_multilambda_extra_gpus.sh
#
# Optional overrides:
#   GPU_FULL=0 GPU_SCORE=2 GPU_START=3 bash scripts/launch_ffhq_multilambda_extra_gpus.sh
#   RUN_FULL=0 bash scripts/launch_ffhq_multilambda_extra_gpus.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}

BATCH_ROOT=${BATCH_ROOT:-$ROOT/np_ffhq_multilambda_extra_overnight}
FOCUS_IMAGES=${FOCUS_IMAGES:-00005,00014,00007,00009,00018,00028,00034}
SEEDS=${SEEDS:-102,103}
LAMBDAS=${LAMBDAS:-"0.005 0.02 0.05"}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
OVERSAMPLE=${OVERSAMPLE:-2}
SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}

# Default GPU assignment assumes the baseline focused run is already occupying GPU0.
GPU_FULL=${GPU_FULL:-1}
GPU_SCORE=${GPU_SCORE:-2}
GPU_START=${GPU_START:-3}

RUN_FULL=${RUN_FULL:-1}
RUN_SCORE_ABLATION=${RUN_SCORE_ABLATION:-1}
RUN_START_ABLATION=${RUN_START_ABLATION:-1}

cd "$REPO"
mkdir -p "$BATCH_ROOT/logs"

run_multilambda_group() {
  local gpu="$1"
  local outroot="$2"
  local label="$3"
  local select_images="$4"
  local score_radius="$5"
  local late_start="$6"
  local log_file="$7"

  mkdir -p "$outroot"
  (
    set -euo pipefail

    common_args=(
      --data_root "$DATA_ROOT"
      --image_list_file "$SPLIT"
      --outdir "$outroot"
      --guided_model_path "$GUIDED_MODEL"
      --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR"
      --variants np_canonical
      --seeds "$SEEDS"
      --np_steps "$NP_STEPS"
      --late_start "$late_start"
      --soft_candidates "$SOFT"
      --hard_candidates "$HARD"
      --score_radius "$score_radius"
      --proj_radius "$PROJ_RADIUS"
      --oversample_values "$OVERSAMPLE"
      --measurement_noise_values "$SIGMA"
      --clip_noisy_magnitude
      --alignments raw,rot180,resolve
      --skip_lpips
      --log_every 100
    )

    if [[ -n "$select_images" ]]; then
      common_args+=(--select_images "$select_images")
    fi

    echo "[$label] LF baseline | gpu=$gpu score_radius=$score_radius late_start=$late_start select_images=${select_images:-FULL25}"
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

    echo "[$label] analyze diagnostic traces"
    python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
      --root "$outroot" \
      --out_prefix "$outroot/${label}_diag"

    echo "[$label] apply multi-config selector"
    python scripts/apply_multiconfig_selector_to_diagnostic.py \
      --root_or_trace "$outroot/${label}_diag_run_trace_summary.csv" \
      --out_prefix "$outroot/${label}_${SELECTOR_STAT}" \
      --selector_stat "$SELECTOR_STAT"

    echo "[$label] done"
  ) > "$log_file" 2>&1 &
  echo "  pid=$! label=$label gpu=$gpu log=$log_file"
}

if [[ "$RUN_FULL" == "1" ]]; then
  OUT="$BATCH_ROOT/gpu${GPU_FULL}_full25_s102_103_sr06_start300"
  LOG="$BATCH_ROOT/logs/gpu${GPU_FULL}_full25_s102_103_sr06_start300.log"
  echo "[launch] full FFHQ-25 multi-lambda selector on GPU$GPU_FULL"
  run_multilambda_group "$GPU_FULL" "$OUT" "full25_sr06_start300" "" "0.6" "300" "$LOG"
fi

if [[ "$RUN_SCORE_ABLATION" == "1" ]]; then
  OUT="$BATCH_ROOT/gpu${GPU_SCORE}_focused_s102_103_sr04_start300"
  LOG="$BATCH_ROOT/logs/gpu${GPU_SCORE}_focused_s102_103_sr04_start300.log"
  echo "[launch] focused score_radius=0.4 ablation on GPU$GPU_SCORE"
  run_multilambda_group "$GPU_SCORE" "$OUT" "focused_sr04_start300" "$FOCUS_IMAGES" "0.4" "300" "$LOG"
fi

if [[ "$RUN_START_ABLATION" == "1" ]]; then
  OUT="$BATCH_ROOT/gpu${GPU_START}_focused_s102_103_sr06_start200"
  LOG="$BATCH_ROOT/logs/gpu${GPU_START}_focused_s102_103_sr06_start200.log"
  echo "[launch] focused proj_start=200 ablation on GPU$GPU_START"
  run_multilambda_group "$GPU_START" "$OUT" "focused_sr06_start200" "$FOCUS_IMAGES" "0.6" "200" "$LOG"
fi

cat <<EOF

Launched extra multi-lambda GPU experiments.

Batch root:
  $BATCH_ROOT

Monitor all logs:
  tail -f $BATCH_ROOT/logs/*.log
  watch -n 5 nvidia-smi

Expected selected summaries:
  $BATCH_ROOT/gpu${GPU_FULL}_full25_s102_103_sr06_start300/full25_sr06_start300_${SELECTOR_STAT}_selected_summary.csv
  $BATCH_ROOT/gpu${GPU_SCORE}_focused_s102_103_sr04_start300/focused_sr04_start300_${SELECTOR_STAT}_selected_summary.csv
  $BATCH_ROOT/gpu${GPU_START}_focused_s102_103_sr06_start200/focused_sr06_start200_${SELECTOR_STAT}_selected_summary.csv
EOF
