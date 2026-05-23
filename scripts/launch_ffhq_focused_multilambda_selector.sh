#!/usr/bin/env bash
set -euo pipefail

# Focused multi-lambda selector experiment.
#
# Runs LF plus S2 lambdas on the focused hard/guard subset with diagnostic traces,
# analyzes the traces, then applies a multi-config selector.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_focused_multilambda_selector.sh
#
# Optional overrides:
#   GPU=0 SEEDS=102,103 LAMBDAS="0.005 0.02 0.05" bash scripts/launch_ffhq_focused_multilambda_selector.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}

OUTROOT=${OUTROOT:-$ROOT/np_ffhq_focused_multilambda_selector}
GPU=${GPU:-0}
SEEDS=${SEEDS:-102,103}
FOCUS_IMAGES=${FOCUS_IMAGES:-00005,00014,00007,00009,00018,00028,00034}
LAMBDAS=${LAMBDAS:-"0.005 0.02 0.05"}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
LATE_START=${LATE_START:-300}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
OVERSAMPLE=${OVERSAMPLE:-2}
SELECTOR_STAT=${SELECTOR_STAT:-post_winner_lf_mse_mean}

cd "$REPO"
mkdir -p "$OUTROOT/logs"
LOG="$OUTROOT/logs/focused_multilambda_selector_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] GPU=$GPU log=$LOG"
(
  set -euo pipefail

  common_args=(
    --data_root "$DATA_ROOT"
    --image_list_file "$SPLIT"
    --outdir "$OUTROOT"
    --guided_model_path "$GUIDED_MODEL"
    --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR"
    --variants np_canonical
    --seeds "$SEEDS"
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
    --select_images "$FOCUS_IMAGES"
    --skip_lpips
    --log_every 100
  )

  echo "[run] LF baseline"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
    "${common_args[@]}" \
    --tag ml_lf \
    --score_mode lf \
    --score_reg_lambda 0.0 \
    --score_reg_lambda_schedule constant \
    --noise_memory_k 0

  for lam in $LAMBDAS; do
    tag="ml_s2_lam${lam//./p}"
    echo "[run] S2 lambda=$lam tag=$tag"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      "${common_args[@]}" \
      --tag "$tag" \
      --score_mode prev_l2 \
      --score_reg_lambda "$lam" \
      --score_reg_lambda_schedule pre_projection_only \
      --noise_memory_k 0
  done

  echo "[analyze] diagnostic trace summary"
  python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
    --root "$OUTROOT" \
    --out_prefix "$OUTROOT/focused_multilambda_diag"

  echo "[select] multi-config selector stat=$SELECTOR_STAT"
  python scripts/apply_multiconfig_selector_to_diagnostic.py \
    --root_or_trace "$OUTROOT/focused_multilambda_diag_run_trace_summary.csv" \
    --out_prefix "$OUTROOT/focused_multilambda_${SELECTOR_STAT}" \
    --selector_stat "$SELECTOR_STAT"

  echo "[done] outputs under $OUTROOT"
) > "$LOG" 2>&1 &

PID=$!
echo "pid=$PID"
cat <<EOF

Output root:
  $OUTROOT

Monitor:
  tail -f "$LOG"
  watch -n 5 nvidia-smi

Expected files:
  $OUTROOT/focused_multilambda_diag_config_summary.csv
  $OUTROOT/focused_multilambda_diag_image_bestofk.csv
  $OUTROOT/focused_multilambda_diag_run_trace_summary.csv
  $OUTROOT/focused_multilambda_${SELECTOR_STAT}_selected_summary.csv
  $OUTROOT/focused_multilambda_${SELECTOR_STAT}_selected_image_level.csv
  $OUTROOT/focused_multilambda_${SELECTOR_STAT}_image_diagnostics.csv
EOF
