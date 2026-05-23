#!/usr/bin/env bash
set -euo pipefail

# Four-GPU next-batch launcher after LF/S2 selector validation.
#
# Experiments:
#   GPU0: full FFHQ-25 LF/S2 tie-break selector with four seeds 100,101,102,103.
#   GPU1: focused S2 projection-start diagnostic on known failures + guard images.
#   GPU2: focused S2 lambda diagnostic on known failures + guard images.
#   GPU3: focused memory fallback diagnostic on known failures + guard images.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_next_batch_4gpu.sh
#
# Optional overrides:
#   ROOT=... REPO=... GUIDED_MODEL=... bash scripts/launch_ffhq_next_batch_4gpu.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}

BATCH_ROOT=${BATCH_ROOT:-$ROOT/np_ffhq_next_batch_20260523}
FOCUS_IMAGES=${FOCUS_IMAGES:-00005,00014,00007,00009,00018,00028,00034}
FOCUS_SEEDS=${FOCUS_SEEDS:-102,103}
FULL_SEEDS=${FULL_SEEDS:-100,101,102,103}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
OVERSAMPLE=${OVERSAMPLE:-2}
TIE_THRESHOLD=${TIE_THRESHOLD:-0.00005}
SKIP_LPIPS=${SKIP_LPIPS:-1}

cd "$REPO"
mkdir -p "$BATCH_ROOT/logs"

common_diag_args=(
  --data_root "$DATA_ROOT"
  --image_list_file "$SPLIT"
  --guided_model_path "$GUIDED_MODEL"
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR"
  --variants np_canonical
  --seeds "$FOCUS_SEEDS"
  --np_steps "$NP_STEPS"
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

# -----------------------------------------------------------------------------
# GPU0: full 4-seed LF/S2 selector control.
# -----------------------------------------------------------------------------
OUT0="$BATCH_ROOT/gpu0_lf_s2_selector_4seeds"
mkdir -p "$OUT0/logs"
LOG0="$BATCH_ROOT/logs/gpu0_lf_s2_selector_4seeds.log"
echo "[launch gpu0] full 4-seed LF/S2 selector -> $LOG0"
(
  set -euo pipefail
  GPU=0 \
  OUTROOT="$OUT0" \
  SEEDS="$FULL_SEEDS" \
  MAX_IMAGES=25 \
  SKIP_LPIPS="$SKIP_LPIPS" \
  SEED_TIE_THRESHOLD="$TIE_THRESHOLD" \
  bash scripts/launch_ffhq_lf_s2_selector.sh
) > "$LOG0" 2>&1 &
echo "  pid=$!"

# -----------------------------------------------------------------------------
# GPU1: focused S2 projection-start diagnostic.
# -----------------------------------------------------------------------------
OUT1="$BATCH_ROOT/gpu1_focused_s2_projstart"
mkdir -p "$OUT1"
LOG1="$BATCH_ROOT/logs/gpu1_focused_s2_projstart.log"
echo "[launch gpu1] focused S2 proj_start diagnostic -> $LOG1"
(
  set -euo pipefail
  for start in 200 300 400 500; do
    tag="g1_s2_start${start}_lam001"
    echo "[gpu1] running $tag"
    CUDA_VISIBLE_DEVICES=1 python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      "${common_diag_args[@]}" \
      --outdir "$OUT1" \
      --tag "$tag" \
      --late_start "$start" \
      --score_mode prev_l2 \
      --score_reg_lambda 0.01 \
      --score_reg_lambda_schedule pre_projection_only \
      --noise_memory_k 0
  done
  python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
    --root "$OUT1" \
    --out_prefix "$OUT1/gpu1_projstart_diag"
) > "$LOG1" 2>&1 &
echo "  pid=$!"

# -----------------------------------------------------------------------------
# GPU2: focused S2 lambda diagnostic.
# -----------------------------------------------------------------------------
OUT2="$BATCH_ROOT/gpu2_focused_s2_lambda"
mkdir -p "$OUT2"
LOG2="$BATCH_ROOT/logs/gpu2_focused_s2_lambda.log"
echo "[launch gpu2] focused S2 lambda diagnostic -> $LOG2"
(
  set -euo pipefail
  for lam in 0.005 0.01 0.02 0.05; do
    tag="g2_s2_start300_lam${lam//./p}"
    echo "[gpu2] running $tag"
    CUDA_VISIBLE_DEVICES=2 python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
      "${common_diag_args[@]}" \
      --outdir "$OUT2" \
      --tag "$tag" \
      --late_start 300 \
      --score_mode prev_l2 \
      --score_reg_lambda "$lam" \
      --score_reg_lambda_schedule pre_projection_only \
      --noise_memory_k 0
  done
  python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
    --root "$OUT2" \
    --out_prefix "$OUT2/gpu2_lambda_diag"
) > "$LOG2" 2>&1 &
echo "  pid=$!"

# -----------------------------------------------------------------------------
# GPU3: focused memory/fallback diagnostic.
# -----------------------------------------------------------------------------
OUT3="$BATCH_ROOT/gpu3_focused_memory_fallback"
mkdir -p "$OUT3"
LOG3="$BATCH_ROOT/logs/gpu3_focused_memory_fallback.log"
echo "[launch gpu3] focused memory fallback diagnostic -> $LOG3"
(
  set -euo pipefail

  echo "[gpu3] running memory hard2 LF fallback"
  CUDA_VISIBLE_DEVICES=3 python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
    "${common_diag_args[@]}" \
    --outdir "$OUT3" \
    --tag g3_memory_k1_hard2_lf \
    --late_start 300 \
    --hard_candidates 2 \
    --score_mode lf \
    --score_reg_lambda 0.0 \
    --score_reg_lambda_schedule constant \
    --noise_memory_k 1

  echo "[gpu3] running memory hard2 S2 fallback"
  CUDA_VISIBLE_DEVICES=3 python scripts/pr_external_difffpr_np_guided_diagnostic_trace.py \
    "${common_diag_args[@]}" \
    --outdir "$OUT3" \
    --tag g3_memory_k1_hard2_s2_lam001 \
    --late_start 300 \
    --hard_candidates 2 \
    --score_mode prev_l2 \
    --score_reg_lambda 0.01 \
    --score_reg_lambda_schedule pre_projection_only \
    --noise_memory_k 1

  python scripts/analyze_ffhq_diagnostic_trace_nopandas.py \
    --root "$OUT3" \
    --out_prefix "$OUT3/gpu3_memory_diag"
) > "$LOG3" 2>&1 &
echo "  pid=$!"

cat <<EOF

Launched next-batch experiments.

Batch root:
  $BATCH_ROOT

Logs:
  tail -f $BATCH_ROOT/logs/*.log

GPU monitor:
  watch -n 5 nvidia-smi

Expected high-level outputs:
  GPU0 selector:
    $OUT0/lf_s2_selector_*/selected_tiebreak_thr${TIE_THRESHOLD}_summary.csv
    $OUT0/lf_s2_selector_*/selected_tiebreak_thr${TIE_THRESHOLD}_image_level.csv
    $OUT0/lf_s2_selector_*/run_level.csv

  GPU1 proj_start diagnostic:
    $OUT1/gpu1_projstart_diag_config_summary.csv
    $OUT1/gpu1_projstart_diag_image_bestofk.csv
    $OUT1/gpu1_projstart_diag_run_trace_summary.csv

  GPU2 lambda diagnostic:
    $OUT2/gpu2_lambda_diag_config_summary.csv
    $OUT2/gpu2_lambda_diag_image_bestofk.csv
    $OUT2/gpu2_lambda_diag_run_trace_summary.csv

  GPU3 memory diagnostic:
    $OUT3/gpu3_memory_diag_config_summary.csv
    $OUT3/gpu3_memory_diag_image_bestofk.csv
    $OUT3/gpu3_memory_diag_run_trace_summary.csv
EOF
