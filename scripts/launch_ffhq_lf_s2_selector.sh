#!/usr/bin/env bash
set -euo pipefail

# Launch the lightweight LF/S2 selector experiment.
#
# Usage from repo root:
#   bash scripts/launch_ffhq_lf_s2_selector.sh
#
# Optional overrides:
#   GPU=0 SEEDS=100,101 MAX_IMAGES=25 SKIP_LPIPS=1 bash scripts/launch_ffhq_lf_s2_selector.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi
SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_lf_s2_selector}
GPU=${GPU:-0}
SEEDS=${SEEDS:-100,101}
MAX_IMAGES=${MAX_IMAGES:-25}
SKIP_LPIPS=${SKIP_LPIPS:-1}
NP_STEPS=${NP_STEPS:-1000}
LATE_START=${LATE_START:-300}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
S2_LAMBDA=${S2_LAMBDA:-0.01}
SIGMA=${SIGMA:-0.05}
OVERSAMPLE=${OVERSAMPLE:-2}

cd "$REPO"
mkdir -p "$OUTROOT/logs"

EXTRA_ARGS=()
if [[ "$SKIP_LPIPS" == "1" ]]; then
  EXTRA_ARGS+=(--skip_lpips)
fi
if [[ -n "$MAX_IMAGES" ]]; then
  EXTRA_ARGS+=(--max_images "$MAX_IMAGES")
fi

LOG="$OUTROOT/logs/lf_s2_selector_$(date +%Y%m%d_%H%M%S).log"
echo "[launch] GPU=$GPU log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup python scripts/pr_external_difffpr_np_guided_lf_s2_selector.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$SPLIT" \
  --outdir "$OUTROOT" \
  --guided_model_path "$GUIDED_MODEL" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --seeds "$SEEDS" \
  --np_steps "$NP_STEPS" \
  --late_start "$LATE_START" \
  --soft_candidates "$SOFT" \
  --hard_candidates "$HARD" \
  --score_radius "$SCORE_RADIUS" \
  --proj_radius "$PROJ_RADIUS" \
  --s2_lambda "$S2_LAMBDA" \
  --s2_lambda_schedule pre_projection_only \
  --oversample_values "$OVERSAMPLE" \
  --measurement_noise_values "$SIGMA" \
  --clip_noisy_magnitude \
  --alignments raw,rot180,resolve \
  --log_every 100 \
  "${EXTRA_ARGS[@]}" \
  > "$LOG" 2>&1 &

PID=$!
echo "pid=$PID"
cat <<EOF

Output root:
  $OUTROOT

Monitor:
  tail -f "$LOG"
  watch -n 5 nvidia-smi

Expected files inside a timestamped lf_s2_selector_* directory:
  configs.csv
  run_level.csv
  selected_image_level.csv
  selected_summary.csv
EOF
