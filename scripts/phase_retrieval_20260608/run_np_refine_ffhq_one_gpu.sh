#!/usr/bin/env bash
set -euo pipefail

# Run NP + anchored measurement refinement on FFHQ.
# Usage:
#   bash scripts/phase_retrieval_20260608/run_np_refine_ffhq_one_gpu.sh 0 full25_s100_103
# Optional environment overrides:
#   SEEDS="100,101,102,103" NOISES="0,0.01,0.05,0.08,0.10" MAX_IMAGES=25 REFINE_STEPS=100

GPU_ID="${1:-0}"
TAG="${2:-ffhq_full25_s100_103_np_refine}"

REPO="${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}"
OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608}"
DATA_ROOT="${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}"
IMAGE_LIST_FILE="${IMAGE_LIST_FILE:-$OUT_ROOT/splits/ffhq_available25.txt}"
GUIDED_MODEL_PATH="${GUIDED_MODEL_PATH:-/egr/research-pac/huang248/models/ffhq_10m.pt}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}"

SEEDS="${SEEDS:-100,101,102,103}"
NOISES="${NOISES:-0,0.01,0.05,0.08,0.10}"
MAX_IMAGES="${MAX_IMAGES:-25}"
NP_STEPS="${NP_STEPS:-1000}"
PROJ_START="${PROJ_START:-300}"
SOFT_K="${SOFT_K:-5}"
HARD_K="${HARD_K:-1}"
SCORE_RADIUS="${SCORE_RADIUS:-0.6}"
PROJ_RADIUS="${PROJ_RADIUS:-0.2}"
S2_LAMBDA="${S2_LAMBDA:-0.01}"
REFINE_STEPS="${REFINE_STEPS:-100}"
REFINE_LR="${REFINE_LR:-0.01}"
REFINE_ANCHORS="${REFINE_ANCHORS:-0,0.01,0.05}"
REFINE_MAG_RADIUS="${REFINE_MAG_RADIUS:-full}"
SKIP_LPIPS="${SKIP_LPIPS:-1}"

cd "$REPO"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prdiff_ffhq

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

bash scripts/phase_retrieval_20260608/prepare_phase_retrieval_20260608.sh
OUTDIR="$OUT_ROOT/np_refine_ffhq/$TAG"
mkdir -p "$OUTDIR"

LPIPS_FLAG=()
if [[ "$SKIP_LPIPS" == "1" ]]; then
  LPIPS_FLAG=(--skip_lpips)
fi

echo "[run] GPU=$GPU_ID TAG=$TAG OUTDIR=$OUTDIR"
echo "[run] seeds=$SEEDS noises=$NOISES max_images=$MAX_IMAGES refine_steps=$REFINE_STEPS anchors=$REFINE_ANCHORS"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/pr_external_difffpr_np_refine_benchmark.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$IMAGE_LIST_FILE" \
  --outdir "$OUTDIR" \
  --guided_model_path "$GUIDED_MODEL_PATH" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset difffpr_ffhq_10m \
  --seeds "$SEEDS" \
  --np_steps "$NP_STEPS" \
  --late_start "$PROJ_START" \
  --soft_candidates "$SOFT_K" \
  --hard_candidates "$HARD_K" \
  --score_radius "$SCORE_RADIUS" \
  --proj_radius "$PROJ_RADIUS" \
  --s2_lambda "$S2_LAMBDA" \
  --s2_lambda_schedule pre_projection_only \
  --oversample_values 2 \
  --measurement_noise_values "$NOISES" \
  --clip_noisy_magnitude \
  --alignments raw,rot180,resolve \
  --max_images "$MAX_IMAGES" \
  --refine_steps "$REFINE_STEPS" \
  --refine_lr "$REFINE_LR" \
  --refine_anchor_weights "$REFINE_ANCHORS" \
  --refine_mag_radius "$REFINE_MAG_RADIUS" \
  --log_every 100 \
  "${LPIPS_FLAG[@]}"
