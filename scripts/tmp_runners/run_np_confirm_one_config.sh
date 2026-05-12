#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
SCORE_LABEL="${2:?score label required}"
PROJ_LABEL="${3:?proj label required}"
PROJ_START="${4:?proj_start required}"
SOFT_K="${5:?soft_k required}"
HARD_K="${6:?hard_k required}"
TAG="${7:?tag required}"

radius_value() {
  if [[ "$1" == "full" ]]; then
    echo "0.72"
  else
    echo "$1"
  fi
}

SCORE_RADIUS="$(radius_value "$SCORE_LABEL")"
PROJ_RADIUS="$(radius_value "$PROJ_LABEL")"

cd /egr/research-pac/huang248/pr_diffusion_repo
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prdiff_ffhq

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
IMAGE_LIST_FILE=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt
GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR

OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_confirm_top8/${TAG}

mkdir -p "$OUTDIR"

echo "[confirm] GPU=${GPU_ID}"
echo "[confirm] tag=${TAG}"
echo "[confirm] score=${SCORE_LABEL} (${SCORE_RADIUS}) proj=${PROJ_LABEL} (${PROJ_RADIUS}) start=${PROJ_START} soft=${SOFT_K} hard=${HARD_K}"
echo "[confirm] OUTDIR=${OUTDIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/pr_external_difffpr_np_guided_benchmark.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$IMAGE_LIST_FILE" \
  --outdir "$OUTDIR" \
  --guided_model_path "$GUIDED_MODEL_PATH" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset difffpr_ffhq_10m \
  --variants np_canonical \
  --seeds 100,101,102,103 \
  --np_steps 1000 \
  --late_start "$PROJ_START" \
  --soft_candidates "$SOFT_K" \
  --hard_candidates "$HARD_K" \
  --score_radius "$SCORE_RADIUS" \
  --proj_radius "$PROJ_RADIUS" \
  --oversample_values 2 \
  --measurement_noise_values 0.05 \
  --alignments rot180 \
  --clip_noisy_magnitude \
  --skip_lpips \
  --fast_eval \
  --max_images 25 \
  --log_every 100
