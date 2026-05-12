#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
LAMBDA="${2:?lambda required}"
TAG="${3:?tag required}"

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

OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_s2_lambda_sweep/${TAG}

mkdir -p "$OUTDIR"

echo "[S2-lambda] GPU=${GPU_ID}"
echo "[S2-lambda] lambda=${LAMBDA}"
echo "[S2-lambda] tag=${TAG}"
echo "[S2-lambda] config: prev_l2, score_radius=0.6, proj_radius=0.2, start=300, soft=5, hard=1, sigma=0.05, seeds=100,101"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/pr_external_difffpr_np_guided_benchmark.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$IMAGE_LIST_FILE" \
  --outdir "$OUTDIR" \
  --guided_model_path "$GUIDED_MODEL_PATH" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset difffpr_ffhq_10m \
  --variants np_canonical \
  --seeds 100,101 \
  --np_steps 1000 \
  --late_start 300 \
  --soft_candidates 5 \
  --hard_candidates 1 \
  --score_radius 0.6 \
  --proj_radius 0.2 \
  --score_mode prev_l2 \
  --score_reg_lambda "$LAMBDA" \
  --score_huber_delta 0.05 \
  --oversample_values 2 \
  --measurement_noise_values 0.05 \
  --alignments raw,rot180 \
  --clip_noisy_magnitude \
  --skip_lpips \
  --fast_eval \
  --max_images 25 \
  --log_every 100
