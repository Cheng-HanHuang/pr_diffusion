#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
NOISE_STD="${2:?noise std required}"
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

OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_noise_sweep_B_fast/${TAG}

mkdir -p "$OUTDIR"

echo "[noise-sweep] GPU=${GPU_ID}"
echo "[noise-sweep] noise_std=${NOISE_STD}"
echo "[noise-sweep] tag=${TAG}"
echo "[noise-sweep] config: score=0.6 proj=0.2 start=300 soft=5 hard=1"
echo "[noise-sweep] OUTDIR=${OUTDIR}"

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
  --late_start 300 \
  --soft_candidates 5 \
  --hard_candidates 1 \
  --score_radius 0.6 \
  --proj_radius 0.2 \
  --oversample_values 2 \
  --measurement_noise_values "$NOISE_STD" \
  --alignments raw,rot180,resolve \
  --clip_noisy_magnitude \
  --max_images 25 \
  --log_every 100
