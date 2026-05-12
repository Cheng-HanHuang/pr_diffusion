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

DATA_ROOT=/egr/research-pac/huang248/data/imagenet/imagenet256_val
IMAGE_LIST_FILE=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/imagenet_available25.txt
MODEL_PATH=/egr/research-pac/huang248/models/imagenet256.pt
GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR

OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_imagenet25_noise_sweep_B_fast/${TAG}

mkdir -p "$OUTDIR"

echo "[np-imagenet25] GPU=${GPU_ID}"
echo "[np-imagenet25] noise_std=${NOISE_STD}"
echo "[np-imagenet25] tag=${TAG}"
echo "[np-imagenet25] OUTDIR=${OUTDIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/pr_external_difffpr_np_guided_benchmark.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$IMAGE_LIST_FILE" \
  --outdir "$OUTDIR" \
  --guided_model_path "$MODEL_PATH" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset difffpr_imagenet256 \
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
